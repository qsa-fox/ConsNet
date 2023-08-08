import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def dlog_activation(x):
	beta = 1.0
	return torch.log(1 + torch.clip(x, min=0) / beta) - torch.log(1 - torch.clip(x, max=0) / beta)

def sqrt_activation(x):
	epsilon = 1e-5
	return torch.sqrt(torch.clip(x, min=0) + epsilon) - torch.sqrt(-torch.clip(x, max=0) + epsilon)

def square_activation(x):
	return torch.square(torch.clip(x, min=0)) - torch.square(torch.clip(x, max=0))

def gaussian(x):
	return torch.exp(-x*x)

def sym_gaussian(x):
	return torch.exp(-x)*(x>0) - torch.exp(x)*(x<0)

activation_dict = {
"relu": torch.nn.ReLU(),
"leaky_relu": torch.nn.LeakyReLU(),
"neg_relu": torch.nn.Hardtanh(float('-inf'), 0),
"relu6": torch.nn.ReLU6(),
"tanh": torch.nn.Tanh(),
"hardtanh": torch.nn.Hardtanh(),
"sigmoid": torch.nn.Sigmoid(),
"dlog": dlog_activation,
"identity": torch.nn.Identity(),
"sqrt": sqrt_activation,
"square": square_activation,
"gaussian": gaussian,
"elu": torch.nn.ELU(),
}


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


def fanin_init(tensor, scale=1):
	size = tensor.size()
	if len(size) == 2:
		fan_in = size[0]
	elif len(size) > 2:
		fan_in = np.prod(size[1:])
	else:
		raise Exception("Shape must be have dimension at least 2.")
	bound = scale / np.sqrt(fan_in)
	return tensor.data.uniform_(-bound, bound)


class ParallelizedLayerMLP(nn.Module):

	def __init__(
			self,
			ensemble_size,
			input_dim,
			output_dim,
			w_std_value=1.0,
			b_init_value=0.0
	):
		super().__init__()

		# approximation to truncated normal of 2 stds
		w_init = torch.randn((ensemble_size, input_dim, output_dim))
		w_init = torch.fmod(w_init, 2) * w_std_value
		self.W = nn.Parameter(w_init, requires_grad=True)

		# constant initialization
		b_init = torch.zeros((ensemble_size, 1, output_dim)).float()
		b_init += b_init_value
		self.b = nn.Parameter(b_init, requires_grad=True)

	def forward(self, x):
		# assumes x is 3D: (ensemble_size, batch_size, dimension)
		return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

	def __init__(
			self,
			ensemble_size,
			hidden_sizes,
			input_size,
			output_size,
			init_w=3e-3,
			hidden_init=fanin_init,
			w_scale=1,
			b_init_value=0.1,
			layer_norm=None,
			batch_norm=False,
			final_init_scale=None,
			activation='relu',
	):
		super().__init__()

		self.ensemble_size = ensemble_size
		self.input_size = input_size
		self.output_size = output_size
		self.elites = [i for i in range(self.ensemble_size)]

		self.sampler = np.random.default_rng()

		# self.hidden_activation = torch.nn.ReLU()
		# self.hidden_activation = torch.nn.Tanh()
		# self.hidden_activation = torch.nn.Hardtanh()
		# self.hidden_activation = torch.nn.ReLU6()
		self.hidden_activation = activation_dict[activation]
		
		self.output_activation = torch.nn.Identity()
		# self.output_activation = activation_dict['sqrt']

		self.layer_norm = layer_norm

		self.fcs = []

		if batch_norm:
			raise NotImplementedError

		in_size = input_size
		for i, next_size in enumerate(hidden_sizes):
			fc = ParallelizedLayerMLP(
				ensemble_size=ensemble_size,
				input_dim=in_size,
				output_dim=next_size,
			)
			for j in self.elites:
				hidden_init(fc.W[j], w_scale)
				fc.b[j].data.fill_(b_init_value)
			self.__setattr__('fc%d' % i, fc)
			self.fcs.append(fc)
			in_size = next_size

		self.last_fc = ParallelizedLayerMLP(
			ensemble_size=ensemble_size,
			input_dim=in_size,
			output_dim=output_size,
		)
		if final_init_scale is None:
			self.last_fc.W.data.uniform_(-init_w, init_w)
			self.last_fc.b.data.uniform_(-init_w, init_w)
		else:
			for j in self.elites:
				torch.nn.init.orthogonal_(self.last_fc.W[j], final_init_scale)
				self.last_fc.b[j].data.fill_(0)

	def forward(self, *inputs, **kwargs):
		self.h_log = []
		flat_inputs = torch.cat(inputs, dim=-1)

		state_dim = inputs[0].shape[-1]

		dim = len(flat_inputs.shape)
		# repeat h to make amenable to parallelization
		# if dim = 3, then we probably already did this somewhere else
		# (e.g. bootstrapping in training optimization)
		if dim < 3:
			flat_inputs = flat_inputs.unsqueeze(0)
			if dim == 1:
				flat_inputs = flat_inputs.unsqueeze(0)
			flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)

		# input normalization
		h = flat_inputs

		# standard feedforward network
		# for _, fc in enumerate(self.fcs):
		# 	h = fc(h)
		# 	h = self.hidden_activation(h)
		# 	if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
		# 		h = self.layer_norm(h)
		# 	self.h_log.append(h)
		for i in range(len(self.fcs)):
			fc = self.fcs[i]
			h = fc(h)
			if i == len(self.fcs) - 1:
				# h = gaussian(h)
				h = self.hidden_activation(h)
			else:
				h = self.hidden_activation(h)
			if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
				h = self.layer_norm(h)
			self.h_log.append(h)
		preactivation = self.last_fc(h)
		output = self.output_activation(preactivation)
  
		# Q shift
		# output = output - 100

		# if original dim was 1D, squeeze the extra created layer
		if dim == 1:
			output = output.squeeze(1)

		# output is (ensemble_size, batch_size, output_size)
		return output

	def sample(self, *inputs):
		preds = self.forward(*inputs)

		return torch.min(preds, dim=0)[0]

	# return torch.max(preds, dim=0)[0]

	def fit_input_stats(self, data, mask=None):
		raise NotImplementedError


class TD3(object):
	def __init__(
			self,
			state_dim,
			action_dim,
			max_action,
			discount=0.99,
			tau=0.005,
			policy_noise=0.2,
			noise_clip=0.5,
			critic_layers=3,
			policy_freq=2,
			n_ensemble=5,
			critic_activation='relu',
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-5)

		self.critic_ensemble = ParallelizedEnsembleFlattenMLP(ensemble_size=n_ensemble,
															hidden_sizes=[256] * critic_layers,
															input_size=state_dim + action_dim,
															output_size=1, layer_norm=None,
															activation=critic_activation).to(device)
		self.critic_target_ensemble = copy.deepcopy(self.critic_ensemble)
		self.critic_optimizer = torch.optim.Adam(self.critic_ensemble.parameters(), lr=3e-4)
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.n_ensemble = n_ensemble
		self.total_it = 0
		self.action_dim = action_dim

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def imitation(self, replay_buffer, batch_size=256):
		self.total_it += 1
		# Sample replay buffer
		state, action, next_state, reward, not_done, next_action = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Compute the target Q value
			target_Q_ensemble = self.critic_target_ensemble(next_state, next_action)
			# target_Q_ensemble_min = torch.min(target_Q_ensemble, dim=0)[0]
			target_Q_ensemble_min = torch.mean(target_Q_ensemble, dim=0)
			target_Q = reward + not_done * self.discount * target_Q_ensemble_min

		# Get current Q estimates
		current_Q_ensemble = self.critic_ensemble(state, action)
		critic_loss = torch.square(current_Q_ensemble - target_Q).sum(0).mean()

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		for param, target_param in zip(self.critic_ensemble.parameters(),
									   self.critic_target_ensemble.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


		return {'Q': current_Q_ensemble.mean().item()}

	def plot_3d(critic, state, action, delta_max=2, step=0.1, n_subplots=4, act_clip=False):
		import matplotlib.pyplot as plt
		from matplotlib import cm
		from matplotlib.ticker import LinearLocator
		fig = plt.figure(figsize=plt.figaspect(0.8))
		state_, action_ = state.clone(), action.clone()
		for i in range(n_subplots):
			if n_subplots > 1:
				ax = fig.add_subplot(int(np.ceil(n_subplots / 2.0)), 2, i+1, projection='3d')
			else:
				ax = fig.add_subplot(1, 1, i + 1, projection='3d')
			directions = torch.randn(2, action_.shape[-1]).cuda()
			direct_x = directions[0] / torch.norm(directions[0])
			direct_y = directions[1] / torch.norm(directions[1])
			delta = torch.FloatTensor(torch.range(-delta_max, delta_max, step)).cuda()
			x = direct_x.unsqueeze(-1) * delta.unsqueeze(0)
			y = direct_y.unsqueeze(-1) * delta.unsqueeze(0)
			xy = action_.unsqueeze(-1).unsqueeze(-1) + x.unsqueeze(-1) + y.unsqueeze(1)
			xy = xy.permute(1, 2, 0)
			if act_clip:
				xy = xy.clip(-1, 1)
			xy_flatten = xy.reshape(-1, xy.shape[-1])
			state_tmp = state_.unsqueeze(0).unsqueeze(0).repeat(len(delta), len(delta), 1)
			state_flatten = state_tmp.reshape(-1, state_tmp.shape[-1])
			z = critic(state_flatten, xy_flatten)
			z = z.mean(0).squeeze().reshape(len(delta), len(delta))

			X = delta.cpu().numpy()
			Y = delta.cpu().numpy()
			Z = z.detach().cpu().numpy()
			X, Y = np.meshgrid(X, Y)

			# plt.subplot(2, 2, i+1)
			# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
			surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
									linewidth=0, antialiased=False)
			ax.set_zlim(Z.min(), Z.max())
			ax.zaxis.set_major_locator(LinearLocator(10))
			ax.zaxis.set_major_formatter('{x:.02f}')
			fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.show()
   
	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1
		# Sample replay buffer
		state, action, next_state, reward, not_done, _ = replay_buffer.sample(batch_size)
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
					torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
					self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q_ensemble = self.critic_target_ensemble(next_state, next_action)
			target_Q_ensemble_min = torch.min(target_Q_ensemble, dim=0)[0]
			target_Q = reward + not_done * self.discount * target_Q_ensemble_min
		# Get current Q estimates
		current_Q_ensemble = self.critic_ensemble(state, action)
		critic_loss = torch.square(current_Q_ensemble - target_Q).sum(0).mean()

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		
		if self.total_it % 1000 == 0:
			wandb.log({'Q': current_Q_ensemble.mean().item(),
					   'critic_loss': critic_loss.item(),
					   })
		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			pi = self.actor(state)
			Q_ensemble = self.critic_ensemble(state, pi)
			Q_ensemble_min = torch.min(Q_ensemble, dim=0)[0]
			actor_loss = -Q_ensemble_min.mean()

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.critic_ensemble.parameters(),
										   self.critic_target_ensemble.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				
		return {'Q': current_Q_ensemble[0].mean().item(),
				'critic_loss': critic_loss.item(),}

	def save(self, filename):
		torch.save(self.critic_ensemble.state_dict(), filename + "_critic_ensemble")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		self.critic_ensemble.load_state_dict(torch.load(filename + "_critic_ensemble"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target_ensemble = copy.deepcopy(self.critic_ensemble)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

	def save_all(self, filename):
		torch.save(self, filename + '_all.pth')

	def load_all(self, filename):
		return torch.load(self, filename + '_all.pth')
