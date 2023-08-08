import numpy as np
import torch
import gym
import argparse
import os
import d4rl
from tqdm import tqdm
from time import time
import TD3_oodnet as TD3
from pathlib import Path
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# torch.set_float32_matmul_precision('high')

envs2 = {
    "halfcheetah-random-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
    "hopper-random-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
    "walker2d-random-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
    "halfcheetah-medium-v2":{'n_ensemble': [5], 'critic_activation': ['relu']},
    "hopper-medium-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
    "walker2d-medium-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
    "halfcheetah-medium-replay-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
    "hopper-medium-replay-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
    "walker2d-medium-replay-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
    "halfcheetah-medium-expert-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
    "hopper-medium-expert-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
    "walker2d-medium-expert-v2": {'n_ensemble': [5], 'critic_activation': ['relu']},
}

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
        )

    def sample2(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind
        )

    def convert_D4RL(self, dataset, r_scale=1.0, r_shift=0):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.next_action = dataset['next_actions']
        self.reward = dataset['rewards'].reshape(-1, 1) * r_scale + r_shift
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_D4RL_sparse(self, dataset, r_scale=1.0, r_shift=0):
        dataset = list(dataset)
        for seq in dataset:
            if seq['rewards'].sum() == 0:
                len_seq = len(seq['observations'])
                self.state[self.ptr:self.ptr+len_seq] = seq['observations'].copy()
                self.action[self.ptr:self.ptr+len_seq] = seq['actions'].copy()
                self.next_state[self.ptr:self.ptr+len_seq] = np.concatenate([seq['observations'][1:], seq['observations'][-1:]], axis=0).copy()
                self.next_action[self.ptr:self.ptr+len_seq] = np.concatenate([seq['actions'][1:], seq['actions'][-1:]], axis=0).copy()
                self.reward[self.ptr:self.ptr+len_seq] = seq['rewards'].copy().reshape(-1, 1) * r_scale + r_shift
                self.not_done[self.ptr:self.ptr+len_seq] = 1 - seq['terminals'].copy().reshape(-1, 1)
                self.ptr += len_seq
        self.size_no_reward = self.ptr
        for seq in dataset:
            if seq['rewards'].sum() != 0:
                len_seq = len(seq['observations'])
                self.state[self.ptr:self.ptr+len_seq] = seq['observations'].copy()
                self.action[self.ptr:self.ptr+len_seq] = seq['actions'].copy()
                self.next_state[self.ptr:self.ptr+len_seq] = np.concatenate([seq['observations'][1:], seq['observations'][-1:]], axis=0).copy()
                self.next_action[self.ptr:self.ptr+len_seq] = np.concatenate([seq['actions'][1:], seq['actions'][-1:]], axis=0).copy()
                self.reward[self.ptr:self.ptr+len_seq] = seq['rewards'].copy().reshape(-1, 1) * r_scale + r_shift
                self.not_done[self.ptr:self.ptr+len_seq] = 1 - seq['terminals'].copy().reshape(-1, 1)
                self.ptr += len_seq
        self.size_reward = self.ptr - self.size_no_reward
        self.size = self.ptr

    def sample_sparse(self, batch_size, reward_ratio=0.5):
        ind_no_reward = np.random.randint(0, self.size_no_reward, size=int(batch_size*(1-reward_ratio)))
        ind_reward = np.random.randint(self.size_no_reward, self.size, size=batch_size-len(ind_no_reward))
        ind = np.concatenate([ind_no_reward, ind_reward])

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
        )

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        Qs = []
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            Qs.append(policy.critic_ensemble(torch.FloatTensor(state).to('cuda'), torch.FloatTensor(action[None]).to(
                'cuda')).detach().cpu().numpy().squeeze())
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


def main(args):
    file_name = f"{args.env}_" \
                f"ensemble{args.n_ensemble}_seed{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./infos"):
        os.makedirs("./infos")
    if not os.path.exists("./models"):
        os.makedirs("./models")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "critic_layers": args.critic_layers,
        "n_ensemble": args.n_ensemble,
        "critic_activation": args.critic_activation,
    }
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env), r_scale=args.reward_scale, r_shift=args.reward_shift)
    # replay_buffer.convert_D4RL_sparse(d4rl.sequence_dataset(env), args.reward_scale, args.reward_shift)
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    # Initialize policy
    policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    evaluations = []
    info_log = []
    t0 = time()
    for t in (range(int(args.max_timesteps))):
        info = policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            info_log.append(info)
            eopch_time = (time() - t0) * 1000 / args.eval_freq
            print(f"Time steps: {t + 1}, epoch_time: {eopch_time:.2f}", info)
            score = eval_policy(policy, args.env, args.seed, mean, std)
            evaluations.append(score)
            wandb.log({'score': score})
            np.save(f"./results/{file_name}", evaluations)
            np.save(f"./infos/{file_name}", info_log)
            if args.save_model: policy.save(f"./models/{file_name}")
            t0 = time()
        if (t + 1) % int(5e5) == 0:
            torch.save(policy, f"./models/{file_name}_steps{t}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="TD3")  # Policy name
    parser.add_argument("--env", default="antmaze-large-play-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    # TD3
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--critic_layers", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--n_ensemble", default=10, type=int)
    parser.add_argument("--reward_scale", default=1, type=float)
    parser.add_argument("--reward_shift", default=0, type=float)
    args = parser.parse_args()

    import wandb
    '''relu-gaussian-identity-sweep2: layers2, actor_lr3e-5'''
    for seed in [0]:
        args.seed = seed
        for env, env_config in envs2.items():
            args.env = env
            for n_ensemble in env_config['n_ensemble']:
                args.n_ensemble = n_ensemble
                for critic_activation in env_config['critic_activation']:
                    args.critic_activation = critic_activation
                    run = wandb.init(project='ConsNet', reinit=True,
                                     settings=wandb.Settings(code_dir='.', log_internal='./null'), 
                                     group='relu-gaussian-identity-sweep2-seeds', mode='online', save_code=True)
                    wandb.config.update(args)
                    print(args)
                    main(args)
                    run.finish()
