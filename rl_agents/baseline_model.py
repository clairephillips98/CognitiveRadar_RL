import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import argparse
from radar_env.radar_gymnasium import RadarEnv
import random
from rl_agents.calculate_stats import radar_stats,radar_stats_analysis
class Runner:
    def __init__(self, args, env_name, number,seed):
        self.args = args
        self.env_name = "Radar_Env"
        self.number = number
        self.seed = seed
        random.seed(seed)
        self.env = RadarEnv(seed)
        self.env_evaluate = RadarEnv(seed)
        self.args.state_dim = self.env.observation_space['observation'].shape
        if type(self.args.state_dim) == int:
            self.args.state_dim = [self.args.state_dim]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = self.env._max_episode_steps  # Maximum number of steps per episode
        print("env={}".format(self.env_name))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        self.writer = SummaryWriter(log_dir='runs/Baseline_Model/{}_env_{}_number_{}_seed_{}'.format('baseline', self.env_name, number, seed))

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training

    def run(self):
        while self.total_steps < (self.args.max_train_steps/self.args.evaluate_freq):
            self.evaluate_policy()
            self.total_steps += 1
    def evaluate_policy(self):
        action=3
        evaluate_reward = 0
        analysis_info = {}
        for _ in range(self.args.evaluate_times):
            state = self.env_evaluate.reset()[0]
            done = False
            episode_reward = 0
            episode_sum_view_time = 0
            while not done:
                action = (action + self.env.game.radars[0].num_states+ 1) % self.env.action_size
                next_state, reward, done, _, _ = self.env_evaluate.step(action)
                episode_reward += reward
                state = next_state
            evaluate_reward += episode_reward
            analysis_info = radar_stats(analysis_info, self.env_evaluate.info_analysis())
        analysis = radar_stats_analysis(analysis_info)
        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward, "None"))
        self.writer.add_scalar('step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        self.writer.add_scalar('step_time_to_first_view_{}'.format(self.env_name), analysis['avg_time_til_first_view'], global_step=self.total_steps)
        self.writer.add_scalar('target_view_rate_to_velocity_corr_{}'.format(self.env_name), analysis['views_vel_corr'], global_step=self.total_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5), help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=True, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")

    args = parser.parse_args()

    env_names = ['CartPole-v1', 'LunarLander-v2']
    env_index = 1
    for seed in [0, 10, 100]:
        runner = Runner(args=args, env_name=env_names[env_index], number=1, seed=seed)
        runner.run()

