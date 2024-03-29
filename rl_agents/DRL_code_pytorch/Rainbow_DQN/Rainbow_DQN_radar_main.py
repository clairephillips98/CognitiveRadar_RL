from torch.utils.tensorboard import SummaryWriter
from rl_agents.DRL_code_pytorch.Rainbow_DQN.replay_buffer import *
from rl_agents.DRL_code_pytorch.Rainbow_DQN.rainbow_dqn import DQN
import argparse
from radar_env.radar_gymnasium import RadarEnv
from rl_agents.calculate_stats import stats
import os
class Runner:
    def __init__(self, args, env_name, number,seed):
        self.args = args
        self.env_name = "Radar_Env_Radar"
        self.number = number
        self.seed = seed
        self.blur_radius = self.args.blur_radius
        print(self.blur_radius)
        if self.args.cdl > 0:
            self.env_name += '_common_destination_{}_odds'.format(self.args.cdl)
        self.env = RadarEnv(seed=seed,blur_radius=self.blur_radius,
                            scale=self.args.scale, sigma=self.args.blur_sigma,
                            common_destination=self.args.common_destination, cdl=self.args.cdl)
        self.env_evaluate = RadarEnv(seed=seed,blur_radius=self.blur_radius,
                                     scale=self.args.scale,sigma=self.args.blur_sigma,
                                     common_destination=self.args.common_destination, cdl=self.args.cdl)
        self.args.state_dim = self.env.observation_space['observation'].shape
        if type(self.args.state_dim) == int:
            self.args.state_dim = [self.args.state_dim]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = self.env._max_episode_steps  # Maximum number of steps per episode
        print("env={}".format(self.env_name))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))
        print("scale={}".format(self.args.scale))
        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)
        self.algorithm = 'DQN'
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'Rainbow_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += '_Double'
            if args.use_dueling:
                self.algorithm += '_Dueling'
            if args.use_noisy:
                self.algorithm += '_Noisy'
            if args.use_per:
                self.algorithm += '_PER'
            if args.use_n_steps:
                self.algorithm += "_N_steps"
        if args.load_model is True:
            if os.path.isfile('models/DQN/net_{}_env_{}_blur_radius_{}_scale_{}_blur_simga_{}.pt'.format(self.algorithm, self.env_name, self.blur_radius, self.scale,self.args.blur_sigma)):
                self.agent.net.load_state_dict(torch.load('models/DQN/net_{}_env_{}_blur_radius_{}_scale_{}_blur_sigma_{}.pt'.format(self.algorithm, self.env_name, self.blur_radius,self.args.scale,self.args.blur_sigma)))
                self.agent.target_net.load_state_dict(torch.load('models/DQN/target_net_{}_env_{}_blur_radius_{}_scale_{}_blur_sigma_{}.pt'.format(self.algorithm, self.env_name,self.blur_radius,self.args.scale,self.args.blur_sigma)))
        self.writer = SummaryWriter(log_dir='runs/DQN/{}_env_{}_number_{}_seed_{}_blur_radius_{}_scale_{}_blur_sigma_{}'.format(self.algorithm, self.env_name, number, seed, self.blur_radius,self.args.scale,self.args.blur_sigma))

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
        if args.use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def run(self, ):
        self.evaluate_policy()
        while self.total_steps < self.args.max_train_steps:
            state = self.env.reset()[0]
            done = False
            episode_steps = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=self.epsilon)
                next_state, reward, done, _,_ = self.env.step(action)
                episode_steps += 1
                self.total_steps += 1

                if not self.args.use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # terminal means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != self.args.episode_limit:
                    if self.env_name == 'LunarLander-v2':
                        if reward <= -100: reward = -1  # good for LunarLander
                    terminal = True
                else:
                    terminal = False

                self.replay_buffer.store_transition(state, action, reward, next_state, terminal, done)  # Store the transition
                state = next_state

                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)

                if self.total_steps % self.args.evaluate_freq == 0:
                    self.evaluate_policy()
                if (self.total_steps/10) % self.args.evaluate_freq == 0:
                    self.save_models()
        self.save_models()
        # Save reward
        np.save('data_train/DQN/{}_env_{}_number_{}_seed_{}_blur_radius_{}.npy'.format(self.algorithm, self.env_name, self.number, self.seed, self.blur_radius), np.array(self.evaluate_rewards))

    def save_models(self):
        torch.save(self.agent.net.state_dict(),'models/DQN/net_{}_env_{}_blur_radius_{}.pt'.format(self.algorithm, self.env_name,self.blur_radius))
        torch.save(self.agent.target_net.state_dict(), 'models/DQN/target_net_{}_env_{}_blur_radius_{}.pt'.format(self.algorithm, self.env_name,self.blur_radius))

    def evaluate_policy(self, ):
        evaluate_reward = 0
        radar_stats = stats()
        self.agent.net.eval()
        for _ in range(self.args.evaluate_times):
            state = self.env_evaluate.reset()[0]
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=0)
                next_state, reward, done, _,_ = self.env_evaluate.step(action)
                episode_reward += reward
                state = next_state
            radar_stats.add_stats(self.env_evaluate.info_analysis())
            evaluate_reward += episode_reward
        self.agent.net.train()
        analysis = radar_stats.stats_analysis()
        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward, self.epsilon))
        self.writer.add_scalar('step_rewards', evaluate_reward, global_step=self.total_steps)
        self.writer.add_scalar('step_time_to_first_view', analysis['avg_time_til_first_view'], global_step=self.total_steps)
        self.writer.add_scalar('target_view_rate_to_velocity_corr', analysis['views_vel_corr'], global_step=self.total_steps)
        self.writer.add_scalar('world_view_avg_loss', analysis['avg_world_loss'], global_step=self.total_steps)


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
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e6), help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=int, default=1, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")
    parser.add_argument("--load_model", type=int, default=1, help="Whether to pick up the last model")
    parser.add_argument("--use_double", type=int, default=1, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=int, default=1, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=int, default=1, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=int, default=1, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=int, default=1, help="Whether to use n_steps Q-learning")
    parser.add_argument("--blur_radius", type=int, default=1, help="size of the radius of the gaussian filter applied to previous views")
    parser.add_argument("--scale", type=int, default=50, help="factor by which the space is scaled down")
    parser.add_argument("--blur_sigma", type=float, default=0.5, help="guassian blur sigma")
    parser.add_argument("--common_destination", type=list, default=[-380,-380], help="a common location for targets come from and go to")
    parser.add_argument("--cdl", type=float, default=0.0, help="how many targets go to location")

    args = parser.parse_args()
    env_index = 1
    for seed in [0, 10, 100]:
        runner = Runner(args=args, env_name="cognitive_radar", number=1, seed=seed)
        runner.run()
