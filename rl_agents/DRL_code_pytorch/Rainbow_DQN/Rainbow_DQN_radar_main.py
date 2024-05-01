import torch
from torch.utils.tensorboard import SummaryWriter
from rl_agents.DRL_code_pytorch.Rainbow_DQN.replay_buffer import *
from rl_agents.DRL_code_pytorch.Rainbow_DQN.rainbow_dqn import DQN
from rl_agents.DRL_code_pytorch.Rainbow_DQN.MARL_orchestrate import MARL_Double_Agent, MARL_Double_RB
from rl_agents.DRL_code_pytorch.Rainbow_DQN.baselines import *
import torchvision.transforms as T

import argparse
from radar_env.radar_gymnasium import RadarEnv
from rl_agents.calculate_stats import stats
from rl_agents.config import set_gpu_name
import os
from utils import action_unpack, action_repack
from random import randint
from math import floor
from functools import reduce

class Runner:
    def __init__(self, args, env_name, number,seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.blur_radius = self.args.blur_radius
        if self.args.cdl > 0:
            self.env_name += '_common_destination_{}_odds'.format(self.args.cdl)
        self.env = RadarEnv(seed=seed, args=self.args)
        self.env_evaluate = RadarEnv(seed=seed, args=self.args)
        self.args.state_dim = self.env.observation_space['observation'].shape
        if type(self.args.state_dim) == int:
            self.args.state_dim = [self.args.state_dim]
        self.args.action_dim = self.env.action_space.n if args.agents == 1 else int(self.env.action_space.n**(1/2)) # action dim is square root possible action space if 2 agents
        self.args.episode_limit = self.env._max_episode_steps  # Maximum number of steps per episode
        print("env={}".format(self.env_name))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))
        print("scale={}".format(self.args.scale))
        print(self.args.type_of_MARL)
        if self.args.type_of_MARL in ['some_shared_info', 'shared_targets_only']: # seperate rewards
            self.use_sep_rewards = True
        else:
            self.use_sep_rewards = False
        if self.args.type_of_MARL in ['some_shared_info', 'some_shared_info_shared_reward', 'shared_targets_only']: #s eperate state spaces
            self.replay_buffer = MARL_Double_RB(args)
        else:
            if args.use_per and args.use_n_steps:
                self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
            elif args.use_per:
                self.replay_buffer = Prioritized_ReplayBuffer(args)
            elif args.use_n_steps:
                self.replay_buffer = N_Steps_ReplayBuffer(args)
            else:
                self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args) if self.args.agents == 1 else MARL_Double_Agent(args) #create 2 agents in case of MARL
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
        if self.args.baseline == 1:
            self.algorithm = "baseline"
        elif self.args.baseline == 2:
            self.algorithm = "no_movement"
        elif self.args.baseline == 3:
            self.algorithm = 'max_variance'
            self.masks = self.env.game.world_view.action_masks
        elif self.args.baseline == 4:
            self.algorithm = 'min_variance'
            self.masks = self.env.game.world_view.action_masks
        elif self.args.baseline == 5:
            self.algorithm = 'highest_blur'
            self.masks = self.env.game.world_view.action_masks
        self.path_name = '{}_{}_env_{}_n{}_br{}_se{}_bs{}_ss{}_sl{}_pnm{}_os{}_r{}_rc{}_a{}'.format(
            self.args.type_of_MARL, self.algorithm, self.env_name, number, self.blur_radius,
            self.args.scale,self.args.blur_sigma,self.args.speed_scale,self.args.speed_layer,
            self.args.penalize_no_movement,self.args.outside_radar_value,
            self.args.radars, self.args.relative_change, self.args.agents)
        self.writer = SummaryWriter(log_dir='runs/DQN/{}'.format(self.path_name))
        if args.load_model:
            if os.path.isfile('models/DQN/net_{}'.format(self.path_name)):
                self.agent.net_load_state_dict(torch.load('models/DQN/net_{}'.format(self.path_name)))
                self.agent.target_net_load_state_dict(torch.load('models/DQN/target_net_{}'.format(self.path_name)))
        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
        if args.use_noisy:  # Noisy net
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps
    def run(self, ):
        self.evaluate_policy()
        if self.args.baseline ==0:
            while self.total_steps < self.args.max_train_steps:
                state = self.env.reset()[0]
                done = False
                episode_steps = 0
                while not done:
                    action = self.agent.choose_action(state, epsilon=self.epsilon)
                    action_ = action_unpack(action, self.args.action_dim) if (self.args.radars == 2) and (self.args.agents == 1) else action
                    next_state, reward, done, _,rewards, _ = self.env.step(action_)
                    episode_steps += 1
                    self.total_steps += 1

                    if not self.args.use_noisy:  # Decay epsilon
                        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                    if done and episode_steps != self.args.episode_limit:
                        terminal = True
                    else:
                        terminal = False
                    if self.use_sep_rewards:
                        self.replay_buffer.store_transition(state, np.array(action), rewards, next_state, terminal, done)  # Store the transition
                    else:
                        self.replay_buffer.store_transition(state, np.array(action), reward, next_state, terminal, done)  # Store the transition
                    state = next_state

                    if self.replay_buffer.cs() >= self.args.batch_size:
                        self.agent.learn(self.replay_buffer, self.total_steps)

                    if self.total_steps % self.args.evaluate_freq == 0:
                        self.evaluate_policy()
                    if (self.total_steps/10) % self.args.evaluate_freq == 0:
                        #self.save_models()
                        None
        else:
            for x in range(int(floor(self.args.max_train_steps/self.args.evaluate_freq))):
                self.total_steps = x*self.args.evaluate_freq
                self.evaluate_policy()
        #self.save_models()
        # self.save_rewards()

    def save_rewards(self):
        if isinstance(self.evaluate_rewards, torch.Tensor):
            er = self.evaluate_rewards.cpu()
        else:
            er = self.evaluate_rewards
        np.save('data_train/DQN/{}_{}_env_{}_number_{}_seed_{}_blur_radius_{}.npy'.format(self.args.type_of_MARL, self.algorithm, self.env_name, self.number, self.seed, self.blur_radius), np.array(er))

    def save_models(self):
        torch.save(self.agent.net_state_dict(),'models/DQN/net_{}'.format(self.path_name))
        torch.save(self.agent.target_net_state_dict(), 'models/DQN/target_net_{}'.format(self.path_name))
    def evaluate_policy(self, ):
        evaluate_reward = 0
        unpenalized_evaluate_reward = 0
        radar_stats = stats()
        if self.args.baseline >= 1:
            if args.radars > 1:
                action_ = list(map(lambda x: randint(0,self.args.action_dim-1), range(args.radars)))
            else:
                action_ = [randint(0,self.args.action_dim-1)]
        else:
            self.agent.net_eval()
        for _ in range(self.args.evaluate_times):
            state = self.env_evaluate.reset()[0]
            done = False
            episode_reward = 0
            episode_unpenalized_reward = 0
            actions = []
            # images = []
            # transform = T.ToPILImage()
            while not done:
                if self.args.baseline == 0:
                    action = self.agent.choose_action(state, epsilon=self.epsilon)
                    action_ = action_unpack(action, self.args.action_dim) if (self.args.radars == 2) and (
                                self.args.agents == 1) else action
                else:
                    action_, action = baselines_next_step(self, action_,state)
                next_state, reward, done, _, rewards,unpenalized_reward = self.env_evaluate.step(action_)
                # if len(images)<20:
                #     images.append(transform(torch.stack([state.squeeze(0)] * 3, dim=0)))
                # if len(images)==20:
                #     images[0].save("./images/xx.gif", save_all=True, append_images=images,loop=0)
                episode_reward += reward
                episode_unpenalized_reward += unpenalized_reward
                state = next_state
                actions.append(action_repack([radar.given_dir for radar in self.env_evaluate.game.radars],self.env.action_size))
                print(actions)
            radar_stats.add_stats(self.env_evaluate.info_analysis(),actions)
            evaluate_reward += episode_reward
            unpenalized_evaluate_reward += episode_unpenalized_reward
        self.agent.net_train()
        analysis = radar_stats.stats_analysis()
        evaluate_reward /= self.args.evaluate_times
        unpenalized_evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward, self.epsilon))
        self.writer.add_scalar('step_rewards', evaluate_reward, global_step=self.total_steps)
        self.writer.add_scalar('step_time_to_first_view', analysis['avg_time_til_first_view'], global_step=self.total_steps)
        self.writer.add_scalar('target_view_rate_to_velocity_corr', analysis['views_vel_corr'], global_step=self.total_steps)
        self.writer.add_scalar('world_view_avg_loss', analysis['avg_world_loss'], global_step=self.total_steps)
        self.writer.add_scalar('percent_targets_seen', analysis['percent_targets_seen'], global_step=self.total_steps)
        self.writer.add_scalar('actions_taken', analysis['unique_actions'], global_step=self.total_steps)
        self.writer.add_scalar('target_view_rate_to_velocity_slope', analysis['veiws_vel_slope'], global_step=self.total_steps)
        self.writer.add_scalar('target_view_rate_to_doppler_velocity_corr', analysis['views_doppler_corr'], global_step=self.total_steps)
        self.writer.add_scalar('average_view_rate', analysis['average_view_rate'], global_step=self.total_steps)
        self.writer.add_scalar('unpenalized_step_reward', unpenalized_evaluate_reward, global_step=self.total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--max_train_steps", type=int, default=int(1.5e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
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
    parser.add_argument("--blur_radius", type=int, default=1, help="br: size of the radius of the gaussian filter applied to previous views")
    parser.add_argument("--scale", type=int, default=23, help="s: factor by which the space is scaled down")
    parser.add_argument("--blur_sigma", type=float, default=0.5, help="bs: guassian blur sigma")
    parser.add_argument("--common_destination", type=list, default=[-200,-200], help="cd: a common location for targets come from and go to")
    parser.add_argument("--cdl", type=float, default=0.0, help="common destination likelihood : how many targets go to location")
    parser.add_argument("--gpu_number", type=int, default=0, help="gpu used")
    parser.add_argument("--speed_layer", type=int, default=0, help="sl: if speed is included in state space")
    parser.add_argument("--speed_scale", type=int, default =1, help="ss: how much the reward is scaled for seeing moving objects compared to not moving object")
    parser.add_argument("--env_name", type=str, default ='radar_sim', help="how much the reward is scaled for seeing moving objects compared to not moving object")
    parser.add_argument("--agents", type=int, default =1, help="a: how much the reward is scaled for seeing moving objects compared to not moving object")
    parser.add_argument("--radars", type=int, default =2, help="r: how much the reward is scaled for seeing moving objects compared to not moving object")
    parser.add_argument("--relative_change", type=int, default =0, help="rc: if 0 then an action is a direction to look in, if 1 then an action is a change in direction to look in since the last action")
    parser.add_argument("--penalize_no_movement", type=int, default =1, help="pnm: if no change in action is taken, and the reward is 0, this action is  penalized with a reward of -1")
    parser.add_argument("--type_of_MARL", type=str, default="single_agent", help="type of shared info in the MARL system")
    parser.add_argument("--baseline", type=int, default=0, help="type of shared info in the MARL system, if its 1 then its move in a circle, if its 2 then no movement, 3 is max variance, 4 is min variance")
    parser.add_argument("--outside_radar_value", type=float, default=0.9, help="value outside of radar observation area")
    args = parser.parse_args()

    env_index = 1
    set_gpu_name("cuda:" + str(args.gpu_number))
    runner = Runner(args=args, env_name=args.env_name, number=1, seed=0)
    runner.run()
