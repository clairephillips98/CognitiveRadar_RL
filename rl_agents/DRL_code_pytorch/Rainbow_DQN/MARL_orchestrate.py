from rl_agents.DRL_code_pytorch.Rainbow_DQN.rainbow_dqn import DQN
from rl_agents.DRL_code_pytorch.Rainbow_DQN.replay_buffer import *


class MARL_Double_Agent(DQN):

    def __init__(self, args):
        self.args = args
        self.agents = {0: DQN(args), 1: DQN(args)}
        if self.args.type_of_MARL in ['some_shared_info', 'some_shared_info_shared_reward', 'shared_targets_only']:
            self.diff_states = True
        else:
            self.diff_states = False
        if ['some_shared_info', 'shared_targets_only']:
            self.diff_rewards = True
        else:
            self.diff_rewards = False
        self.replay_buffers = {}

    def choose_action(self, state, epsilon):
        if self.diff_states:
            return list(map(lambda x: self.agents[x].choose_action(state[x], epsilon), range(self.args.agents)))
        else:
            return list(map(lambda x: self.agents[x].choose_action(state, epsilon), range(self.args.agents)))
        return action

    def net_eval(self):
        map(lambda x: self.agents[x].net.eval(), range(self.args.agents))

    def net_train(self):
        map(lambda x: self.agents[x].net.train(), range(self.args.agents))

    def learn(self, replay_buffer, total_steps):
        map(lambda x: self.agents[x].learn(replay_buffer[x], total_steps, x), range(self.args.agents))

    def net_load_state_dict(self, path):
        map(lambda x: self.agents[x].net.load_state_dict(path), range(self.args.agents))

    def target_net_load_state_dict(self, path):
        map(lambda x: self.agents[x].target_net.load_state_dict(path), range(self.args.agents))

    def net_state_dict(self):

        return self.agents[0].net.state_dict(), self.agents[1].net.state_dict()

    def target_net_state_dict(self):
        return self.agents[0].target_net.state_dict(), self.agents[1].target_net.state_dict()


class MARL_Double_RB:
    def __init__(self, args):
        self.args = args
        if args.use_per and args.use_n_steps:
            self.replay_buffer = {0: N_Steps_Prioritized_ReplayBuffer(args), 1: N_Steps_Prioritized_ReplayBuffer(args)}
        elif args.use_per:
            self.replay_buffer = {0: Prioritized_ReplayBuffer(args), 1: Prioritized_ReplayBuffer(args)}
        elif args.use_n_steps:
            self.replay_buffer = {0: N_Steps_ReplayBuffer(args), 1: N_Steps_ReplayBuffer(args)}
        else:
            self.replay_buffer = {0: ReplayBuffer(args), 1: ReplayBuffer(args)}

        if self.args.type_of_MARL in ['some_shared_info', 'some_shared_info_shared_reward', 'shared_targets_only']:
            self.diff_states = True
        else:
            self.diff_states = False
        if self.args.type_of_MARL in ['some_shared_info', 'shared_targets_only']:
            self.diff_rewards = True
        else:
            self.diff_rewards = False
    def store_transition(self, state, action, reward, next_state, terminal, done):
        if (self.diff_states == True) & (self.diff_rewards == True):
            map(lambda x: self.replay_buffer[x].store_transition(state[x], action, reward[x], next_state[x],
                                                                 terminal, done), range(self.args.agents))
        elif (self.diff_states == False) & (self.diff_rewards == True):

            map(lambda x: self.replay_buffer[x].store_transition(state, action, reward[x], next_state, terminal,
                                                                 done), range(self.args.agents))
        elif (self.diff_states == True) & (self.diff_rewards == False):
            map(lambda x: self.replay_buffer[x].store_transition(state[x], action, reward, next_state[x], terminal,
                                                                 done), range(self.args.agents))
        else:
            map(lambda x: self.replay_buffer[x].store_transition(state, action, reward, next_state, terminal,

                                                                 done), range(self.args.agents))

    def cs(self):
        return self.replay_buffer[0].current_size
