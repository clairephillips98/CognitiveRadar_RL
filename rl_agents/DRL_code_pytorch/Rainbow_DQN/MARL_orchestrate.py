from rl_agents.DRL_code_pytorch.Rainbow_DQN.rainbow_dqn import DQN
from rl_agents.DRL_code_pytorch.Rainbow_DQN.replay_buffer import *
import threading

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)

def thread_result(fuct_lst, args_list):
    threads = {}
    for i, x in enumerate(fuct_lst):
        threads[i] = ThreadWithResult(target=x, args=args_list[i])
    for i in range(len(fuct_lst)):
        threads[i].start()
    for i in range(len(fuct_lst)):
        threads[i].join()
    items = []
    for i in range(len(fuct_lst)):
        items.append(threads[i].result)
    return items

def thread(fuct_lst, args_list):
    threads = {}
    for i, x in enumerate(fuct_lst):
        threads[i] = threading.Thread(target=x, args=args_list[i])
    for i in range(len(fuct_lst)):
        threads[i].start()
    for i in range(len(fuct_lst)):
        threads[i].join()

class MARL_Double_Agent(DQN):

    def __init__(self, args):
        self.args = args
        self.agents = {0: DQN(args), 1: DQN(args)}
        if self.args.type_of_MARL in ['some_shared_info', 'some_shared_info_shared_reward', 'shared_targets_only']:
            self.diff_states = True
        else:
            self.diff_states = False
        if self.args.type_of_MARL in ['some_shared_info', 'shared_targets_only']:
            self.diff_rewards = True
        else:
            self.diff_rewards = False
        self.replay_buffers = {}

    def choose_action(self, state, epsilon):
        if self.diff_states:
            return thread_result(
                list(map(lambda x: self.agents[x].choose_action, range(self.args.agents))),
                list(map(lambda x:(state[x], epsilon), range(self.args.agents))))
        else:
            return thread_result(list(map(lambda x: self.agents[x].choose_action, range(self.args.agents))),
                list(map(lambda x:(state, epsilon), range(self.args.agents))))

    def net_eval(self):
        list(map(lambda x: self.agents[x].net.eval(), range(self.args.agents))),


    def net_train(self):
        list(map(lambda x: self.agents[x].net.train(), range(self.args.agents)))

    def learn(self, replay_buffer, total_steps):
        if self.args.type_of_MARL in ['some_shared_info', 'some_shared_info_shared_reward', 'shared_targets_only']:
            list(map(lambda x: self.agents[x].learn(replay_buffer.replay_buffer[x], total_steps, x), range(self.args.agents)))
        else:
            list(map(lambda x: self.agents[x].learn(replay_buffer, total_steps, x), range(self.args.agents)))

    def net_load_state_dict(self, path):
        list(map(lambda x: self.agents[x].net.load_state_dict(path), range(self.args.agents)))

    def target_net_load_state_dict(self, path):
        list(map(lambda x: self.agents[x].target_net.load_state_dict(path), range(self.args.agents)))

    def net_state_dict(self):

        return self.agents[0].net.state_dict(), self.agents[1].net.state_dict()

    def target_net_state_dict(self):
        return self.agents[0].target_net.state_dict(), self.agents[1].target_net.state_dict()

import copy
class MARL_Double_RB:
    def __init__(self, args):
        self.args = args
        args_0 = copy.copy(args)
        args_1 = copy.copy(args)
        if args.type_of_MARL == "shared_targets_only":
            args_1.state_dim = [args_1.state_dim[1], args_1.state_dim[0]]
        if args.use_per and args.use_n_steps:
            self.replay_buffer = {0: N_Steps_Prioritized_ReplayBuffer(args_0), 1: N_Steps_Prioritized_ReplayBuffer(args_1)}
        elif args.use_per:
            self.replay_buffer = {0: Prioritized_ReplayBuffer(args_0), 1: Prioritized_ReplayBuffer(args_1)}
        elif args.use_n_steps:
            self.replay_buffer = {0: N_Steps_ReplayBuffer(args_0), 1: N_Steps_ReplayBuffer(args_1)}
        else:
            self.replay_buffer = {0: ReplayBuffer(args_0), 1: ReplayBuffer(args_1)}

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
            thread(
                list(map(lambda x: self.replay_buffer[x].store_transition, range(self.args.agents))),
                list(map(lambda x: (state[x], action, reward[x], next_state[x], terminal, done), range(self.args.agents)))
            )
        elif (self.diff_states == False) & (self.diff_rewards == True):
            thread(
                list(map(lambda x: self.replay_buffer[x].store_transition, range(self.args.agents))),
                list(map(lambda x: (state, action, reward[x], next_state, terminal, done), range(self.args.agents)))
            )
        elif (self.diff_states == True) & (self.diff_rewards == False):
            thread(
                list(map(lambda x: self.replay_buffer[x].store_transition, range(self.args.agents))),
                list(map(lambda x: (state[x], action, reward, next_state[x], terminal,
                                                                 done), range(self.args.agents)))
            )
        else:
            thread(
                list(map(lambda x: self.replay_buffer[x].store_transition, range(self.args.agents))),
                list(map(lambda x: (state, action, reward, next_state, terminal,done), range(self.args.agents)))
            )
    def cs(self):
        return self.replay_buffer[0].current_size
