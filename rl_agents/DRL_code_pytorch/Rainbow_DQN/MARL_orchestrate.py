from rl_agents.DRL_code_pytorch.Rainbow_DQN.rainbow_dqn import DQN


class MARL_Double_Agent(DQN):

    def __init__(self, args):
        self.agents = {0: DQN(args), 1: DQN(args)}
        if self.args.type_of_MARL in ['some_shared_info', 'some_shared_info_shared_reward', 'shared_targets_only']:
            self.diff_states = True
        else:
            self.diff_states=False
        if ['some_shared_info', 'shared_targets_only']:
            self.diff_rewards= True
        else:
            self.diff_rewards= False

    def choose_action(self, state, epsilon):
        if self.diff_states:
            return list(map(lambda x: self.agents[i].choose_action(state[i], epsilon),range(self.args.agents)))
        else:
            return list(map(lambda x: self.agents[i].choose_action(state, epsilon), range(self.args.agents)))
        return action

    def net_eval(self):
        map(lambda x: self.agents[x].net.eval(),range(self.args.agents))


    def net_train(self):
        map(lambda x: self.agents[x].net.train(),range(self.args.agents))

    def learn(self, replay_buffer, total_steps):
        map(lambda x: self.agents[x].learn(replay_buffer, total_steps, x),range(self.args.agents))

    def net_load_state_dict(self, path):
        map(lambda x: self.agents[x].net.load_state_dict(path),range(self.args.agents))

    def target_net_load_state_dict(self, path):
        map(lambda x: self.agents[x].target_net.load_state_dict(path),range(self.args.agents))

    def net_state_dict(self):

        return self.agents[0].net.state_dict(), self.agents[1].net.state_dict()

    def target_net_state_dict(self):
        return self.agents[0].target_net.state_dict(), self.agents[1].target_net.state_dict()
