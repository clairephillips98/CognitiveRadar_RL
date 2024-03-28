from rl_agents.DRL_code_pytorch.Rainbow_DQN.rainbow_dqn import DQN


class MARL_Double_Agent(DQN):

    def __init__(self, args):
        self.agent_1 = DQN(args)
        self.agent_2 = DQN(args)

    def choose_action(self, state, epsilon):
        action_1 = self.agent_1.choose_action(state, epsilon)
        action_2 = self.agent_2.choose_action(state, epsilon)
        action = [action_1, action_2]
        return action

    def net_eval(self):
        self.agent_1.net.eval()
        self.agent_2.net.eval()

    def net_train(self):
        print('here')
        self.agent_1.net.train()
        self.agent_2.net.train()

    def learn(self, replay_buffer, total_steps):
        self.agent_1.learn(replay_buffer, total_steps, 0)
        self.agent_2.learn(replay_buffer, total_steps, 1)

    def net_load_state_dict(self, path):
        self.agent_1.net.load_state_dict(path)
        self.agent_2.net.load_state_dict(path)

    def target_net_load_state_dict(self, path):
        self.agent_1.target_net.load_state_dict(path)
        self.agent_2.target_net.load_state_dict(path)

    def net_state_dict(self):
        return self.agent_1.net.state_dict(), self.agent_2.net.state_dict()

    def target_net_state_dict(self):
        return self.agent_1.target_net.state_dict(), self.agent_2.target_net.state_dict()
