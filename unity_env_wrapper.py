class EnvSingleWrapper:
    def __init__(self, env, train_mode=False):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.train_mode = train_mode
        brain = env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        state = self.reset()
        self.state_size = state.shape[1]
        self.num_agents = 1

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        states = env_info.vector_observations
        return states

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        return next_states, rewards, dones


class EnvMultipleWrapper:
    def __init__(self, env, train_mode=False):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.train_mode = train_mode
        brain = env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size

        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        states = env_info.vector_observations
        self.state_size = states.shape[1]
        self.num_agents = len(env_info.agents)

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        states = env_info.vector_observations
        return states

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        return next_states, rewards, dones
