import copy
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class SimpleEnv(MultiAgentEnv):
    """
    Simple test environment
    The agents have a state represented by 2 variables. The agents can move the
    value of any agent’s state variable (including their own) up or down by 1. 
    
    Action space 
    List of number of points to add to each variable in range [0, 5)
    [agent1.var1, agent1.var2, agent2.var1, agent2.var2]
    
    Observation space
    The value of every variable for every agent. The last value is the id of the agent.
    [agent1.var1, agent1.var2, agent2.var1, agent2.var2, id]
    
    Reward
    """

    def __init__(self, config):
        self.n_agents = config['n_agents']
        self.n_vars = config['n_vars']
        self.reward_weights = config['reward_weights']
        self._step_count = 0
        
        action_space = tuple([(spaces.Box(low=0, high=5, shape=(self.n_vars,))) for i in range(self.n_agents)])
        self.action_space = spaces.Tuple(action_space)        
        
        observation_space = [(spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_vars,))) for i in range(self.n_agents)]
        observation_space.append(spaces.Discrete(self.n_agents))
        observation_space = tuple(observation_space)
        self.observation_space = spaces.Tuple(observation_space)

        self.agents_var = None
        self._agent_dones = None
        self._total_episode_reward = None
        self.steps_beyond_done = None
        
        self.agent_ids = []
        self.agent_idx = {}
        for i in range(self.n_agents):
            self.agent_ids.append('agent_' + str(i))
            self.agent_idx['agent_' + str(i)] = i
        
    def get_agent_obs(self):
        _obs = {}
        for agent_i in range(self.n_agents):
            # add state
            _agent_i_obs = copy.copy(self.agent_var)

            #add agent id
            _agent_i_obs.append(agent_i)

            _obs[self.agent_ids[agent_i]] = _agent_i_obs

        return _obs
    
    def reset(self):
        self.agent_var = [([0] * self.n_vars) for i in range(self.n_agents)]
        self._step_count = 0
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self._agent_dones = [False for _ in range(self.n_agents)]
        
        return self.get_agent_obs()

    def __update_agent_action(self, agent_i, action):
        action = (action - np.mean(action))
        # Make the actions have a bigger affect on the other agent than itself.
        scale = [2] * self.n_agents
        scale[self.agent_idx[agent_i]] = 1
        action *= scale
        self.agent_var = (np.array(self.agent_var) + action).tolist()

    def get_first_values(self, agent_var):
        """Gives the first value for each agent."""
        return np.array(agent_var)[:,0]
        
    def get_rewards(self, pre_agent_var):
        """Rewards"""
        rewards = self.get_first_values(self.agent_var) - self.get_first_values(pre_agent_var)
        reward_dict = {}
        for i, r in enumerate(rewards):
            reward_dict[self.agent_ids[i]] = r
        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]
        return reward_dict
    
    def get_true_rewards(self, pre_agent_var):
        """Rewards list"""
        rewards = []
        for i in range(self.n_agents):
            r = np.dot(self.agent_var[i], self.reward_weights[i]) - np.dot(pre_agent_var[i], self.reward_weights[i])
            rewards.append(r)
        return rewards
    
    def step(self, action_dict):
        assert len(action_dict) == self.n_agents

        self._step_count += 1

        pre_agent_var = self.agent_var

        for agent_i, action in action_dict.items():
            self.__update_agent_action(agent_i, action)

        rewards = self.get_rewards(pre_agent_var)
        true_rewards = self.get_true_rewards(pre_agent_var)
        
        done = {
            "__all__": self._step_count >= 10,
        }

        return self.get_agent_obs(), rewards, done, {'true_rewards': true_rewards}