import copy
import numpy as np
import gym
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

gym.logger.set_level(40)

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
        self._max_step_count = config['max_step_count']
        self._true_reward_weights = config['true_reward_weights']
        self._step_count = 0
        
        action_space = tuple([(spaces.Box(low=0, high=5, shape=(self.n_vars,))) for i in range(self.n_agents)])
        self.action_space = spaces.Tuple(action_space)        
        
        observation_space = [(spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_vars,))) for i in range(self.n_agents)]
        observation_space = tuple(observation_space)
        self.observation_space = spaces.Tuple(observation_space)

        self.agents_var = None
        self._agent_dones = None
        self._total_episode_reward = None
        self.steps_beyond_done = None
        self.last_true_reward = np.zeros(self.n_agents)
        
        self.agent_ids = [] #['agent_0', 'agent_1', ...]
        self.agent_idx = {} #{'agent_0': 0, ...}
        for i in range(self.n_agents):
            self.agent_ids.append('agent_' + str(i))
            self.agent_idx['agent_' + str(i)] = i

    def _correct_obs_order(self, agent_i, obs):
        current_agent_obs = obs.pop(agent_i)
        obs.insert(0, current_agent_obs)
        return obs

    def get_agent_obs(self, agent_var):
        """
        Returns a dictionary with all agents observations
        """
        _obs = {}
        for agent_i in range(self.n_agents):
            # add state
            _agent_i_obs = copy.copy(agent_var)

            _agent_i_obs = self._correct_obs_order(agent_i, _agent_i_obs)

            _obs[self.agent_ids[agent_i]] = _agent_i_obs

        return _obs
    
    def reset(self):
        """
        Resets the environment and returns observations
        """
        self.agent_var = [([0] * self.n_vars) for i in range(self.n_agents)]
        self._step_count = 0
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self._agent_dones = [False for _ in range(self.n_agents)]
        
        return self.get_agent_obs(self.agent_var)

    def _correct_action_order(self, agent_i, action):
        current_agent_act = action.pop(0)
        action.insert(self.agent_idx[agent_i], current_agent_act)
        return action

    def __update_agent_action(self, agent_i, action):
        """
        Apply agent_i's action
        """
        action = list(action)
        self._correct_action_order(agent_i, action)
        action = (action - np.mean(action))
        # Make the actions have a bigger affect on the other agent than itself.
        scale = np.array([2] * self.n_agents)
        scale[self.agent_idx[agent_i]] = 1
        action = np.multiply(action, scale[:, np.newaxis])
        self.agent_var = (np.array(self.agent_var) + action).tolist()

    def get_true_rewards(self, pre_agent_var):
        """
        Returns list of rewards according to the true reward function
        """
        diff = (np.array(self.agent_var) - np.array(pre_agent_var))
        rewards = np.dot(diff, self._true_reward_weights)
        return rewards
    
    def get_rewards(self, pre_agent_var):
        """
        Returns dictionary of rewards according to each agents reward weights
        Currently just a dot product of reward weights and state.
        TODO: Change from dot product to neural network
        """
        rewards = []
        reward_dict = {}
        for i in range(self.n_agents):
            cur_obs = np.asarray(self.get_agent_obs(self.agent_var)[self.agent_ids[i]]).flatten()
            old_obs = np.asarray(self.get_agent_obs(pre_agent_var)[self.agent_ids[i]]).flatten()
            r = np.dot(cur_obs, self.reward_weights[i]) - np.dot(old_obs, self.reward_weights[i])
            rewards.append(r)
            reward_dict[self.agent_ids[i]] = r
        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]
        return reward_dict
    
    def step(self, action_dict):
        assert len(action_dict) == self.n_agents

        self._step_count += 1

        pre_agent_var = self.agent_var

        for agent_i, action in action_dict.items():
            self.__update_agent_action(agent_i, action)

        rewards = self.get_rewards(pre_agent_var)
        self.last_true_reward = self.get_true_rewards(pre_agent_var)
        
        done = {
            "__all__": self._step_count >= self._max_step_count,
        }
        return self.get_agent_obs(self.agent_var), rewards, done, {}