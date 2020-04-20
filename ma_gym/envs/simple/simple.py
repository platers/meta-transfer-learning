import copy
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text, draw_score_board

logger = logging.getLogger(__name__)


class Simple(gym.Env):
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

    def __init__(self, n_agents=2, n_vars=2, step_cost=0, full_observable=False, max_steps=100):
        self.n_agents = n_agents
        self.n_vars = n_vars
        self._max_steps = max_steps
        self._step_count = None
        self._step_cost = step_cost
        self.full_observable = full_observable

        self.action_space = MultiAgentObservationSpace([spaces.Box(low=0, high=5, shape=(self.n_agents, self.n_vars)) for _ in range(self.n_agents)])
        self.observation_space = MultiAgentObservationSpace([spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_agents * self.n_vars + 1,)) for _ in range(self.n_agents)])

        self.agents_var = None
        self._agent_dones = None
        self._total_episode_reward = None
        self.steps_beyond_done = None
        
    def __init_full_obs(self):
        self.agent_var = [([0] * self.n_vars) for i in range(self.n_agents)]

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            # add state
            _agent_i_obs = copy.copy(self.agent_var)

            #add agent id
            _agent_i_obs.append(agent_i)

            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def reset(self):
        self.__init_full_obs()
        self._step_count = 0
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self._agent_dones = [False for _ in range(self.n_agents)]

        return self.get_agent_obs()

    def __update_agent_action(self, agent_i, action):
        action = (action - np.mean(action)).astype(int)
        # Make the actions have a bigger affect on the other agent than itself.
        scale = [2] * self.n_agents
        scale[agent_i] = 1
        action *= scale
        self.agent_var = (np.array(self.agent_var) + action).tolist()
        #print(self.agent_var)

    def get_first_values(self, agent_var):
        """Gives the first value for each agent."""
        return np.array(agent_var)[:,0]

    def step(self, agents_action):
        assert len(agents_action) == self.n_agents

        self._step_count += 1

        pre_agent_var = self.agent_var

        for agent_i, action in enumerate(agents_action):
            self.__update_agent_action(agent_i, action)

        rewards = self.get_first_values(self.agent_var) - self.get_first_values(pre_agent_var)

        if self._step_count >= self._max_steps:
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        # Following snippet of code was refereed from:
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L124
        if all(self._agent_dones):
            self.steps_beyond_done = 0
        elif self.steps_beyond_done is not None:
            if self.steps_beyond_done == 0:
                logger.warning(
                    "You are calling 'step()' even though this environment has already returned all(dones) = True for "
                    "all agents. You should always call 'reset()' once you receive 'all(dones) = True' -- any further"
                    " steps are undefined behavior.")
            self.steps_beyond_done += 1
            rewards = [0 for _ in range(self.n_agents)]

        return self.get_agent_obs(), rewards, self._agent_dones, {}

    def render(self, mode='human'):
        pass

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]



