# A proof of concept showing using a genetic algorithm with our environment.
# It is similar to https://github.com/DEAP/deap/blob/a0b78956e28387785e3bb6e2b4b1f1b32c2b3883/examples/ga/onemax_short.py

import array
import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import gym
import ma_gym

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

max_action = 5

toolbox.register("attr", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def get_action(agent_i, reward_weights, obs):
  # This only works for this simple environment
  # In this enviroment, the reward weights and actions are the same
  action = np.sign(np.asarray(reward_weights) + 0.00001) * max_action #add small constant to avoid 0
  return action

def evalOneMax(individual):
  """Runs the environment. It always takes the same action determined by the individual's genes.
  It returns the total reward as the fitness."""
  env = gym.make('Simple-v0')
  done_n = [False for _ in range(env.n_agents)]
  ep_reward = 0

  alturism_amount = individual

  obs_n = env.reset()
  for _ in range(5):
    actions = [(get_action(i, alturism_amount, obs_n[i])) for i in range(env.n_agents)]
    obs_n, reward_n, done_n, info = env.step(actions)
    ep_reward += reward_n[0]
  env.close()

  return ep_reward,

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
  pop = toolbox.population(n=300)
  hof = tools.HallOfFame(1)
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("std", np.std)
  stats.register("min", np.min)
  stats.register("max", np.max)

  pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10,
                                 stats=stats, halloffame=hof, verbose=True)

  print("pop", pop)

  return pop, log, hof

if __name__ == "__main__":
  main()