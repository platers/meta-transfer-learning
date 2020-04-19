# A proof of concept showing using a genetic algorithm with our environment.
# It is similar to https://github.com/DEAP/deap/blob/a0b78956e28387785e3bb6e2b4b1f1b32c2b3883/examples/ga/onemax_short.py

import array
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import gym
import ma_gym

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

max_action = 5

toolbox.register("attr", random.randint, 0, max_action)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def get_action(agent_i, alturism_amount):
  # TODO: This only works for 2 agents.
  if agent_i == 0:
    return [max_action-alturism_amount, 0, alturism_amount, 0]
  else:
    return [alturism_amount, 0, max_action-alturism_amount, 0]

def evalOneMax(individual):
  """Runs the environment. It always takes the same action determined by the individual's genes.
  It returns the total reward as the fitness."""
  env = gym.make('Simple-v0')
  done_n = [False for _ in range(env.n_agents)]
  ep_reward = 0

  alturism_amount = individual[0]

  env.reset()
  for _ in range(3):
    obs_n, reward_n, done_n, info = env.step([get_action(0, alturism_amount), get_action(1, alturism_amount)])
    ep_reward += reward_n[0]
  env.close()

  # The environment always gives 0 reward for some reason.
  return ep_reward,

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
  pop = toolbox.population(n=300)
  hof = tools.HallOfFame(1)
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", numpy.mean)
  stats.register("std", numpy.std)
  stats.register("min", numpy.min)
  stats.register("max", numpy.max)

  pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                                 stats=stats, halloffame=hof, verbose=True)

  print("pop", pop)

  return pop, log, hof

if __name__ == "__main__":
  main()