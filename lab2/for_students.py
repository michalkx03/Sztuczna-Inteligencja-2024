from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 190
n_elite = 10

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
mutation_rate = 1
population = initial_population(len(items), population_size) #step 1
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    # Roulette Wheel Selection
    total_fitness = sum(fitness(items, knapsack_max_capacity, ind) for ind in population)
    probabilities = [fitness(items, knapsack_max_capacity, ind) / total_fitness for ind in population]
    selected_parents = random.choices(population, weights=probabilities, k=n_selection)

    # One-Point Crossover
    new_population= []
    for i in range(0, len(selected_parents), 2):
        parent1, parent2 = selected_parents[i], selected_parents[i + 1]
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        new_population.append(child1)
        new_population.append(child2)

    for child in new_population:
        if(random.random()<mutation_rate):
            index = random.randint(0,len(child)-1)
            child[index]=not child[index]

    # Elitism
    elite = sorted(population, key=lambda ind: fitness(items, knapsack_max_capacity, ind), reverse=True)[:n_elite]

    new_population.extend(elite)
    population = new_population
    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
