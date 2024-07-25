#  STOP METALI

import pygad
import numpy as np
import math

def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)

def fitness_func(model, solution, solution_idx):
    return endurance(solution[0], solution[1], solution[2], solution[3], solution[4], solution[5])

gene_space = {'low': 0.0, 'high': 1.0}

sol_per_pop = 50       # Liczba rozwiązań w populacji
num_genes = 6          # Liczba genów (x, y, z, u, v, w)
num_generations = 100  # Liczba pokoleń
num_parents_mating = 20  # Liczba rodziców do krzyżowania
keep_parents = 5       # Liczba rodziców zachowywanych dla następnej generacji
parent_selection_type = "sss"  # Typ selekcji rodziców (steady-state selection)
crossover_type = "single_point"  # Typ krzyżowania
mutation_type = "random"  # Typ mutacji
mutation_percent_genes = 20  # Procent genów poddawanych mutacji

ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
normal_solution = np.exp(solution)
print("Parameters of the best solution : {normal_solution}".format(normal_solution=normal_solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

ga_instance.plot_fitness()