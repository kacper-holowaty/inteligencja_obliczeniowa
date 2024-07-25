# PROBLEM PLECAKOWY

import pygad
import numpy as np
import time

items = [
    {"name": "zegar", "value": 100, "weight": 7},
    {"name": "obraz-pejzaż", "value": 300, "weight": 7},
    {"name": "obraz-portret", "value": 200, "weight": 6},
    {"name": "radio", "value": 40, "weight": 2},
    {"name": "laptop", "value": 500, "weight": 5},
    {"name": "lampka nocna", "value": 70, "weight": 6},
    {"name": "srebrne sztućce", "value": 100, "weight": 1},
    {"name": "porcelana", "value": 250, "weight": 3},
    {"name": "figura z brązu", "value": 300, "weight": 10},
    {"name": "skórzana torebka", "value": 280, "weight": 3},
    {"name": "odkurzacz", "value": 300, "weight": 15}
]

gene_space = [0, 1]
max_weight = 25

def fitness_func(model, solution, solution_idx):
    total_weight = np.sum(solution * [item["weight"] for item in items])
    total_value = np.sum(solution * [item["value"] for item in items])
    if total_weight > max_weight:
        return 0
    else:
        return total_value
    
fitness_function = fitness_func
sol_per_pop = 50
num_genes = len(items)
num_generations = 100
num_parents_mating = 20
keep_parents = 5
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

# ga_instance = pygad.GA(gene_space=gene_space,
#                        num_generations=num_generations,
#                        num_parents_mating=num_parents_mating,
#                        fitness_func=fitness_function,
#                        sol_per_pop=sol_per_pop,
#                        num_genes=num_genes,
#                        parent_selection_type=parent_selection_type,
#                        keep_parents=keep_parents,
#                        crossover_type=crossover_type,
#                        mutation_type=mutation_type,
#                        mutation_percent_genes=mutation_percent_genes)

# ga_instance.run()

# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# selected_items = [items[i]["name"] for i in range(len(solution)) if solution[i] == 1]
# print("Selected items: ", selected_items)
# print("Total value: ", solution_fitness)
# print("Total weight: ", np.sum(solution * [item["weight"] for item in items]))

# ga_instance.plot_fitness()

best_solution_value = 1630
successful_runs = 0
total_time = 0
num_runs = 10

for _ in range(num_runs):
    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
    )
    
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if solution_fitness == best_solution_value:
        successful_runs += 1
        total_time += (end_time - start_time)

print("Successful runs: ", successful_runs)
print("Success rate: ", (successful_runs / num_runs) * 100, "%")
if successful_runs > 0:
    print("Average time for successful runs: ", total_time / successful_runs, " seconds")


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

selected_items = [items[i]["name"] for i in range(len(solution)) if solution[i] == 1]
print("Selected items: ", selected_items)
print("Total value: ", solution_fitness)
print("Total weight: ", np.sum(solution * [item["weight"] for item in items]))

ga_instance.plot_fitness()