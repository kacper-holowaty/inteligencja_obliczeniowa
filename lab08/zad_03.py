# LABIRYNT

import pygad
import numpy as np
import time

# 1 -> ściana, 0 -> ścieżka 
labirynt = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

start = (1, 1)
exit = (10, 10)

def fitness_func(model, solution, solution_idx):
    x, y = start
    kara = 0
    nagroda = 0

    for i, move in enumerate(solution):
        if x == exit[0] and y == exit[1]:
            nagroda = 5/(i + 1)  # dojście do końca labiryntu
            break
        elif move == 0 and labirynt[x, y-1] != 1:  # w lewo
            y -= 1
        elif move == 1 and labirynt[x-1, y] != 1:  # w górę
            x -= 1
        elif move == 2 and labirynt[x, y+1] != 1:  # w prawo
            y += 1
        elif move == 3 and labirynt[x+1, y] != 1:  # w dół
            x += 1
        else:
            kara = (abs(x-exit[0]) + abs(y-exit[1])) * 0.2 # kara za uderzenie w ścianę
            break

    distance = abs(x-exit[0]) + abs(y-exit[1])
    
    if x < 0 or x >= labirynt.shape[0] or y < 0 or y >= labirynt.shape[1]:
        return 0
    
    fitness = 1 / (distance + 1) - kara + nagroda
    return fitness


fitness_function = fitness_func

gene_space = [0, 1, 2, 3]
num_generations = 4000
num_parents_mating = 20
sol_per_pop = 50
num_genes = 30
keep_parents = 5
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 8


results = []
for i in range(10):
    ga_instance = pygad.GA(gene_space=gene_space,
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
                       stop_criteria="reach_1")
    start_timer = time.time()
    ga_instance.run()
    end_timer = time.time()
    results.append(end_timer - start_timer)

print("Wszystkie rezultaty:")
for i in results:
    print(f"{i:.2f} s")
print(f"średni czas: {sum(results)/10:.2f} s")


ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)


ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Najlepsze rozwiązanie: ", solution)
print("Fitness najlepszego rozwiązania: ", solution_fitness)

ga_instance.plot_fitness()