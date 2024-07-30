# PSO DLA STOPU METALI 

import numpy as np
import math
import pyswarms as ps 
from pyswarms.utils.functions import single_obj as fx 
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
 
def endurance(params):
    x, y, z, u, v, w = params
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)

def objective_function(X):
    n_particles = X.shape[0]
    j = [endurance(X[i]) for i in range(n_particles)]
    return -np.array(j)

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9} 
x_min = np.zeros(6)
x_max = np.ones(6)
my_bounds = (x_min, x_max) 

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds) 

best_cost, best_pos = optimizer.optimize(objective_function, iters=1000)

print(f"Best cost: {-best_cost}")
print(f"Best position: {best_pos}")

cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.savefig("zad_01.png")
plt.show()