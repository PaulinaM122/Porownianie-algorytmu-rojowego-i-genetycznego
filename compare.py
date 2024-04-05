import matplotlib.pyplot as plt
import numpy as np
import pygad
import pyswarms as ps
import time

# Definicja funkcji celu
def funea(x1, x2):
    z = np.sqrt(x1**2 + x2**2)
    y = x1**2 + 5*x2**2 + 5*z*(np.sin(6*np.arctan2(x2, x1) + 5*z))**3 \
        - 100*g(x1+3, x2+3) - 125*g(x1-2, x2-2) - np.exp(x1*.005)
    return y


# Funkcje ograniczeń
def g(x1, x2):
    return np.max([x1, x2])


# Definicja funkcji oceny
def fitness_func(instance, solution, solution_idx):
    x1, x2 = solution
    fitness = 1 / (funea(x1, x2) + 1e-8)
    return fitness


# Tworzenie instancji problemu optymalizacyjnego
num_generations = 100
num_parents_mating = 10
sol_per_pop = 20
num_genes = 2
mutation_num_genes = 1
parent_selection_type = "rank"
crossover_type = "single_point"
mutation_type = "random"
keep_parents = 1
last_fitness = 0

# Inicjalizacja populacji
population_size = (sol_per_pop, num_genes)
initial_pop = np.random.uniform(low=-5.0, high=5.0, size=population_size)

ga_instance = pygad.GA(initial_population=initial_pop,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_num_genes=mutation_num_genes,
                       keep_parents=keep_parents)

# Przebieg algorytmu genetycznego
fitness_progress = []
best_solution_progress = []
best_solutions_3D = []
best_solutions_2D = []

start_time = time.time()
for generation in range(num_generations):
    ga_instance.run()
    best_solution = ga_instance.best_solution()
    best_solution_progress.append(best_solution)
    fitness_progress.append(best_solution[1])
    best_solutions_3D.append((best_solution[0][0], best_solution[0][1], funea(*best_solution[0])))
    best_solutions_2D.append((best_solution[0][0], best_solution[0][1]))

    #print(f"Generation {generation+1}: Best solution {best_solution[0]} with fitness {best_solution[1]}")
end_time = time.time()
ga_time = end_time - start_time
print("GA execution time: ", ga_time)

# Wykres funkcji celu i punktów najlepszych osobników
fig, ax = plt.subplots()
X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = funea(X, Y)
ax.contour(X, Y, Z, levels=50, zorder=5)
for solution in best_solutions_2D:
    ax.scatter(solution[0], solution[1], c='red', zorder=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2D plot of objective function and best solutions')
plt.show()

# Wypisanie wyniku i liczby iteracji
print(f"\nBest solution found:\n{best_solution[0]}\nFitness: {best_solution[1]}")
print(f"Number of iterations: {len(fitness_progress)}")

#---------optymalizacja rojowa----------

# Definicja funkcji oceny
def fitness_func2(x):
    return (-1)*(1 / (funea(x[:, 0], x[:, 1]) + 1e-8))

# Ustawienia algorytmu rojowego
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options)

# Uruchomienie algorytmu rojowego
start_time = time.time()
cost, pos = optimizer.optimize(fitness_func2, iters=100, verbose=True)
end_time = time.time()
pso_time = end_time - start_time
print("PSO execution time: ", pso_time)

# Wyświetlenie wyniku
print("Best solution found:")
print(f"    x1 = {pos[0]:.6f}")
print(f"    x2 = {pos[1]:.6f}")
print(f"Fitness = {1/cost:.6f}")

# Stworzenie tablicy wielowymiarowej z historią pozycji roju
history_pos = np.array(optimizer.pos_history)

# Wykres funkcji celu i ruchu roju
fig, ax = plt.subplots()
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = funea(X, Y)
ax.contour(X, Y, Z, levels=50, cmap='cool', zorder=5)
ax.scatter(history_pos[:, 0], history_pos[:, 1], color='black', s=10, alpha=0.5, zorder=10)


plt.show()