import matplotlib.pyplot as plt
import numpy as np
import pygad
import pyswarms as ps

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
for generation in range(num_generations):
    ga_instance.run()
    best_solution = ga_instance.best_solution()
    best_solution_progress.append(best_solution)
    fitness_progress.append(best_solution[1])
    best_solutions_3D.append((best_solution[0][0], best_solution[0][1], funea(*best_solution[0])))
    best_solutions_2D.append((best_solution[0][0], best_solution[0][1]))

    print(f"Generation {generation+1}: Best solution {best_solution[0]} with fitness {best_solution[1]}")

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


# Wykres funkcji celu i punktów najlepszych osobników
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Wykres funkcji celu
x1_vals = np.linspace(-5, 5, 100)
x2_vals = np.linspace(-5, 5, 100)
x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)
z = funea(x1_mesh, x2_mesh)
ax.plot_surface(x1_mesh, x2_mesh, z, cmap='viridis', alpha=0.8)

'''
# Punkty najlepszych osobników
x1_best = [sol[0][0] for sol in best_solution_progress]
x2_best = [sol[0][1] for sol in best_solution_progress]
z_best = [funea(sol[0][0], sol[0][1]) for sol in best_solution_progress]
ax.scatter(x1_best, x2_best, z_best, c='r', marker='o')

# Wykres 3D dla najlepszego osobnika
best_solution = ga_instance.best_solution()
ax.scatter(best_solution[0][0], best_solution[0][1], funea(*best_solution[0]), c='g', marker='*')
'''

# Ustawienie etykiet osi
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')

plt.show()


# Wykres z najlepszym osobnikiem każdej populacji
plt.plot(fitness_progress)
plt.title("Best solution per generation")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()


# Wypisanie wyniku i liczby iteracji
print(f"\nBest solution found:\n{best_solution[0]}\nFitness: {best_solution[1]}")
print(f"Number of iterations: {len(fitness_progress)}")


