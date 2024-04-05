import numpy as np
import matplotlib.pyplot as plt
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
def fitness_func(x):
    # sprawdź, czy każdy punkt w roju mieści się w zakresie [-5, 5]
    if np.any(np.abs(x) > 5):
        # zwróć duży koszt, jeśli któryś punkt przekracza zakres
        return np.inf
    return (-1)*(1 / (funea(x[:, 0], x[:, 1]) + 1e-8))

# Ustawienia algorytmu rojowego
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options)

# Uruchomienie algorytmu rojowego
cost, pos = optimizer.optimize(fitness_func, iters=100, verbose=True)

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

