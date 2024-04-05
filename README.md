# Swarm vs. Genetic Algorithm Comparison

This project aims to compare the performance of two optimization algorithms: Particle Swarm Optimization (PSO) and Genetic Algorithm (GA). The algorithms are evaluated based on their ability to optimize a given objective function.

## Objective Function

The objective function used in this project is a multi-dimensional function that represents a complex optimization problem. It is defined as:

\[ f(x_1, x_2) = x_1^2 + 5x_2^2 + 5z(sin(6 arctan(x_2/x_1) + 5z))^3 - 100g(x_1+3, x_2+3) - 125g(x_1-2, x_2-2) - e^{(0.005x_1)} \]

where \( z = \sqrt{x_1^2 + x_2^2} \) and \( g(x_1, x_2) = max(x_1, x_2) \).

## Files

- **pso.py**: Implementation of Particle Swarm Optimization algorithm.
- **main.py**: Implementation of Genetic Algorithm.
- **compare.py**: Script to compare the performance of PSO and GA.

## How to Run

To run the comparison between PSO and GA, execute the `compare.py` script. Ensure you have the necessary dependencies installed.

```bash
python compare.py
```

## Results

The results of the comparison include the best solution found by each algorithm, their corresponding fitness values, and execution times.

## Dependencies

- numpy
- matplotlib
- pyswarms
- pygad
