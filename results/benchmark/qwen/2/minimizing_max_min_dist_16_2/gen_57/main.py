# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
from sklearn.metrics.pairwise import euclidean_distances
from typing import Tuple, Optional, List
import warnings
from deap import base, creator, algorithms, tools
import random


class PointDispersionOptimizer:
    """Optimizes point placement to maximize min/max distance ratio."""

    def __init__(self, n_points: int = 16, dimension: int = 2):
        self.n_points = n_points
        self.dimension = dimension
        self.benchmark_ratio = 0.2786
        self.evolutionary_timeout = 60  # seconds for evolutionary search

    def compute_min_max_ratio(self, points: np.ndarray) -> float:
        """Compute the min/max distance ratio for given points."""
        if points.shape[0] < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = euclidean_distances(points)
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)

        # Compute min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return ratio (avoid division by zero)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist

    def initialize_points(self, strategy: str = "deterministic") -> np.ndarray:
        """Generate initial point configuration."""
        if strategy == "deterministic":
            # Use known good configuration
            return np.array([
                [0.25, 0.25], [0.75, 0.25],
                [0.25, 0.75], [0.75, 0.75],
                [0.1, 0.1], [0.9, 0.1],
                [0.1, 0.9], [0.9, 0.9],
                [0.3, 0.5], [0.7, 0.5],
                [0.5, 0.3], [0.5, 0.7],
                [0.4, 0.4], [0.6, 0.6],
                [0.4, 0.6], [0.6, 0.4]
            ])
        elif strategy == "grid":
            # Generate grid-based initialization
            grid_x = np.linspace(0.1, 0.9, 4)
            grid_y = np.linspace(0.1, 0.9, 4)
            x_grid, y_grid = np.meshgrid(grid_x, grid_y)
            points = np.column_stack([x_grid.ravel(), y_grid.ravel()])[:self.n_points]
            # Add small perturbation
            points += np.random.normal(0, 0.02, points.shape)
            return np.clip(points, 0, 1)
        elif strategy == "random":
            # Random initialization
            return np.random.uniform(0.05, 0.95, (self.n_points, self.dimension))
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

    def constraint_function(self, points_flat: np.ndarray) -> np.ndarray:
        """Constraint function ensuring points stay within [0,1]^2."""
        points = points_flat.reshape(-1, self.dimension)
        # Each point coordinate should be between 0 and 1
        return np.concatenate([
            points[:, 0],  # x coordinates
            points[:, 1],  # y coordinates
            1 - points[:, 0],  # 1 - x coordinates
            1 - points[:, 1]   # 1 - y coordinates
        ])

    def objective_function(self, points_flat: np.ndarray) -> float:
        """Objective function: minimize negative of min/max ratio."""
        return -self.compute_min_max_ratio(points_flat.reshape(-1, self.dimension))

    def optimize_single_start(self, initial_points: np.ndarray,
                            max_iter: int = 500) -> Tuple[np.ndarray, float, float]:
        """Perform optimization from single starting point."""
        start_time = time.time()

        # Flatten for optimization
        initial_flat = initial_points.flatten()

        # Define bounds: 0 <= x_i <= 1, 0 <= y_i <= 1
        bounds = [(0, 1) for _ in range(self.n_points * self.dimension)]

        # Define constraint
        cons = {'type': 'ineq', 'fun': self.constraint_function}

        # First, try L-BFGS-B for coarse optimization
        try:
            result_coarse = minimize(
                self.objective_function,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
            )

            # Then refine with SLSQP
            result_fine = minimize(
                self.objective_function,
                result_coarse.x,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-9}
            )

            end_time = time.time()
            eval_time = end_time - start_time

            # Extract optimized points
            optimized_points = result_fine.x.reshape(-1, self.dimension)
            optimized_points = np.clip(optimized_points, 0, 1)

            final_ratio = self.compute_min_max_ratio(optimized_points)

            return optimized_points, final_ratio, eval_time

        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            # Return original points if optimization fails
            end_time = time.time()
            eval_time = end_time - start_time
            final_ratio = self.compute_min_max_ratio(initial_points)
            return initial_points, final_ratio, eval_time

    def evolutionary_search(self) -> Tuple[np.ndarray, float, float]:
        """Perform evolutionary search for better solutions."""
        # Setup DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Define the individual as a list of points (flattened)
        def create_individual():
            # Initialize with random points in [0,1]^2
            return [random.uniform(0, 1) for _ in range(self.n_points * self.dimension)]

        def evaluate_individual(individual):
            # Convert flattened individual back to points array
            points = np.array(individual).reshape(-1, self.dimension)
            # Ensure points are within bounds
            points = np.clip(points, 0, 1)
            # Return the min/max distance ratio (negative because we want to maximize)
            return (self.compute_min_max_ratio(points),)

        toolbox.register("individual", creator.Individual, create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create initial population
        pop = toolbox.population(n=50)

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Evolution parameters
        n_generations = 50
        start_time = time.time()

        # Main evolutionary loop
        for gen in range(n_generations):
            # Check timeout
            if time.time() - start_time > self.evolutionary_timeout:
                break

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the old population with the new generation
            pop[:] = offspring

        # Find the best individual
        best_ind = tools.selBest(pop, 1)[0]
        best_points = np.array(best_ind).reshape(-1, self.dimension)
        best_points = np.clip(best_points, 0, 1)
        best_ratio = self.compute_min_max_ratio(best_points)
        end_time = time.time()

        return best_points, best_ratio, end_time - start_time

    def optimize_with_multiple_starts(self) -> Tuple[np.ndarray, float, float]:
        """Run optimization with multiple starting strategies and return best result."""
        strategies = ["deterministic", "grid", "random"]
        best_points = None
        best_ratio = -np.inf
        best_time = float('inf')

        # Try each initialization strategy
        for strategy in strategies:
            try:
                initial_points = self.initialize_points(strategy)
                optimized_points, final_ratio, eval_time = self.optimize_single_start(
                    initial_points, max_iter=500
                )

                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()
                    best_time = eval_time

            except Exception as e:
                warnings.warn(f"Strategy {strategy} failed: {e}")
                continue

        # Try evolutionary search if we have some time budget left
        if best_ratio < 0.3 and self.evolutionary_timeout > 0:
            try:
                ev_points, ev_ratio, ev_time = self.evolutionary_search()
                if ev_ratio > best_ratio:
                    best_ratio = ev_ratio
                    best_points = ev_points.copy()
                    best_time = ev_time
            except Exception as e:
                warnings.warn(f"Evolutionary search failed: {e}")
                pass

        # Fallback to deterministic if nothing worked
        if best_points is None:
            fallback_points = self.initialize_points("deterministic")
            return fallback_points, self.compute_min_max_ratio(fallback_points), 0.0

        return best_points, best_ratio, best_time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Create optimizer instance
    optimizer = PointDispersionOptimizer(n_points=16, dimension=2)

    # Perform optimization
    best_points, best_ratio, eval_time = optimizer.optimize_with_multiple_starts()

    # Compute benchmark ratio
    benchmark_ratio = best_ratio / optimizer.benchmark_ratio if optimizer.benchmark_ratio != 0 else 0.0

    # Print metrics
    print(f"Final min/max ratio: {best_ratio:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    print(f"Evaluation time: {eval_time:.6f}s")

    return best_points


# EVOLVE-BLOCK-END