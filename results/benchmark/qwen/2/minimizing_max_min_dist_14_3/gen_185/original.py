# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import math
import time
from typing import Tuple, Optional
from copy import deepcopy

class PointOptimizer:
    def __init__(self, num_points: int = 14, dimension: int = 3, population_size: int = 30):
        self.num_points = num_points
        self.dimension = dimension
        self.population_size = population_size
        self.best_points = None
        self.best_ratio = 0.0
        self.max_generations = 1000
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

    def fibonacci_sphere(self, samples: int = 14) -> np.ndarray:
        """Generate points distributed evenly on a sphere using Fibonacci method"""
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms

    def calculate_ratio(self, points: np.ndarray) -> float:
        """Calculate the min/max distance ratio"""
        if len(points) < 2:
            return 0.0

        try:
            distances = pdist(points)

            if len(distances) == 0:
                return 0.0

            d_min = np.min(distances)
            d_max = np.max(distances)

            if d_max <= 0:
                return 0.0

            return d_min / d_max
        except Exception:
            return 0.0

    def generate_initial_population(self) -> list:
        """Generate diverse initial population using multiple strategies"""
        population = []

        # Strategy 1: Fibonacci sphere with random perturbations
        fib_points = self.fibonacci_sphere(self.num_points)
        for i in range(self.population_size // 3):
            np.random.seed(i)
            perturbed = fib_points + np.random.normal(0, 0.03, fib_points.shape)
            population.append(self.project_to_sphere(perturbed))

        # Strategy 2: Random points on sphere
        for i in range(self.population_size // 3):
            np.random.seed(i + 1000)
            random_points = np.random.randn(self.num_points, self.dimension)
            population.append(self.project_to_sphere(random_points))

        # Strategy 3: Spherical Voronoi-based initialization
        for i in range(self.population_size // 3):
            np.random.seed(i + 2000)
            # Generate points and create Voronoi diagram
            points = np.random.randn(self.num_points, self.dimension)
            points = self.project_to_sphere(points)

            try:
                # Create SphericalVoronoi diagram
                sv = SphericalVoronoi(points, radius=1.0)
                # Use Voronoi vertices as new points (if enough vertices)
                if len(sv.vertices) >= self.num_points:
                    voronoi_points = sv.vertices[:self.num_points]
                    population.append(self.project_to_sphere(voronoi_points))
                else:
                    population.append(points)
            except:
                population.append(points)

        return population

    def fitness(self, points: np.ndarray) -> float:
        """Fitness function is simply the min/max distance ratio"""
        return self.calculate_ratio(points)

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutate an individual by perturbing some points"""
        mutated = individual.copy()
        num_mutations = max(1, int(self.mutation_rate * self.num_points))

        for _ in range(num_mutations):
            idx = np.random.randint(0, self.num_points)
            # Apply small perturbations in tangent plane and project back
            perturbation = np.random.normal(0, 0.01, 3)

            # Make sure we're not moving out of bounds
            if np.linalg.norm(mutated[idx]) > 0:
                # Project perturbation onto tangent plane
                tangent_perturbation = perturbation - np.dot(perturbation, mutated[idx]) * mutated[idx]
                mutated[idx] = mutated[idx] + tangent_perturbation

            # Project back to sphere
            mutated[idx] = self.project_to_sphere(mutated[idx].reshape(1, 3)).reshape(-1)

        return mutated

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover two parents to produce offspring"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Single-point crossover on the point indices
        crossover_point = np.random.randint(1, self.num_points)

        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])

        # Project children back to sphere
        child1 = self.project_to_sphere(child1)
        child2 = self.project_to_sphere(child2)

        return child1, child2

    def tournament_selection(self, population: list, fitnesses: list, tournament_size: int = 3) -> np.ndarray:
        """Select parent using tournament selection"""
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()

    def optimize_with_evolution(self) -> Tuple[np.ndarray, float]:
        """Evolutionary optimization with SphericalVoronoi enhancements"""
        # Generate initial population
        population = self.generate_initial_population()

        # Evaluate initial population
        fitnesses = [self.fitness(individual) for individual in population]

        # Track best solution
        best_idx = np.argmax(fitnesses)
        self.best_ratio = fitnesses[best_idx]
        self.best_points = population[best_idx].copy()

        # Evolution loop
        for generation in range(self.max_generations):
            # Sort population by fitness (descending)
            sorted_indices = np.argsort(fitnesses)[::-1]
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitnesses = [fitnesses[i] for i in sorted_indices]

            # Create new population
            new_population = []

            # Elitism: keep best individuals
            elite_count = max(1, self.population_size // 6)
            new_population.extend(sorted_population[:elite_count])

            # Generate rest through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.population_size]

            # Evaluate new population
            fitnesses = [self.fitness(individual) for individual in population]

            # Update global best
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_ratio:
                self.best_ratio = fitnesses[best_idx]
                self.best_points = population[best_idx].copy()

            # Adaptive mutation rate based on diversity
            if generation > 50 and generation % 20 == 0:
                diversity = np.std(fitnesses)
                if diversity < 0.01:  # Low diversity, increase mutation
                    self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
                elif diversity > 0.05:  # High diversity, decrease mutation
                    self.mutation_rate = max(0.05, self.mutation_rate * 0.8)

        return self.best_points, self.best_ratio

    def run_optimization(self) -> np.ndarray:
        """Run evolutionary optimization"""
        try:
            points, ratio = self.optimize_with_evolution()
            return points
        except Exception as e:
            # Fallback to previous approach if evolution fails
            print(f"Evolution failed: {e}")
            np.random.seed(42)
            points = np.random.randn(self.num_points, self.dimension)
            points = self.project_to_sphere(points)
            return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = PointOptimizer(num_points=14, dimension=3)
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END