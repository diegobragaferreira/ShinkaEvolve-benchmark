# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import random
import time

import math
from scipy.spatial.transform import Rotation as R

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel Voronoi-evolutionary hybrid approach combining geometric relaxation with genetic operators.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance with numerical stability."""
        if len(points) < 2:
            return 0.0

        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def generate_fibonacci_sphere_points(n_points):
        """Generate points on a sphere using Fibonacci algorithm."""
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def stereographic_project(points_3d):
        """Project 3D points to 2D using stereographic projection from south pole."""
        points_2d = []
        for x, y, z in points_3d:
            # Stereographic projection from south pole (0,0,-1)
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])

        points_2d = np.array(points_2d)

        # Normalize to unit square
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)

        if x_max > x_min and y_max > y_min:
            points_2d[:, 0] = (points_2d[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            points_2d[:, 1] = (points_2d[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05

        return points_2d

    def initialize_population(size=50):
        """Initialize diverse population using multiple geometric patterns."""
        population = []
        np.random.seed(42)

        for i in range(size):
            # Mix of different initialization strategies
            if i % 4 == 0:
                # Grid pattern with perturbations
                grid_size = 4
                x = np.linspace(0.1, 0.9, grid_size)
                y = np.linspace(0.1, 0.9, grid_size)
                xx, yy = np.meshgrid(x, y)
                points = np.column_stack([xx.ravel(), yy.ravel()])
                points = points[:16] if len(points) > 16 else points
                noise = np.random.normal(0, 0.01, points.shape)
                points += noise
                points = np.clip(points, 0, 1)
            elif i % 4 == 1:
                # Spiral pattern
                points = []
                for j in range(16):
                    angle = 2 * np.pi * j / 16
                    radius = 0.4 * (j / 16)
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                    points.append([x, y])
                points = np.array(points)
                # Add noise
                noise = np.random.normal(0, 0.01, points.shape)
                points += noise
                points = np.clip(points, 0, 1)
            elif i % 4 == 2:
                # Spherical Voronoi-based initialization
                points_3d = generate_fibonacci_sphere_points(16)
                points = stereographic_project(points_3d)
            else:
                # Random initialization with clustering avoidance
                points = np.random.rand(16, 2)
                # Slightly perturb to avoid degeneracy
                points += np.random.normal(0, 0.001, points.shape)
                points = np.clip(points, 0, 1)

            population.append(points.copy())

        return population

    def voronoi_fitness(points):
        """Calculate fitness based on Voronoi properties and distance ratio."""
        if len(points) < 2:
            return 0.0

        # Calculate distance ratio
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0

        distance_ratio = min_dist / max_dist

        # Bonus for good Voronoi properties (more uniform cells)
        try:
            vor = Voronoi(points)
            # Calculate variance of Voronoi cell areas as a measure of uniformity
            areas = []
            for i in range(len(points)):
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 2:
                    vertices = np.array([vor.vertices[j] for j in region if j >= 0])
                    if len(vertices) >= 3:
                        # Approximate area using shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        areas.append(area)

            if areas:
                area_variance = np.var(areas)
                # Normalize by max possible area in unit square
                area_uniformity = 1.0 / (1.0 + area_variance)
                return distance_ratio * area_uniformity
            else:
                return distance_ratio

        except:
            return distance_ratio

    def crossover(parent1, parent2):
        """Create offspring using blending crossover with Voronoi-aware mutation."""
        # Blend crossover - mix points from both parents
        alpha = random.random()
        child = parent1 * alpha + parent2 * (1 - alpha)

        # Add some random variation to encourage exploration
        mutation_strength = 0.02
        noise = np.random.normal(0, mutation_strength, child.shape)
        child += noise
        child = np.clip(child, 0, 1)

        return child

    def mutate(individual, mutation_rate=0.1):
        """Mutate individual with Voronoi-aware perturbation."""
        mutated = individual.copy()

        # Apply mutation with probabilistic approach
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Voronoi-aware mutation - consider local neighborhood
                # Add noise proportional to average interpoint distance
                avg_dist = np.mean(squareform(pdist(mutated)))
                noise_magnitude = avg_dist * 0.1 if avg_dist > 0 else 0.01
                noise = np.random.normal(0, noise_magnitude, 2)
                mutated[i] += noise
                mutated[i] = np.clip(mutated[i], 0, 1)

        return mutated

    def voronoi_relaxation(points, iterations=20):
        """Apply Voronoi relaxation with improved convergence."""
        current_points = points.copy()

        for _ in range(iterations):
            try:
                vor = Voronoi(current_points)
                new_points = np.zeros_like(current_points)

                # For each point, calculate the centroid of its Voronoi cell
                for i in range(len(current_points)):
                    region = vor.regions[vor.point_region[i]]

                    if -1 in region or len(region) < 3:
                        # If region is unbounded or too small, keep current position
                        new_points[i] = current_points[i]
                        continue

                    vertices = np.array([vor.vertices[j] for j in region if j >= 0])

                    if len(vertices) < 3:
                        new_points[i] = current_points[i]
                        continue

                    # Calculate centroid
                    centroid = np.mean(vertices, axis=0)
                    new_points[i] = np.clip(centroid, 0, 1)

                # Smooth transition to new positions
                current_points = current_points * 0.8 + new_points * 0.2

            except:
                # Fallback to simple perturbation if Voronoi fails
                current_points += np.random.normal(0, 0.005, current_points.shape)
                current_points = np.clip(current_points, 0, 1)

        return current_points

    def evolutionary_optimization():
        """Main evolutionary optimization loop."""
        # Initialize population
        population = initialize_population(50)

        # Evaluate initial fitness
        fitness_scores = [voronoi_fitness(individual) for individual in population]

        best_individual = population[np.argmax(fitness_scores)]
        best_fitness = max(fitness_scores)

        # Evolutionary parameters
        generations = 100
        elite_size = 10
        tournament_size = 5

        for gen in range(generations):
            # Sort population by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitness = [fitness_scores[i] for i in sorted_indices]

            # Keep elite
            elite = sorted_population[:elite_size]

            # Create new population
            new_population = elite.copy()

            # Generate offspring through tournament selection and crossover
            while len(new_population) < len(population):
                # Tournament selection
                tournament_indices = random.sample(range(len(sorted_population)), tournament_size)
                tournament_fitness = [sorted_fitness[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]

                parent1 = sorted_population[winner_idx]

                # Select second parent
                tournament_indices2 = random.sample(range(len(sorted_population)), tournament_size)
                tournament_fitness2 = [sorted_fitness[i] for i in tournament_indices2]
                winner_idx2 = tournament_indices2[np.argmax(tournament_fitness2)]
                parent2 = sorted_population[winner_idx2]

                # Crossover
                child = crossover(parent1, parent2)

                # Mutation with Voronoi awareness
                child = mutate(child, mutation_rate=0.15)

                # Apply Voronoi relaxation for better geometric properties
                if random.random() < 0.3:  # 30% chance
                    child = voronoi_relaxation(child, iterations=10)

                new_population.append(child)

            population = new_population[:len(population)]

            # Re-evaluate fitness
            fitness_scores = [voronoi_fitness(individual) for individual in population]

            # Update best solution
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx].copy()

            # Early stopping condition
            if gen > 20 and abs(best_fitness - max(fitness_scores)) < 1e-8:
                break

        return best_individual

    # Try evolutionary approach first
    try:
        evolved_solution = evolutionary_optimization()
        evolved_ratio = calculate_min_max_ratio(evolved_solution)

        # Also try hybrid approach with local refinement
        bounds = [(0, 1)] * 32

        def objective(x):
            points = x.reshape(-1, 2)
            distances = squareform(pdist(points))
            np.fill_diagonal(distances, np.inf)
            d_min = np.min(distances)
            d_max = np.max(distances)
            if d_max == 0:
                return -1e10
            return -d_min / d_max

        # Local refinement of the evolutionary solution
        refined = minimize(
            objective,
            evolved_solution.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1)] * 32,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        if refined.success:
            final_points = refined.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            final_ratio = calculate_min_max_ratio(final_points)

            if final_ratio > evolved_ratio:
                return final_points
            else:
                return evolved_solution

        return evolved_solution

    except Exception:
        # Fallback to simpler approach
        pass

    # Fallback to structured initialization with refinement
    try:
        # Initialize with better structured points
        grid_size = 4
        x = np.linspace(0.1, 0.9, grid_size)
        y = np.linspace(0.1, 0.9, grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        points = points[:16]
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        points = np.clip(points, 0, 1)

        # Local refinement
        bounds = [(0, 1)] * 32

        def objective(x):
            points = x.reshape(-1, 2)
            distances = squareform(pdist(points))
            np.fill_diagonal(distances, np.inf)
            d_min = np.min(distances)
            d_max = np.max(distances)
            if d_max == 0:
                return -1e10
            return -d_min / d_max

        refined = minimize(
            objective,
            points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1)] * 32,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        if refined.success:
            final_points = refined.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            return final_points

    except Exception:
        pass

    # Final fallback
    return points

# EVOLVE-BLOCK-END