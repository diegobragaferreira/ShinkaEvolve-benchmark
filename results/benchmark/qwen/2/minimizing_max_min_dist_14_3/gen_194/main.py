# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
import math
import time
from typing import Tuple, List, Optional
import warnings

class VoronoiEvaluator:
    """Handles evaluation of point configurations using Voronoi-based metrics"""
    
    @staticmethod
    def calculate_min_max_ratio(points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum pairwise distances."""
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
    
    @staticmethod
    def voronoi_entropy_score(points: np.ndarray) -> float:
        """
        Calculate entropy-based score of Voronoi cell distribution.
        High entropy indicates more uniform cell distribution.
        """
        try:
            sv = SphericalVoronoi(points)
            areas = sv.calculate_areas()
            # Normalize areas
            areas = areas / np.sum(areas)
            # Entropy calculation
            entropy = -np.sum(areas * np.log(areas + 1e-10))
            return entropy
        except Exception:
            # Fallback for cases where SphericalVoronoi fails
            return 0.0
    
    @classmethod
    def combined_fitness(cls, points: np.ndarray, ratio_weight: float = 1.0, 
                        uniformity_weight: float = 0.1) -> float:
        """Combined fitness function incorporating both distance ratio and uniformity."""
        ratio = cls.calculate_min_max_ratio(points)
        uniformity = cls.voronoi_entropy_score(points)
        return ratio_weight * ratio + uniformity_weight * uniformity

class PointInitializer:
    """Handles various methods of initializing point configurations"""
    
    @staticmethod
    def fibonacci_sphere(samples: int = 14) -> np.ndarray:
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
    
    @classmethod
    def initialize_multiple_strategies(cls, num_points: int = 14) -> List[Tuple[str, np.ndarray]]:
        """Generate multiple initialization strategies for diverse starting points"""
        strategies = []
        
        # Strategy 1: Fibonacci sphere with multiple seeds
        for seed in [42, 123, 456, 789]:
            np.random.seed(seed)
            fib_points = cls.fibonacci_sphere(num_points)
            # Scale to unit cube for better initialization
            fib_points = (fib_points * 0.5 + 0.5).clip(0, 1)
            strategies.append(("fibonacci_seed_" + str(seed), fib_points))
        
        # Strategy 2: Random initialization
        for seed in [999, 111, 222, 333]:
            np.random.seed(seed)
            random_points = np.random.rand(num_points, 3)
            strategies.append(("random_seed_" + str(seed), random_points))
            
        # Strategy 3: Uniform distribution
        uniform_points = np.random.uniform(0.1, 0.9, (num_points, 3))
        strategies.append(("uniform", uniform_points))
        
        return strategies

class EvolutionaryOptimizer:
    """Implements advanced evolutionary optimization with Voronoi-based fitness"""
    
    def __init__(self, population_size: int = 20, max_generations: int = 2000):
        self.population_size = population_size
        self.max_generations = max_generations
        self.evaluator = VoronoiEvaluator()
    
    def generate_neighbor_config(self, current_points: np.ndarray, 
                               perturbation_strength: float = 0.05) -> np.ndarray:
        """Generate neighbor configuration with spherical constraint handling"""
        neighbor_points = current_points.copy()

        # Select random points to modify
        num_modify = max(1, len(current_points) // 4)
        indices_to_modify = np.random.choice(len(current_points), num_modify, replace=False)

        for idx in indices_to_modify:
            # Generate perturbation that preserves spherical nature using tangent plane projection
            random_vec = np.random.randn(3)
            normal_vec = current_points[idx]
            tangent_vec = random_vec - np.dot(random_vec, normal_vec) * normal_vec
            # Normalize tangent vector
            tangent_norm = np.linalg.norm(tangent_vec)
            if tangent_norm > 1e-10:
                tangent_vec = tangent_vec / tangent_norm
            # Apply perturbation
            perturbation = tangent_vec * np.random.normal(0, perturbation_strength)
            neighbor_points[idx] += perturbation
            # Project back to sphere ensuring numerical stability
            norm = np.linalg.norm(neighbor_points[idx])
            if norm > 1e-10:
                neighbor_points[idx] = neighbor_points[idx] / norm

        return neighbor_points
    
    def evaluate_population(self, population: List[np.ndarray]) -> List[float]:
        """Evaluate fitness for entire population"""
        return [self.evaluator.combined_fitness(individual) for individual in population]
    
    def tournament_selection(self, population: List[np.ndarray], 
                           fitness_scores: List[float], tournament_size: int = 3) -> List[np.ndarray]:
        """Perform tournament selection with fitness proportionality"""
        selected = []
        for _ in range(len(population)):
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        return selected
    
    def evolve(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Main evolutionary optimization loop"""
        # Initialize population
        population = [initial_points.copy()]
        for i in range(self.population_size - 1):
            individual = self.generate_neighbor_config(initial_points, 0.1 + 0.05 * i)
            population.append(individual)
        
        best_individual = None
        best_fitness = -np.inf
        
        # Track history for adaptive mechanisms
        history = []
        
        for generation in range(self.max_generations):
            # Evaluate fitness for all individuals
            fitness_scores = self.evaluate_population(population)
            
            # Track best solution
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            current_best_individual = population[current_best_idx].copy()
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()
            
            # Track history
            history.append(best_fitness)
            if len(history) > 10:
                history.pop(0)
            
            # Selection with adaptive pressure
            selected_population = self.tournament_selection(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            
            # Elitism: preserve best individual
            new_population.append(best_individual.copy())
            
            for i in range(0, len(selected_population) - 1, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i+1] if i+1 < len(selected_population) else selected_population[0]
                
                # Blend crossover
                alpha = np.random.random()
                child1 = parent1 * alpha + parent2 * (1 - alpha)
                child2 = parent2 * alpha + parent1 * (1 - alpha)
                
                # Project children back to valid space
                for j in range(len(child1)):
                    norm = np.linalg.norm(child1[j])
                    if norm > 1e-10:
                        child1[j] = child1[j] / norm
                    norm = np.linalg.norm(child2[j])
                    if norm > 1e-10:
                        child2[j] = child2[j] / norm
                
                # Adaptive mutation
                generation_ratio = generation / self.max_generations
                mutation_strength = 0.05 * (1 - generation_ratio)
                
                child1 = self.generate_neighbor_config(child1, mutation_strength)
                child2 = self.generate_neighbor_config(child2, mutation_strength)
                
                new_population.extend([child1, child2])
            
            # Trim population to exact size
            population = new_population[:self.population_size]
            
            # Diversity maintenance
            if generation % 100 == 0 and generation > 0:
                # Add diversity by introducing random individuals
                for _ in range(2):
                    random_individual = self.generate_neighbor_config(initial_points, 0.2)
                    if len(population) < self.population_size:
                        population.append(random_individual)
                    else:
                        # Replace worst performers
                        worst_indices = np.argsort(fitness_scores)[:2]
                        for idx in worst_indices:
                            if idx < len(population):
                                population[idx] = self.generate_neighbor_config(initial_points, 0.25)
            
            # Early termination check
            if len(history) >= 3 and all(abs(history[-1] - x) < 1e-8 for x in history[-3:]):
                break
                
        return best_individual, best_fitness

class GradientRefiner:
    """Implements multiple refinement strategies with gradient-based optimization"""
    
    def __init__(self, evaluator: VoronoiEvaluator):
        self.evaluator = evaluator
    
    def refine_with_l_bfgs(self, points: np.ndarray, max_iter: int = 500) -> Tuple[np.ndarray, float]:
        """Refine using L-BFGS optimization"""
        def objective(x_flat):
            points_local = x_flat.reshape(-1, 3)
            # Keep points on unit cube constraint
            points_local = np.clip(points_local, 0, 1)
            return -self.evaluator.calculate_min_max_ratio(points_local)
        
        try:
            result = minimize(
                objective, 
                points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8},
                tol=1e-6
            )
            refined_points = result.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            return refined_points, -result.fun
        except Exception:
            return points, self.evaluator.calculate_min_max_ratio(points)
    
    def iterative_refinement(self, points: np.ndarray, max_iter: int = 1000) -> Tuple[np.ndarray, float]:
        """Simple iterative refinement with sphere constraint"""
        current_points = points.copy()
        current_ratio = self.evaluator.calculate_min_max_ratio(current_points)
        best_ratio = current_ratio
        best_points = current_points.copy()
        
        # Iterative improvement with careful constraint handling
        for iteration in range(max_iter):
            neighbor_points = current_points.copy()
            point_idx = np.random.randint(len(neighbor_points))
            
            # Small perturbation with adaptive scaling
            perturbation_magnitude = 0.001 * (1.0 - iteration/max_iter)
            perturbation = np.random.normal(0, perturbation_magnitude, 3)
            neighbor_points[point_idx] += perturbation
            
            # Clamp to unit cube
            neighbor_points = np.clip(neighbor_points, 0, 1)
            
            new_ratio = self.evaluator.calculate_min_max_ratio(neighbor_points)
            
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = neighbor_points.copy()
                current_points = neighbor_points.copy()
        
        return best_points, best_ratio

class PointOptimizer:
    """Main optimization controller orchestrating the complete workflow"""
    
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.evaluator = VoronoiEvaluator()
        self.initializer = PointInitializer()
        self.evolutionary = EvolutionaryOptimizer(population_size=20, max_generations=1500)
        self.refiner = GradientRefiner(self.evaluator)
        self.best_points = None
        self.best_ratio = 0.0
    
    def project_to_unit_cube(self, points: np.ndarray) -> np.ndarray:
        """Project points to unit cube [0,1]^3"""
        # Find min/max along each axis
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Handle case where there's no variation
        ranges = max_coords - min_coords
        if np.any(ranges == 0):
            # If any dimension has no variation, return points centered at 0.5
            return np.full_like(points, 0.5)

        # Scale to [0,1] range
        normalized = (points - min_coords) / ranges

        # Ensure they're clipped to [0,1]
        return np.clip(normalized, 0, 1)
    
    def run_multi_start_optimization(self) -> np.ndarray:
        """Run optimization from multiple starting points"""
        # Get multiple initialization strategies
        strategies = self.initializer.initialize_multiple_strategies(self.num_points)
        
        # Run evolutionary optimization from each starting point
        best_points = None
        best_ratio = 0.0
        
        for strategy_name, initial_points in strategies:
            try:
                # Apply evolutionary optimization
                evolved_points, evolved_ratio = self.evolutionary.evolve(initial_points)
                
                # Final refinement
                refined_points, refined_ratio = self.refiner.refine_with_l_bfgs(evolved_points)
                
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
                    
                # Also try iterative refinement as fallback
                iter_points, iter_ratio = self.refiner.iterative_refinement(evolved_points)
                if iter_ratio > best_ratio:
                    best_ratio = iter_ratio
                    best_points = iter_points.copy()
                    
            except Exception as e:
                # Continue with other strategies if one fails
                continue
        
        # If no good solution found, fall back to simple approach
        if best_points is None:
            # Simple Fibonacci initialization + refinement
            fib_points = self.initializer.fibonacci_sphere(self.num_points)
            fib_points = (fib_points * 0.5 + 0.5).clip(0, 1)
            best_points, best_ratio = self.refiner.refine_with_l_bfgs(fib_points)
            
        return best_points
    
    def run_optimization(self) -> np.ndarray:
        """Main entry point for optimization"""
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Run multi-start optimization
        optimized_points = self.run_multi_start_optimization()
        
        # Final projection to unit cube
        final_points = self.project_to_unit_cube(optimized_points)
        
        return final_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = PointOptimizer(num_points=14, dimension=3)
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END