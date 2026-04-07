# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import math
from collections import defaultdict
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class MultiScaleInitializer:
    """Generates high-quality initial configurations using multi-scale approach"""
    
    @staticmethod
    def generate_voronoi_seeds(n_points: int, boundary_margin: float = 0.05) -> np.ndarray:
        """Generate well-distributed seed points using Voronoi diagram."""
        points = np.random.rand(n_points, 2) * (1 - 2*boundary_margin) + boundary_margin
        return points

    @staticmethod
    def compute_voronoi_radii(points: np.ndarray, boundary_margin: float = 0.05) -> np.ndarray:
        """Compute radii based on Voronoi cell areas."""
        vor = Voronoi(points)
        radii = []
        for i, point in enumerate(points):
            distances = cdist([point], np.delete(points, i, axis=0))[0]
            min_distance = np.min(distances)
            max_radius = min(point[0], 1-point[0], point[1], 1-point[1])
            estimated_radius = min(min_distance/3.0, max_radius * 0.8)
            radii.append(max(estimated_radius, 0.001))
        return np.array(radii)

    @staticmethod
    def generate_hierarchical_initialization(n_circles: int) -> np.ndarray:
        """Generate initial configuration using hierarchical approach"""
        individual = np.zeros((n_circles, 3))
        
        # Stage 1: Voronoi-based coarse placement
        seed_points = MultiScaleInitializer.generate_voronoi_seeds(n_circles)
        radii = MultiScaleInitializer.compute_voronoi_radii(seed_points)
        
        # Stage 2: Fine-tune with structured refinement
        for i in range(n_circles):
            individual[i] = [seed_points[i][0], seed_points[i][1], radii[i]]
            
        # Stage 3: Local optimization to improve initial configuration
        MultiScaleInitializer._refine_initial_placement(individual)
        
        return individual

    @staticmethod
    def _refine_initial_placement(individual: np.ndarray, iterations: int = 10):
        """Refine initial placement using local optimization"""
        n = len(individual)
        for _ in range(iterations):
            # Try to increase radii while maintaining constraints
            for i in range(n):
                x, y, r = individual[i]
                # Compute max possible radius
                max_radius = min(x, 1-x, y, 1-y)
                
                # Only consider increasing radius if beneficial and feasible
                if r < max_radius * 0.95:
                    # Check if increasing radius would cause overlaps
                    can_increase = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = individual[j]
                            dx = x2 - x
                            dy = y2 - y
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance < r + r2:
                                can_increase = False
                                break
                    
                    if can_increase:
                        # Increase radius slightly
                        new_r = min(r * 1.05, max_radius * 0.95)
                        individual[i, 2] = new_r

class SpatialIndexer:
    """Efficient spatial indexing for constraint checking"""
    
    @staticmethod
    def create_spatial_grid(points: np.ndarray, cell_size: float = 0.15) -> dict:
        """Create a spatial grid for efficient neighbor lookups"""
        grid = defaultdict(list)
        for i, (x, y) in enumerate(points):
            grid[(int(x//cell_size), int(y//cell_size))].append(i)
        return grid
    
    @staticmethod
    def get_neighbors_in_range(grid: dict, point: tuple, cell_size: float, 
                             search_radius: float) -> list:
        """Get all indices of points within search radius"""
        x, y = point
        center_cell = (int(x//cell_size), int(y//cell_size))
        neighbors = []
        
        # Check surrounding cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in grid:
                    neighbors.extend(grid[cell])
        
        return neighbors
    
    @staticmethod
    def efficient_overlap_check(individual: np.ndarray, threshold: float = 0.001) -> float:
        """Fast overlap detection using spatial indexing"""
        penalty = 0.0
        n = len(individual)
        
        if n <= 1:
            return penalty
            
        # Create spatial grid
        cell_size = 0.15
        grid = SpatialIndexer.create_spatial_grid(individual[:, :2], cell_size)
        
        # Check overlaps
        for i in range(n):
            x1, y1, r1 = individual[i]
            neighbors = SpatialIndexer.get_neighbors_in_range(grid, (x1, y1), cell_size, r1*2)
            
            for j in neighbors:
                if i >= j:
                    continue
                    
                x2, y2, r2 = individual[j]
                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < r1 + r2:
                    overlap = (r1 + r2) - distance
                    # More severe penalty for smaller overlaps
                    penalty += overlap * 1000 * (1.0 + overlap * 0.05)
        
        return penalty

class ConstraintValidator:
    """Validates constraints with optimized penalty calculation"""
    
    @staticmethod
    def validate_containment(individual: np.ndarray) -> float:
        """Check containment constraints and return penalty"""
        penalty = 0.0
        for i in range(len(individual)):
            x, y, r = individual[i]
            # Calculate boundary violations
            left_violation = max(0, r - x)
            right_violation = max(0, x + r - 1)
            bottom_violation = max(0, r - y)
            top_violation = max(0, y + r - 1)

            if left_violation > 0 or right_violation > 0 or bottom_violation > 0 or top_violation > 0:
                penalty += 5000 * (left_violation + right_violation + bottom_violation + top_violation)
        return penalty

    @staticmethod
    def validate_overlaps(individual: np.ndarray) -> float:
        """Use spatial indexing for efficient overlap validation"""
        return SpatialIndexer.efficient_overlap_check(individual)

    @staticmethod
    def validate_all_constraints(individual: np.ndarray) -> tuple:
        """Validate all constraints and return penalties"""
        containment_penalty = ConstraintValidator.validate_containment(individual)
        overlap_penalty = ConstraintValidator.validate_overlaps(individual)
        return containment_penalty, overlap_penalty

class HybridLocalOptimizer:
    """Local optimization using constraint-aware gradient descent"""
    
    @staticmethod
    def optimize_positions(individual: np.ndarray, max_iter: int = 20) -> np.ndarray:
        """Apply local optimization to improve positions and radii"""
        result = individual.copy()
        n = len(result)
        
        # Iterative local improvement
        for iteration in range(max_iter):
            improved = False
            
            # Optimization for each circle
            for i in range(n):
                x_old, y_old, r_old = result[i]
                best_x, best_y, best_r = x_old, y_old, r_old
                best_fitness = np.sum(result[:, 2]) - (
                    ConstraintValidator.validate_containment(result) + 
                    ConstraintValidator.validate_overlaps(result)
                )
                
                # Try small movements
                for dx in [-0.005, -0.002, 0, 0.002, 0.005]:
                    for dy in [-0.005, -0.002, 0, 0.002, 0.005]:
                        x_new = x_old + dx
                        y_new = y_old + dy
                        
                        # Ensure within bounds
                        x_new = np.clip(x_new, r_old, 1-r_old)
                        y_new = np.clip(y_new, r_old, 1-r_old)
                        
                        # Test if this change improves the solution
                        temp_result = result.copy()
                        temp_result[i, 0] = x_new
                        temp_result[i, 1] = y_new
                        
                        # Check if new position is valid
                        if (temp_result[i, 0] - temp_result[i, 2] >= 0 and 
                            temp_result[i, 0] + temp_result[i, 2] <= 1 and
                            temp_result[i, 1] - temp_result[i, 2] >= 0 and
                            temp_result[i, 1] + temp_result[i, 2] <= 1):
                            
                            # Check overlap
                            valid = True
                            for j in range(n):
                                if i != j:
                                    x1, y1, r1 = temp_result[i]
                                    x2, y2, r2 = temp_result[j]
                                    dx = x2 - x1
                                    dy = y2 - y1
                                    distance = math.sqrt(dx*dx + dy*dy)
                                    if distance < r1 + r2:
                                        valid = False
                                        break
                            
                            if valid:
                                temp_fitness = np.sum(temp_result[:, 2]) - (
                                    ConstraintValidator.validate_containment(temp_result) + 
                                    ConstraintValidator.validate_overlaps(temp_result)
                                )
                                
                                if temp_fitness > best_fitness:
                                    best_x, best_y = x_new, y_new
                                    best_fitness = temp_fitness
                                    improved = True
            
            # Apply best improvements
            if improved:
                result[i, 0] = best_x
                result[i, 1] = best_y
                
        return result

class PopulationManager:
    """Handles population operations with diversity maintenance"""
    
    @staticmethod
    def calculate_diversity(population: List[np.ndarray]) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 0.0
            
        # Use average pairwise distance between individuals
        total_distance = 0.0
        count = 0
        
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                dist = np.linalg.norm(population[i].flatten() - population[j].flatten())
                total_distance += dist
                count += 1
                
        return total_distance / count if count > 0 else 0.0
    
    @staticmethod
    def fitness_sharing(population: List[np.ndarray], 
                       fitness_scores: List[float], 
                       sharing_threshold: float = 0.05) -> List[float]:
        """Apply fitness sharing to maintain diversity"""
        n = len(population)
        shared_fitness = fitness_scores.copy()
        
        if n < 2:
            return shared_fitness
            
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate normalized similarity
                    diff = np.linalg.norm(population[i].flatten() - population[j].flatten())
                    similarity = max(0, 1 - diff / sharing_threshold)
                    shared_fitness[i] -= 0.1 * similarity * fitness_scores[i]
                    
        return shared_fitness

class CirclePackingEvolution:
    """Main evolutionary optimization engine"""
    
    def __init__(self, n_circles: int = 26):
        self.n_circles = n_circles
        self.population_size = 150
        self.generations = 500
        self.mutation_rate = 0.15
        self.elite_size = 15
        self.diversity_threshold = 0.02

    def evaluate_fitness(self, individual: np.ndarray) -> float:
        """Evaluate fitness with improved constraint handling"""
        total_radius = np.sum(individual[:, 2])
        
        # Get all penalties
        containment_penalty, overlap_penalty = ConstraintValidator.validate_all_constraints(individual)
        
        return total_radius - containment_penalty - overlap_penalty

    def generate_individual(self) -> np.ndarray:
        """Generate a new individual with enhanced initialization"""
        # Use multi-scale initialization
        individual = MultiScaleInitializer.generate_hierarchical_initialization(self.n_circles)
        
        # Apply local optimization to refine the initial configuration
        individual = HybridLocalOptimizer.optimize_positions(individual)
        
        return individual

    def run_evolution(self) -> np.ndarray:
        """Run the enhanced evolutionary optimization process"""
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = self.generate_individual()
            population.append(individual)

        # Evolutionary loop
        best_fitness_history = []
        last_improvement_gen = 0
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_fitness(individual)
                fitness_scores.append(fitness)

            # Apply fitness sharing for diversity maintenance
            shared_fitness = PopulationManager.fitness_sharing(population, fitness_scores)
            
            # Track best fitness
            best_fitness = max(shared_fitness)
            best_fitness_history.append(best_fitness)

            # Select top individuals (elitism)
            sorted_indices = np.argsort(shared_fitness)[::-1]
            elite = [population[i] for i in sorted_indices[:self.elite_size]]

            # Generate new population
            new_population = elite.copy()

            # Adaptive mutation rate: decrease over time
            adaptive_mutation_rate = self.mutation_rate * (1 - generation / self.generations)
            if adaptive_mutation_rate < 0.02:
                adaptive_mutation_rate = 0.02

            # Fill rest of population through modified evolutionary operations
            while len(new_population) < self.population_size:
                # Tournament selection with better fitness sharing
                parent1_idx = np.random.choice(len(population), 
                                              min(5, len(population)), 
                                              replace=False)
                parent1_fit = [shared_fitness[i] for i in parent1_idx]
                parent1 = population[parent1_idx[np.argmax(parent1_fit)]]
                
                parent2_idx = np.random.choice(len(population), 
                                              min(5, len(population)), 
                                              replace=False)
                parent2_fit = [shared_fitness[i] for i in parent2_idx]
                parent2 = population[parent2_idx[np.argmax(parent2_fit)]]

                # Create children with more sophisticated crossover
                child1, child2 = self._hybrid_crossover(parent1, parent2)
                
                # Apply mutation with adaptive rate
                if random.random() < adaptive_mutation_rate:
                    child1 = self._adaptive_mutation(child1, adaptive_mutation_rate)
                if random.random() < adaptive_mutation_rate:
                    child2 = self._adaptive_mutation(child2, adaptive_mutation_rate)

                # Local optimization
                child1 = HybridLocalOptimizer.optimize_positions(child1)
                child2 = HybridLocalOptimizer.optimize_positions(child2)
                
                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.population_size]

            # Diversity maintenance
            if generation % 20 == 0:
                diversity = PopulationManager.calculate_diversity(population)
                if diversity < self.diversity_threshold:
                    # Introduce new random individuals to maintain diversity
                    for _ in range(5):
                        population.append(self.generate_individual())

            # Early stopping criteria
            if generation > 50 and len(best_fitness_history) >= 10:
                recent_improvement = best_fitness_history[-1] - best_fitness_history[-10]
                if recent_improvement < 0.001:
                    last_improvement_gen += 1
                    if last_improvement_gen > 10:
                        break
                else:
                    last_improvement_gen = 0

            # Print progress
            if generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

        # Return best solution
        final_fitness_scores = [self.evaluate_fitness(ind) for ind in population]
        best_index = np.argmax(final_fitness_scores)
        return population[best_index]

    def _hybrid_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hybrid crossover that preserves good characteristics"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Use uniform crossover with preference for better parent characteristics
        crossover_point = random.randint(1, len(parent1) - 1)

        # Copy segments from different parents based on fitness
        parent1_fitness = self.evaluate_fitness(parent1)
        parent2_fitness = self.evaluate_fitness(parent2)
        
        # Prefer better parent for most segments
        if parent1_fitness > parent2_fitness:
            # Copy mostly from parent1
            child1[crossover_point:, :] = parent1[crossover_point:, :]
            child2[crossover_point:, :] = parent2[crossover_point:, :]
        else:
            # Copy mostly from parent2
            child1[crossover_point:, :] = parent2[crossover_point:, :]
            child2[crossover_point:, :] = parent1[crossover_point:, :]
            
        return child1, child2

    def _adaptive_mutation(self, individual: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Adaptive mutation that varies strength based on generation"""
        result = individual.copy()
        
        # Different mutation strategies for different stages
        if random.random() < 0.3:  # Large mutation for exploration
            for i in range(len(result)):
                if random.random() < mutation_rate * 2:
                    result[i, 0] += np.random.normal(0, 0.02)
                    result[i, 1] += np.random.normal(0, 0.02)
                    result[i, 2] += np.random.normal(0, 0.01)
                    
                    # Keep within bounds
                    result[i, 0] = np.clip(result[i, 0], 0.01, 0.99)
                    result[i, 1] = np.clip(result[i, 1], 0.01, 0.99)
                    result[i, 2] = max(0.001, result[i, 2])
        else:  # Small mutation for exploitation
            for i in range(len(result)):
                if random.random() < mutation_rate:
                    result[i, 0] += np.random.normal(0, 0.005)
                    result[i, 1] += np.random.normal(0, 0.005)
                    result[i, 2] += np.random.normal(0, 0.002)
                    
                    # Keep within bounds
                    result[i, 0] = np.clip(result[i, 0], 0.01, 0.99)
                    result[i, 1] = np.clip(result[i, 1], 0.01, 0.99)
                    result[i, 2] = max(0.001, result[i, 2])

        return result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    engine = CirclePackingEvolution(n_circles=26)
    return engine.run_evolution()

# EVOLVE-BLOCK-END