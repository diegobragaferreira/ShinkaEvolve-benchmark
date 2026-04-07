# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.distance import cdist
import random
from typing import List, Tuple
import time
from collections import defaultdict

# Global constants
POPULATION_SIZE = 120
GENERATIONS = 200
INITIAL_TOURNAMENT_SIZE = 3
FINAL_TOURNAMENT_SIZE = 8
MUTATION_RATE_START = 0.15
MUTATION_RATE_END = 0.015
ELITISM_COUNT = 8
MAX_ATTEMPTS = 500
GRID_RESOLUTION = 20  # For spatial indexing

class CircleValidator:
    @staticmethod
    def is_valid_placement(circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and don't overlap"""
        n = len(circles)
        
        # Check containment constraints efficiently
        radii = circles[:, 2]
        positions = circles[:, :2]
        
        # Vectorized containment check
        if np.any(radii <= 0) or np.any(positions[:, 0] < radii) or np.any(positions[:, 0] > 1 - radii) or \
           np.any(positions[:, 1] < radii) or np.any(positions[:, 1] > 1 - radii):
            return False

        # Check overlap constraints using spatial indexing for efficiency
        if n <= 50:  # For small populations, use direct pairwise comparison
            for i in range(n):
                x, y, r = circles[i]
                for j in range(i+1, n):
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < r + r2:
                        return False
        else:  # For larger populations, use spatial index with bounds checking
            tree = cKDTree(positions)
            for i in range(n):
                x, y, r = circles[i]
                # Find nearby circles (within 2*r distance) - this is more efficient than checking all pairs
                indices = tree.query_ball_point([x, y], 2*r)
                for j in indices:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if distance < r + r2:
                            return False

        return True
    
    @staticmethod
    def compute_overlap_penalty(circles: np.ndarray) -> float:
        """Compute penalty based on overlap amount"""
        n = len(circles)
        penalty = 0.0
        
        # Compute overlap penalties for all pairs
        for i in range(n):
            x, y, r = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                overlap = max(0, r + r2 - distance)
                if overlap > 0:
                    penalty += overlap ** 2  # Quadratic penalty to strongly discourage overlaps
        
        return penalty

class CircleInitializer:
    @staticmethod
    def create_voronoi_based_initialization(num_circles: int) -> np.ndarray:
        """Create an initialization using Voronoi diagram for better spacing"""
        circles = np.zeros((num_circles, 3))
        
        # Generate points inside unit square
        points = np.random.rand(num_circles, 2) * 0.8 + 0.1  # Keep away from edges
        
        # Create Voronoi diagram to get natural distribution
        vor = Voronoi(points)
        
        # Get Voronoi vertices as center candidates
        centers = []
        # Use Voronoi cell centroids as initial positions
        for i in range(len(vor.points)):
            if i < num_circles:
                centers.append(vor.points[i])
        
        # If we have fewer centers than required, add random ones
        while len(centers) < num_circles:
            centers.append([random.random(), random.random()])
        
        centers = np.array(centers[:num_circles])
        
        # Refine using Lloyd relaxation
        for _ in range(3):  # Few iterations for speed
            # For each circle, compute its influence area (Voronoi region centroid)
            # Simplified version: just adjust position towards center of mass of nearby circles
            new_centers = centers.copy()
            for i in range(num_circles):
                # Get neighbors (circles within certain distance)
                distances = np.sqrt(np.sum((centers - centers[i])**2, axis=1))
                neighbors = np.where(distances < 0.3)[0]
                if len(neighbors) > 1:
                    # Move towards mean of neighbors
                    mean_pos = np.mean(centers[neighbors], axis=0)
                    # But stay within bounds
                    new_centers[i] = np.clip(mean_pos, [0.01, 0.01], [0.99, 0.99])
            centers = new_centers
        
        # Now assign radii based on distance to nearest neighbor
        for i in range(num_circles):
            x, y = centers[i]
            # Calculate minimum distance to any other circle
            min_dist = float('inf')
            for j in range(num_circles):
                if i != j:
                    x2, y2 = centers[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    min_dist = min(min_dist, dist)
            
            # Radius is half the minimum distance, bounded
            r = min(0.15, min_dist / 2)
            r = max(0.01, r)  # Ensure minimum radius
            
            # Add small random perturbation
            x += (random.random() - 0.5) * 0.02
            y += (random.random() - 0.5) * 0.02
            
            # Keep within bounds
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            circles[i] = [x, y, r]
            
        return circles
    
    @staticmethod
    def create_multi_scale_grid_initialization(num_circles: int) -> np.ndarray:
        """Create a multi-scale grid-based initialization for circles"""
        circles = np.zeros((num_circles, 3))
        
        # Try different grid configurations to find a good initial setup
        configs = [
            (int(np.ceil(np.sqrt(num_circles))), int(np.ceil(num_circles / np.ceil(np.sqrt(num_circles))))),
            (5, 6),
            (6, 5), 
            (4, 7),
            (7, 4)
        ]
        
        best_config = None
        best_score = -np.inf
        
        for rows, cols in configs:
            if rows * cols >= num_circles:
                # Create positions
                grid_positions = []
                for i in range(rows):
                    for j in range(cols):
                        if len(grid_positions) >= num_circles:
                            break
                        x = (j + 0.5) / cols
                        y = (i + 0.5) / rows
                        grid_positions.append((x, y))
                
                if len(grid_positions) >= num_circles:
                    # Calculate score for this configuration
                    score = 0
                    temp_circles = np.zeros((num_circles, 3))
                    for i in range(num_circles):
                        x, y = grid_positions[i]
                        # Add small random perturbation
                        x += (random.random() - 0.5) * 0.02
                        y += (random.random() - 0.5) * 0.02
                        r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                        temp_circles[i] = [x, y, r]
                        score += r
                    
                    if score > best_score:
                        best_score = score
                        best_config = (grid_positions, rows, cols)
        
        if best_config:
            grid_positions, rows, cols = best_config
            for i in range(num_circles):
                x, y = grid_positions[i]
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles[i] = [x, y, r]
        else:
            # Fallback to random initialization
            for i in range(num_circles):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles[i] = [x, y, r]
                
        return circles
    
    @classmethod
    def create_initial_population(cls, pop_size: int, num_circles: int) -> List[np.ndarray]:
        """Create initial population with enhanced initialization"""
        population = []
        
        for _ in range(pop_size):
            # Try Voronoi-based initialization first (better spacing)
            circles = cls.create_voronoi_based_initialization(num_circles)
            
            # Apply progressive local optimization
            improved = CircleOptimizer.progressive_optimize(circles, num_circles)
            
            # Make sure it's valid
            if CircleValidator.is_valid_placement(improved):
                circles = improved
            else:
                # Fallback to grid initialization
                circles = cls.create_multi_scale_grid_initialization(num_circles)
                improved = CircleOptimizer.progressive_optimize(circles, num_circles)
                if CircleValidator.is_valid_placement(improved):
                    circles = improved
            
            population.append(circles)
            
        return population

class CircleOptimizer:
    @staticmethod
    def progressive_optimize(circles: np.ndarray, num_circles: int) -> np.ndarray:
        """Apply progressive optimization stages"""
        circles_copy = circles.copy()
        
        # Stage 1: Quick greedy improvements
        circles_copy = CircleOptimizer.greedy_radius_increase(circles_copy, num_circles)
        
        # Stage 2: Physics-inspired repulsion
        circles_copy = CircleOptimizer.apply_repulsion_force(circles_copy, num_circles)
        
        # Stage 3: Systematic local search
        circles_copy = CircleOptimizer.systematic_local_search(circles_copy, num_circles)
        
        return circles_copy
    
    @staticmethod
    def greedy_radius_increase(circles: np.ndarray, num_circles: int) -> np.ndarray:
        """Quickly increase radius for all circles while avoiding overlaps"""
        circles_copy = circles.copy()
        
        improvement_threshold = 1e-6
        max_iterations = 100
        
        for iteration in range(max_iterations):
            improved = False
            # Process circles in random order
            circle_order = list(range(num_circles))
            random.shuffle(circle_order)
            
            for i in circle_order:
                x, y, r = circles_copy[i]
                
                # Calculate maximum possible radius at current position
                max_r = min(x, 1-x, y, 1-y)
                
                # Try to increase radius as much as possible while avoiding overlaps
                if max_r > r + improvement_threshold:
                    # Try to increase radius as much as possible
                    new_r = max_r
                    
                    # Check overlap constraints with existing circles
                    valid_radius = True
                    for j in range(num_circles):
                        if i != j:
                            x2, y2, r2 = circles_copy[j]
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < new_r + r2:
                                valid_radius = False
                                break
                    
                    if valid_radius:
                        circles_copy[i, 2] = new_r
                        improved = True
                    else:
                        # Try a smaller increase
                        step_size = 0.0005
                        test_r = min(r + step_size, max_r)
                        while test_r > r + improvement_threshold and not valid_radius:
                            valid_radius = True
                            for j in range(num_circles):
                                if i != j:
                                    x2, y2, r2 = circles_copy[j]
                                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                                    if distance < test_r + r2:
                                        valid_radius = False
                                        break
                            
                            if valid_radius:
                                circles_copy[i, 2] = test_r
                                improved = True
                                break
                            else:
                                test_r -= step_size
            
            if not improved:
                break
                
        return circles_copy
    
    @staticmethod
    def apply_repulsion_force(circles: np.ndarray, num_circles: int) -> np.ndarray:
        """Apply repulsive forces to avoid overlaps and improve packing"""
        circles_copy = circles.copy()
        
        # Calculate repulsive forces from neighbors
        for _ in range(50):  # Limited iterations to improve performance
            forces = np.zeros((num_circles, 2))
            for i in range(num_circles):
                x, y, r = circles_copy[i]
                current_r = r
                
                # Calculate repulsive forces from neighbors
                for j in range(num_circles):
                    if i != j:
                        x2, y2, r2 = circles_copy[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if distance < current_r + r2:
                            # Repulsive force to move away from overlapping circle
                            dx = x - x2
                            dy = y - y2
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist > 0:
                                # Normalize and apply repulsive force
                                # Force strength decreases with distance and overlap
                                force_mag = (current_r + r2 - distance) / (dist + 1e-8) * 0.05
                                forces[i, 0] += dx / dist * force_mag
                                forces[i, 1] += dy / dist * force_mag
            
            # Apply forces to positions
            for i in range(num_circles):
                x, y, r = circles_copy[i]
                new_x = x + forces[i, 0]
                new_y = y + forces[i, 1]
                
                # Keep within bounds
                new_x = np.clip(new_x, r, 1-r)
                new_y = np.clip(new_y, r, 1-r)
                circles_copy[i, 0] = new_x
                circles_copy[i, 1] = new_y
        
        return circles_copy
    
    @staticmethod
    def systematic_local_search(circles: np.ndarray, num_circles: int) -> np.ndarray:
        """Systematic local search to fine-tune the configuration"""
        circles_copy = circles.copy()
        
        # Iterate multiple times for fine-tuning
        for iteration in range(50):
            improved = False
            
            # Process circles in random order
            circle_order = list(range(num_circles))
            random.shuffle(circle_order)
            
            for i in circle_order:
                x, y, r = circles_copy[i]
                
                # Try to slightly improve position
                best_x, best_y, best_r = x, y, r
                best_score = r  # Maximize radius
                
                # Try nearby positions
                for dx in [-0.005, -0.002, 0, 0.002, 0.005]:
                    for dy in [-0.005, -0.002, 0, 0.002, 0.005]:
                        if abs(dx) + abs(dy) == 0:
                            continue
                            
                        test_x = x + dx
                        test_y = y + dy
                        
                        # Keep within bounds
                        if test_x < r or test_x > 1-r or test_y < r or test_y > 1-r:
                            continue
                            
                        # Check validity of new configuration
                        valid = True
                        temp_circles = circles_copy.copy()
                        temp_circles[i, 0] = test_x
                        temp_circles[i, 1] = test_y
                        
                        for j in range(num_circles):
                            if i != j:
                                x2, y2, r2 = temp_circles[j]
                                distance = np.sqrt((test_x - x2)**2 + (test_y - y2)**2)
                                if distance < r + r2:
                                    valid = False
                                    break
                                    
                        if valid:
                            # Calculate score (radius) for this position
                            score = r  # Just radius for now
                            if score > best_score:
                                best_score = score
                                best_x, best_y = test_x, test_y
                
                # Update if beneficial
                if best_x != x or best_y != y:
                    circles_copy[i, 0] = best_x
                    circles_copy[i, 1] = best_y
                    improved = True
            
            if not improved:
                break
                
        return circles_copy

class EvolutionEngine:
    @staticmethod
    def constraint_aware_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover with constraint awareness - weigh based on overlap risk"""
        n = len(parent1)
        child = np.zeros_like(parent1)
        
        # First, assess overlap risk for each circle pair
        overlap_risk_scores = np.zeros(n)
        for i in range(n):
            x1, y1, r1 = parent1[i]
            x2, y2, r2 = parent2[i]
            
            # Calculate distance between corresponding circles
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            overlap_risk = max(0, r1 + r2 - distance)
            overlap_risk_scores[i] = overlap_risk
        
        # Use weighted crossover where circles with lower overlap risk
        # are more likely to be inherited from the same parent
        for i in range(n):
            # Probability of inheriting from parent1 inversely proportional to overlap risk
            if overlap_risk_scores[i] < 0.01:
                # Low overlap risk, choose randomly
                inherit_from_parent1 = random.random() < 0.5
            elif overlap_risk_scores[i] < 0.05:
                # Medium overlap risk, biased towards parent1
                inherit_from_parent1 = random.random() < 0.7
            else:
                # High overlap risk, biased towards parent2
                inherit_from_parent1 = random.random() < 0.3
            
            if inherit_from_parent1:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        
        # Post-crossover constraint validation and repair
        repaired = CircleRepair.repair_constraints(child)
        return repaired

    @staticmethod
    def adaptive_mutation(individual: np.ndarray, generation: int, diversity: float) -> np.ndarray:
        """Apply adaptive mutation with diversity-aware parameters"""
        mutated = individual.copy()
        n = len(mutated)
        
        # Adaptive mutation rate based on generation and diversity
        mutation_rate = MUTATION_RATE_START - (MUTATION_RATE_START - MUTATION_RATE_END) * (generation / GENERATIONS)
        # Increase mutation rate when diversity is low (for exploration)
        if diversity < 0.25:
            mutation_rate *= 1.3
            
        mutation_mask = np.random.random(n) < mutation_rate
        
        # Separate position and radius mutations with adaptive weights
        pos_mutations = mutation_mask & (np.random.random(n) < 0.5)
        rad_mutations = mutation_mask & ~pos_mutations
        
        # Mutate positions with adaptive step sizes
        if np.any(pos_mutations):
            step_size = 0.1 * (1.0 - generation / GENERATIONS) + 0.02  # Decrease over time but keep higher base
            mutated[pos_mutations, 0] += (np.random.random(np.sum(pos_mutations)) - 0.5) * step_size
            mutated[pos_mutations, 1] += (np.random.random(np.sum(pos_mutations)) - 0.5) * step_size
            
            # Keep within bounds
            mutated[pos_mutations, 0] = np.clip(mutated[pos_mutations, 0], 0.01, 0.99)
            mutated[pos_mutations, 1] = np.clip(mutated[pos_mutations, 1], 0.01, 0.99)
        
        # Mutate radii with adaptive step sizes
        if np.any(rad_mutations):
            rad_step_size = 0.04 * (1.0 - generation / GENERATIONS) + 0.01
            mutated[rad_mutations, 2] += (np.random.random(np.sum(rad_mutations)) - 0.5) * rad_step_size
            # Ensure positive radius
            mutated[rad_mutations, 2] = np.maximum(0.001, mutated[rad_mutations, 2])
        
        # Repair any constraint violations
        repaired = CircleRepair.repair_constraints(mutated)
        return repaired

class CircleRepair:
    @staticmethod
    def repair_constraints(circles: np.ndarray) -> np.ndarray:
        """Repair any constraint violations with enhanced repair strategy"""
        repaired = circles.copy()
        n = len(repaired)
        
        # Ensure all circles are within bounds and have positive radius
        for i in range(n):
            x, y, r = repaired[i]
            r = max(0.001, r)
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            repaired[i] = [x, y, r]
        
        # Apply constraint repair with early termination
        for _ in range(8):  # Reduced iterations for performance
            any_changes = False
            for i in range(n):
                x, y, r = repaired[i]
                # Check overlaps and adjust if needed
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = repaired[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        min_distance = r + r2
                        if distance < min_distance:
                            # Move circle away from overlapping one
                            dx = x2 - x
                            dy = y2 - y
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist > 0:
                                factor = (min_distance - distance) / dist * 0.1
                                x += dx * factor
                                y += dy * factor
                                any_changes = True
                
                # Keep within bounds
                r = max(0.001, r)
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                repaired[i] = [x, y, r]
            
            if not any_changes:
                break
        
        return repaired

class CirclePack26:
    def __init__(self):
        self.best_solution = None
        self.best_fitness = -np.inf
        self.start_time = time.time()
        self.max_time = 60  # seconds
    
    def calculate_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate population diversity metric"""
        if len(population) < 2:
            return 1.0
        
        # Calculate average distance between individuals
        total_distances = 0
        count = 0
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                diff = np.abs(population[i] - population[j])
                distance = np.sum(diff)
                total_distances += distance
                count += 1
        
        if count == 0:
            return 1.0
            
        avg_distance = total_distances / count
        # Normalize diversity to [0, 1] range
        normalized = min(1.0, avg_distance / 10.0)  # Assuming max distance is around 10
        return normalized
    
    def evaluate_fitness(self, individual: np.ndarray) -> float:
        """Evaluate fitness of an individual - maximize sum of radii, 
        but penalize overlap violations"""
        if CircleValidator.is_valid_placement(individual):
            sum_radii = np.sum(individual[:, 2])
            return sum_radii
        else:
            # For invalid solution, apply penalty based on overlap magnitude
            overlap_penalty = CircleValidator.compute_overlap_penalty(individual)
            sum_radii = np.sum(individual[:, 2])
            return sum_radii - overlap_penalty * 100  # Strong penalty
    
    def should_terminate(self) -> bool:
        """Check if we should terminate due to time limit"""
        return time.time() - self.start_time > self.max_time * 0.95  # Leave some buffer
    
    def evolve(self) -> np.ndarray:
        """Main evolutionary loop with adaptive parameters and early termination"""
        # Create initial population
        population = CircleInitializer.create_initial_population(POPULATION_SIZE, 26)
        
        # Track diversity for adaptive behavior
        previous_diversity = 0.0
        
        for generation in range(GENERATIONS):
            # Check early termination
            if self.should_terminate():
                break
                
            # Evaluate fitness of each individual
            fitnesses = [self.evaluate_fitness(individual) for individual in population]
            
            # Track best solution so far
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_solution = population[max_fitness_idx].copy()
            
            # Calculate population diversity for adaptive behavior
            diversity = self.calculate_diversity(population)
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
            elites = [population[i].copy() for i in elite_indices]
            
            # Create new population
            new_population = elites.copy()
            
            # Generate offspring through selection, crossover, and mutation
            while len(new_population) < POPULATION_SIZE:
                # Selection with adaptive tournament size
                tournament_size = INITIAL_TOURNAMENT_SIZE + int((FINAL_TOURNAMENT_SIZE - INITIAL_TOURNAMENT_SIZE) * (generation / GENERATIONS))
                parent1 = self.adaptive_tournament_selection(population, fitnesses, tournament_size, [diversity])
                parent2 = self.adaptive_tournament_selection(population, fitnesses, tournament_size, [diversity])
                
                # Crossover with constraint awareness
                child = EvolutionEngine.constraint_aware_crossover(parent1, parent2)
                
                # Mutation with adaptive parameters
                child = EvolutionEngine.adaptive_mutation(child, generation, diversity)
                
                # Add to new population
                new_population.append(child)
            
            population = new_population[:POPULATION_SIZE]
            
            # Early stopping based on diversity stagnation
            if generation > 10 and abs(diversity - previous_diversity) < 0.001:
                # Diversity hasn't changed much, potentially stuck in local optimum
                pass  # Continue evolution to see if it improves
            
            previous_diversity = diversity
        
        # Return the best solution found
        if self.best_solution is not None:
            return self.best_solution
        else:
            # Fallback to final population if no valid solution was found
            return population[0]
    
    def adaptive_tournament_selection(self, population: List[np.ndarray], fitnesses: List[float],
                                     tournament_size: int, diversities: List[float]) -> np.ndarray:
        """Select individual using adaptive tournament selection based on population diversity"""
        # Scale tournament size based on diversity
        scaled_tournament = max(3, min(10, tournament_size + int(diversities[0] * 3)))
        tournament_indices = random.sample(range(len(population)), scaled_tournament)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    packer = CirclePack26()
    return packer.evolve()

# EVOLVE-BLOCK-END