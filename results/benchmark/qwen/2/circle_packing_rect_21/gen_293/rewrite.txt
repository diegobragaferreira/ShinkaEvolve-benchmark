# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random
from typing import Tuple, List
import time
import warnings

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH = 1.2  # Optimized rectangle dimensions from high-performing implementations
RECT_HEIGHT = 0.8
NUM_CIRCLES = 21
POPULATION_SIZE = 80
MAX_GENERATIONS = 150
INITIAL_MUTATION_RATE = 0.3
TOURNAMENT_SIZE = 5
SEED = 42
PHYSICS_REFINEMENT_STEPS = 300
LOCAL_OPTIMIZATION_ITERATIONS = 100

class CirclePackingOptimizer:
    def __init__(self, width: float = RECT_WIDTH, height: float = RECT_HEIGHT,
                 num_circles: int = NUM_CIRCLES):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.rect_area = width * height
        self.best_solution = None
        self.best_fitness = float('-inf')
        
        # Initialize random seed for reproducibility
        np.random.seed(SEED)
        random.seed(SEED)

    def is_valid_position(self, x: float, y: float, r: float) -> bool:
        """Check if circle center is within bounds"""
        return (r <= x <= self.width - r and 
                r <= y <= self.height - r)

    def is_valid_circle(self, x: float, y: float, r: float) -> bool:
        """Check if circle is valid (within bounds and positive radius)"""
        return (0 < r and 
                self.is_valid_position(x, y, r))

    def check_overlap_simple(self, circles: np.ndarray, idx1: int, idx2: int) -> bool:
        """Simple overlap check using precomputed squared distances"""
        x1, y1, r1 = circles[idx1]
        x2, y2, r2 = circles[idx2]
        dx = x1 - x2
        dy = y1 - y2
        dist_sq = dx*dx + dy*dy
        radius_sum = r1 + r2
        return dist_sq < radius_sum * radius_sum

    def check_all_overlaps(self, circles: np.ndarray) -> int:
        """Efficiently check all overlaps using spatial indexing"""
        violations = 0
        try:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)
            
            # Find neighbors within 2*max_radius distance
            max_radius = np.max(circles[:, 2])
            if max_radius > 0:
                pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
                
                for i, j in pairs:
                    if self.check_overlap_simple(circles, i, j):
                        violations += 1
                        
        except Exception as e:
            warnings.warn(f"Error in overlap checking: {e}")
            # Fallback to brute force when spatial indexing fails
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    if self.check_overlap_simple(circles, i, j):
                        violations += 1
        return violations

    def calculate_total_radius_sum(self, circles: np.ndarray) -> float:
        """Calculate sum of all circle radii"""
        return np.sum(circles[:, 2])

    def calculate_fitness(self, circles: np.ndarray) -> Tuple[float, int]:
        """
        Calculate fitness: sum of radii with penalty for constraint violations
        
        Returns:
            Tuple of (fitness_score, number_of_violations)
        """
        # Quick boundary check first
        boundary_violations = 0
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r):
                boundary_violations += 1000  # Heavy penalty

        # If boundary violations, return immediately with penalty
        if boundary_violations > 0:
            return -1.0, boundary_violations

        total_radius = self.calculate_total_radius_sum(circles)
        
        # Check overlap violations
        overlap_violations = self.check_all_overlaps(circles)
        
        # Return fitness score with penalties
        penalty_weight = 5000.0  # Increased penalty for stricter enforcement
        return total_radius - (penalty_weight * overlap_violations), overlap_violations

    def initialize_population(self, pop_size: int) -> List[np.ndarray]:
        """Generate diverse initial population with strategic grid placement"""
        population = []
        
        # Determine grid configuration for 21 circles
        grid_rows = int(np.ceil(np.sqrt(self.num_circles * 1.2)))  # Slight adjustment for better spacing
        grid_cols = int(np.ceil(self.num_circles / grid_rows))
        
        # Ensure sufficient grid space
        while grid_rows * grid_cols < self.num_circles:
            grid_rows += 1
            
        # Calculate spacing with padding
        cell_width = self.width / (grid_cols + 1.5)
        cell_height = self.height / (grid_rows + 1.5)
        
        # Generate population
        for _ in range(pop_size):
            circles = np.zeros((self.num_circles, 3))
            
            # Grid-based initialization with randomness
            idx = 0
            for i in range(grid_rows):
                for j in range(grid_cols):
                    if idx >= self.num_circles:
                        break
                    # Position with more substantial randomization
                    x = (j + 1) * cell_width + np.random.uniform(-cell_width * 0.3, cell_width * 0.3)
                    y = (i + 1) * cell_height + np.random.uniform(-cell_height * 0.3, cell_height * 0.3)
                    # Base radius with more variation
                    r = min(cell_width, cell_height) * 0.32 + np.random.uniform(-0.03, 0.03)
                    r = max(0.01, min(0.2, r))
                    
                    # Ensure within bounds
                    x = max(r, min(self.width - r, x))
                    y = max(r, min(self.height - r, y))
                    
                    circles[idx] = [x, y, r]
                    idx += 1
            
            # Fill remaining positions strategically
            for i in range(idx, self.num_circles):
                # Better fallback placement strategy
                attempts = 0
                max_attempts = 100
                while attempts < max_attempts:
                    # Try to place in a region that maximizes available space
                    x = np.random.uniform(0.05, self.width - 0.05)
                    y = np.random.uniform(0.05, self.height - 0.05)
                    
                    # Estimate minimum radius based on proximity to existing circles
                    min_dist = float('inf')
                    for k in range(i):
                        dx = x - circles[k, 0]
                        dy = y - circles[k, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = min(min_dist, dist)
                    
                    # Conservative radius estimation
                    if min_dist > 0.1:
                        r = min(0.12, min_dist * 0.25)
                    elif min_dist > 0.05:
                        r = min(0.08, min_dist * 0.3)
                    else:
                        # Near another circle, use smaller radius
                        r = np.random.uniform(0.02, 0.06)
                        
                    r = max(0.01, min(0.2, r))
                    
                    # Verify validity
                    if self.is_valid_circle(x, y, r):
                        circles[i] = [x, y, r]
                        break
                    attempts += 1
                
                # Last resort: random placement
                if attempts >= max_attempts:
                    x = np.random.uniform(0.05, self.width - 0.05)
                    y = np.random.uniform(0.05, self.height - 0.05)
                    r = np.random.uniform(0.01, 0.15)
                    circles[i] = [x, y, r]
            
            population.append(circles)
            
        return population

    def tournament_selection(self, population: List[np.ndarray], 
                           fitness_scores: List[Tuple[float, int]]) -> np.ndarray:
        """Tournament selection with better diversity"""
        tournament_indices = np.random.choice(len(population), min(TOURNAMENT_SIZE, len(population)), replace=False)
        tournament_fitness = [(i, fitness_scores[i][0]) for i in tournament_indices]
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)
        return population[tournament_fitness[0][0]].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Uniform crossover with better trait preservation"""
        child = parent1.copy()
        
        # Use better parent's traits more often
        parent1_fitness, _ = self.calculate_fitness(parent1)
        parent2_fitness, _ = self.calculate_fitness(parent2)
        
        # Select better parent
        better_parent = parent1 if parent1_fitness >= parent2_fitness else parent2
        
        # 75% chance to inherit from better parent (more selective)
        mask = np.random.rand(self.num_circles) > 0.25
        
        for i in range(self.num_circles):
            if mask[i]:
                child[i] = better_parent[i].copy()
                
        return child

    def mutate(self, individual: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Mutation with adaptive step sizing and better balance"""
        mutated = individual.copy()
        
        for i in range(self.num_circles):
            if np.random.rand() < mutation_rate:
                # 60% position mutation, 40% radius mutation (balanced)
                if np.random.rand() < 0.6:
                    # Position mutation with adaptive step
                    step = 0.15 * (1.0 - mutation_rate)  # Larger steps for early stages
                    mutated[i, 0] = np.random.uniform(
                        max(0.001, mutated[i, 0] - step),
                        min(self.width - 0.001, mutated[i, 0] + step)
                    )
                    mutated[i, 1] = np.random.uniform(
                        max(0.001, mutated[i, 1] - step),
                        min(self.height - 0.001, mutated[i, 1] + step)
                    )
                else:
                    # Radius mutation with careful bounds
                    step = 0.05 * (1.0 - mutation_rate)
                    mutated[i, 2] = np.random.uniform(
                        max(0.001, mutated[i, 2] - step),
                        min(0.2, mutated[i, 2] + step)
                    )
                    
        return mutated

    def physics_refinement(self, circles: np.ndarray, steps: int = PHYSICS_REFINEMENT_STEPS) -> np.ndarray:
        """Physics-based refinement to improve solution quality with simpler approach"""
        # Create working copy
        refined_circles = circles.copy()
        positions = refined_circles[:, :2].copy()
        radii = refined_circles[:, 2].copy()
        
        # Precompute maximum relevant distance for neighbor search
        max_radius = np.max(radii) if np.max(radii) > 0 else 0.1
        
        # Physics parameters
        boundary_force = 100.0
        repulsion_force = 200.0
        dt = 0.01
        
        # Iterative physics simulation
        for step in range(steps):
            # Calculate forces for each circle
            forces = np.zeros_like(positions)
            
            # Direct pairwise computation for simplicity and reliability
            # Only consider nearby circles for performance
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Consider collision avoidance for nearby circles only
                    if distance < (radii[i] + radii[j]) and distance > 1e-8:
                        # Repulsive force
                        force_magnitude = repulsion_force * (1.0 - distance/(radii[i] + radii[j]))
                        fx = force_magnitude * dx / distance
                        fy = force_magnitude * dy / distance
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # Apply boundary forces
            for i in range(self.num_circles):
                x, y, r = positions[i, 0], positions[i, 1], radii[i]
                
                # Boundary forces with stronger repulsion
                if x - r < 0:
                    forces[i, 0] += boundary_force * (0 - (x - r))
                if x + r > self.width:
                    forces[i, 0] += boundary_force * (self.width - (x + r))
                if y - r < 0:
                    forces[i, 1] += boundary_force * (0 - (y - r))
                if y + r > self.height:
                    forces[i, 1] += boundary_force * (self.height - (y + r))
            
            # Update positions with velocity-based approach
            for i in range(self.num_circles):
                # Apply force to update velocity and position
                positions[i, 0] += forces[i, 0] * dt
                positions[i, 1] += forces[i, 1] * dt
                
                # Keep within bounds
                positions[i, 0] = np.clip(positions[i, 0], r, self.width - r)
                positions[i, 1] = np.clip(positions[i, 1], r, self.height - r)
            
            # Update circles
            refined_circles[:, :2] = positions
            
            # Early termination if forces are small
            if step > 10 and step % 50 == 0:
                total_force = np.sum(np.linalg.norm(forces, axis=1))
                if total_force < 1e-4:
                    break
                
        return refined_circles

    def local_optimization(self, circles: np.ndarray) -> np.ndarray:
        """Advanced local optimization for fine-tuning with more reliable approach"""
        optimized = circles.copy()
        
        # Simultaneous optimization - try to increase all radii
        for iteration in range(LOCAL_OPTIMIZATION_ITERATIONS):
            improved = False
            
            # Try to maximize each circle's radius
            for i in range(self.num_circles):
                current_x, current_y, current_r = optimized[i]
                
                # Create objective function that tries to maximize radius of circle i
                def radius_objective(r):
                    temp_circles = optimized.copy()
                    temp_circles[i, 2] = r[0]
                    
                    # Check validity
                    if not self.is_valid_circle(temp_circles[i, 0], temp_circles[i, 1], r[0]):
                        return 1e10
                    
                    # Check overlaps - more efficient version using spatial index
                    try:
                        violations = 0
                        # Check against existing circles efficiently
                        for j in range(self.num_circles):
                            if i != j and self.check_overlap_simple(temp_circles, i, j):
                                violations += 1
                        return -r[0] + 5000 * violations  # Strong penalty
                    except:
                        return 1e10  # Fallback
                
                # Maximize radius with bounds - more conservative approach
                bounds = [(1e-6, min(self.width/2, self.height/2, current_r * 2))]
                try:
                    # Simple optimization attempt - if failed, keep current
                    result = minimize(radius_objective, [current_r], bounds=bounds, method='L-BFGS-B', tol=1e-6)
                    if result.success:
                        new_r = max(1e-6, result.x[0])
                        # Only accept improvement if it's actually better
                        if new_r > current_r + 1e-6:
                            optimized[i, 2] = new_r
                            improved = True
                except:
                    # If optimization fails, continue - don't break the process
                    pass  # Continue with current value
            
            # Stop if no significant improvement
            if not improved:
                break
                
        return optimized

    def optimize(self) -> np.ndarray:
        """Main optimization loop with hybrid approach"""
        start_time = time.time()
        
        # Phase 1: Initialize population
        population = self.initialize_population(POPULATION_SIZE)
        
        # Phase 2: Evolutionary optimization
        stagnant_generations = 0
        max_stagnant_generations = 30
        
        for generation in range(MAX_GENERATIONS):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness, violations = self.calculate_fitness(individual)
                fitness_scores.append((fitness, violations))
            
            # Track best solution
            gen_best_idx = np.argmax([f[0] for f in fitness_scores])
            gen_best_fitness = fitness_scores[gen_best_idx][0]
            
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_solution = population[gen_best_idx].copy()
                stagnant_generations = 0
            else:
                stagnant_generations += 1
            
            # Print progress
            if generation % 25 == 0:
                print(f"Generation {generation}: Best fitness = {gen_best_fitness:.6f}")
            
            # Early stopping
            if stagnant_generations >= max_stagnant_generations:
                print(f"Converged at generation {generation}")
                break
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(self.best_solution.copy())
            
            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                adaptive_mutation_rate = INITIAL_MUTATION_RATE * (1.0 - generation / MAX_GENERATIONS)
                adaptive_mutation_rate = max(0.05, adaptive_mutation_rate)
                child = self.mutate(child, adaptive_mutation_rate)
                
                # Physics refinement
                child = self.physics_refinement(child, PHYSICS_REFINEMENT_STEPS // 5)
                
                new_population.append(child)
            
            population = new_population[:POPULATION_SIZE]
        
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best fitness achieved: {self.best_fitness:.6f}")
        
        # Phase 3: Final refinement
        if self.best_solution is not None:
            # Apply physics refinement
            self.best_solution = self.physics_refinement(self.best_solution, PHYSICS_REFINEMENT_STEPS)
            
            # Apply local optimization
            self.best_solution = self.local_optimization(self.best_solution)
            
            # Final verification
            final_fitness, _ = self.calculate_fitness(self.best_solution)
            print(f"Final refined fitness: {final_fitness:.6f}")
        
        return self.best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create optimizer instance
    optimizer = CirclePackingOptimizer(width=RECT_WIDTH, height=RECT_HEIGHT, num_circles=NUM_CIRCLES)
    
    # Run optimization
    circles = optimizer.optimize()
    
    # Ensure valid output
    if circles is None or len(circles) != NUM_CIRCLES:
        circles = np.zeros((NUM_CIRCLES, 3))
        np.random.seed(SEED)
        for i in range(NUM_CIRCLES):
            circles[i, 0] = np.random.uniform(0.01, RECT_WIDTH - 0.01)
            circles[i, 1] = np.random.uniform(0.01, RECT_HEIGHT - 0.01)
            circles[i, 2] = np.random.uniform(0.01, 0.15)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")