# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Optional
import time
from collections import defaultdict

# Global constants for optimization
POPULATION_SIZE = 120
GENERATIONS = 500
INITIAL_MUTATION_RATE = 0.25
FINAL_MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 7
BOUNDARY_PENALTY_BASE = 1000.0
OVERLAP_PENALTY_BASE = 10000.0
ELITISM_COUNT = 5
BOUNDARY_MARGIN = 0.01
SPATIAL_INDEXING_THRESHOLD = 50

class CircleEvaluator:
    """Handles all circle validation and fitness computation logic"""
    
    @staticmethod
    def is_valid_position(x: float, y: float, r: float) -> bool:
        """Check if a circle position is valid (within bounds)"""
        return (r <= x <= 1 - r and r <= y <= 1 - r)
    
    @staticmethod
    def calculate_penalty(circles: np.ndarray) -> Tuple[float, float, float]:
        """Calculate penalty based on constraint violations"""
        penalty = 0.0
        boundary_violations = 0.0
        overlap_violations = 0.0
        
        # Check containment penalties
        for circle in circles:
            x, y, r = circle
            if not CircleEvaluator.is_valid_position(x, y, r):
                # Calculate violation amounts
                left_violation = max(0, r - x)
                right_violation = max(0, r - (1 - x))
                bottom_violation = max(0, r - y)
                top_violation = max(0, r - (1 - y))
                boundary_violations += (left_violation + right_violation + 
                                      bottom_violation + top_violation)
        
        # Check overlaps using efficient spatial indexing
        valid_circles = [c for c in circles if c[2] > 0]
        if len(valid_circles) > 1:
            positions = np.array([[c[0], c[1]] for c in valid_circles])
            
            if len(valid_circles) > SPATIAL_INDEXING_THRESHOLD:
                # Use KDTree for large populations
                tree = cKDTree(positions)
                pairs = tree.query_pairs(2 * np.max([c[2] for c in valid_circles]))
                
                for i, j in pairs:
                    if i != j:
                        c1 = valid_circles[i]
                        c2 = valid_circles[j]
                        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                        if distance < (c1[2] + c2[2]):
                            overlap = (c1[2] + c2[2]) - distance
                            overlap_violations += overlap
            else:
                # Use direct distance calculation for small populations
                distances = cdist(positions, positions)
                for i in range(len(valid_circles)):
                    for j in range(i + 1, len(valid_circles)):
                        if distances[i, j] < (valid_circles[i][2] + valid_circles[j][2]):
                            overlap = (valid_circles[i][2] + valid_circles[j][2]) - distances[i, j]
                            overlap_violations += overlap
            
            penalty = BOUNDARY_PENALTY_BASE * boundary_violations + \
                     OVERLAP_PENALTY_BASE * overlap_violations
        
        return penalty, boundary_violations, overlap_violations

class CircleInitializer:
    """Handles circle initialization and population creation"""
    
    @staticmethod
    def generate_structured_positions(n_circles: int) -> List[Tuple[float, float]]:
        """Generate structured positions using grid-based approach"""
        # Create a more structured initial distribution
        # Use a grid-based approach with systematic placement
        grid_size = max(4, int(np.ceil(np.sqrt(n_circles))))
        positions = []
        
        # Create a structured grid
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) < n_circles:
                    x = 0.1 + (i / (grid_size - 1)) * 0.8
                    y = 0.1 + (j / (grid_size - 1)) * 0.8
                    positions.append((x, y))
        
        # Fill remaining positions with random but constrained placement
        while len(positions) < n_circles:
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            positions.append((x, y))
            
        return positions
    
    @staticmethod
    def generate_voronoi_distribution(n_points: int, n_circles: int) -> List[Tuple[float, float]]:
        """Generate points using Voronoi diagram approach for better distribution"""
        points = []
        
        # Create systematic grid with perturbations
        grid_size = max(6, int(np.ceil(np.sqrt(n_points))))
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < n_points:
                    x = 0.05 + (i / (grid_size - 1)) * 0.90
                    y = 0.05 + (j / (grid_size - 1)) * 0.90
                    # Add small random perturbation
                    x += random.uniform(-0.03, 0.03)
                    y += random.uniform(-0.03, 0.03)
                    # Clip to valid range
                    x = max(0.05, min(0.95, x))
                    y = max(0.05, min(0.95, y))
                    points.append((x, y))
        
        # Add random points to fill out the space
        while len(points) < n_points:
            points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
        
        return points[:n_circles]
    
    @staticmethod
    def poisson_disk_sampling(width: float, height: float, min_distance: float, max_attempts: int = 30) -> np.ndarray:
        """Generate points using Poisson disk sampling for uniform distribution"""
        # Grid to track occupied cells
        cell_size = min_distance / np.sqrt(2)
        grid_width = int(np.ceil(width / cell_size))
        grid_height = int(np.ceil(height / cell_size))
        grid = np.full((grid_height, grid_width), -1, dtype=int)
        
        # List of points and active list
        points = []
        active_list = []
        
        # Add first point randomly
        first_point = np.random.rand(2) * [width, height]
        first_point[0] = np.clip(first_point[0], BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
        first_point[1] = np.clip(first_point[1], BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
        
        points.append(first_point)
        active_list.append(0)
        
        # Index for grid
        def get_grid_index(point):
            x, y = point
            grid_x = int(x / cell_size)
            grid_y = int(y / cell_size)
            return grid_y, grid_x
        
        grid[get_grid_index(first_point)] = 0
        
        attempts = 0
        while active_list and attempts < max_attempts:
            # Pick random point from active list
            idx = np.random.randint(len(active_list))
            point_idx = active_list[idx]
            point = points[point_idx]
            
            # Try to generate new point
            found = False
            for _ in range(max_attempts):
                angle = np.random.rand() * 2 * np.pi
                radius = np.random.uniform(min_distance, 2 * min_distance)
                
                new_point = point + np.array([radius * np.cos(angle), radius * np.sin(angle)])
                
                # Check boundaries
                if (new_point[0] < BOUNDARY_MARGIN or new_point[0] > 1 - BOUNDARY_MARGIN or
                    new_point[1] < BOUNDARY_MARGIN or new_point[1] > 1 - BOUNDARY_MARGIN):
                    continue
                
                # Check grid for nearby points
                grid_y, grid_x = get_grid_index(new_point)
                valid = True
                
                # Check nearby cells
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = grid_y + dy, grid_x + dx
                        if 0 <= ny < grid_height and 0 <= nx < grid_width:
                            neighbor_idx = grid[ny, nx]
                            if neighbor_idx != -1:
                                neighbor = points[neighbor_idx]
                                dist = np.linalg.norm(new_point - neighbor)
                                if dist < min_distance:
                                    valid = False
                                    break
                    if not valid:
                        break
                
                if valid:
                    points.append(new_point)
                    active_list.append(len(points) - 1)
                    grid[grid_y, grid_x] = len(points) - 1
                    found = True
                    break
            
            if not found:
                active_list.pop(idx)
            
            attempts += 1
        
        return np.array(points)
    
    @staticmethod
    def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
        """Initialize population with hybrid approach combining Voronoi, Poisson, and structured methods"""
        # Set seeds for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Try Poisson disk sampling first, fallback to Voronoi then structured
        try:
            poisson_points = CircleInitializer.poisson_disk_sampling(1.0, 1.0, 0.15)
            voronoi_points = CircleInitializer.generate_voronoi_distribution(max(36, n_circles * 2), n_circles)
        except Exception:
            # Fallback to Voronoi generation if Poisson fails
            poisson_points = []
            voronoi_points = CircleInitializer.generate_voronoi_distribution(max(36, n_circles * 2), n_circles)
        
        positions = CircleInitializer.generate_structured_positions(n_circles)
        population = []
        
        for _ in range(pop_size):
            individual = np.zeros((n_circles, 3))
            
            # Assign positions with mixed strategies
            for i in range(n_circles):
                # Use Poisson points when available, otherwise Voronoi, otherwise structured
                if i < len(poisson_points):
                    x, y = poisson_points[i]
                elif i < len(voronoi_points):
                    x, y = voronoi_points[i]
                else:
                    x, y = positions[i]
                
                # Add structured perturbation to maintain good distribution
                if i < len(positions):
                    # Add small structured perturbation based on position
                    perturbation_x = random.uniform(-0.02, 0.02) + 0.01 * (i % 3 - 1)
                    perturbation_y = random.uniform(-0.02, 0.02) + 0.01 * (i % 2 - 1)
                    x += perturbation_x
                    y += perturbation_y
                else:
                    # Random placement for extra diversity
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                
                # Clip to valid range
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                individual[i, 0] = x
                individual[i, 1] = y
                
                # Assign radius based on proximity to edges and systematic approach
                margin = min(x, y, 1 - x, 1 - y)
                base_radius = min(0.15, margin / 2.0)
                
                # Use more structured radius assignment to avoid very small circles
                # that could cause convergence issues
                if base_radius > 0.01:
                    # Use a combination of base radius and systematic variation
                    radius_variation = random.uniform(0.6, 1.2)
                    individual[i, 2] = max(0.01, base_radius * radius_variation)
                else:
                    individual[i, 2] = random.uniform(0.01, 0.1)
            
            # Refine solution to ensure validity
            individual = CircleRefiner.refine_solution(individual)
            population.append(individual)
        
        return population

class CircleRefiner:
    """Handles local refinement and constraint enforcement"""
    
    @staticmethod
    def refine_solution(circles: np.ndarray) -> np.ndarray:
        """Apply local refinement to fix constraint violations"""
        refined = circles.copy()
        
        # Phase 1: Fix containment violations
        for i in range(len(refined)):
            x, y, r = refined[i]
            if not CircleEvaluator.is_valid_position(x, y, r):
                if r > x:
                    x = r + 0.001
                if r > y:
                    y = r + 0.001
                if r > (1 - x):
                    x = 1 - r - 0.001
                if r > (1 - y):
                    y = 1 - r - 0.001
                refined[i, 0] = x
                refined[i, 1] = y
        
        # Phase 2: Iterative overlap resolution
        max_iter = 100
        for iteration in range(max_iter):
            changed = False
            valid_indices = [i for i in range(len(refined)) if refined[i, 2] > 0]
            
            for i in valid_indices:
                x, y, r = refined[i]
                
                for j in valid_indices:
                    if i != j:
                        ox, oy, oradius = refined[j]
                        distance = np.sqrt((x - ox)**2 + (y - oy)**2)
                        
                        if distance < (r + oradius):
                            if distance > 0.0001:
                                dx = (x - ox) / distance
                                dy = (y - oy) / distance
                                
                                # Reduce radius to prevent further overlap
                                new_r = max(0.001, (r + oradius) * 0.99 - distance)
                                if new_r < r and new_r > 0.001:
                                    refined[i, 2] = new_r
                                    changed = True
                                
                                # Adjust position to separate circles
                                separation = 0.001
                                refined[i, 0] = x + dx * separation
                                refined[i, 1] = y + dy * separation
                                
                                # Ensure containment after adjustment
                                refined[i, 0] = np.clip(refined[i, 0], refined[i, 2], 1 - refined[i, 2])
                                refined[i, 1] = np.clip(refined[i, 1], refined[i, 2], 1 - refined[i, 2])
                            else:
                                # If circles are at same position, move one slightly
                                refined[i, 0] += random.uniform(-0.001, 0.001)
                                refined[i, 1] += random.uniform(-0.001, 0.001)
                                changed = True
            
            if not changed:
                break
        
        # Final containment check and correction
        for i in range(len(refined)):
            x, y, r = refined[i]
            r = max(0.001, r)
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            refined[i, 0] = x
            refined[i, 1] = y
            refined[i, 2] = r
        
        return refined

class EvolutionaryOperator:
    """Handles evolutionary operators (selection, crossover, mutation)"""
    
    @staticmethod
    def adaptive_mutation_rate(generation: int, max_generations: int) -> float:
        """Adaptive mutation rate that decreases over time"""
        return INITIAL_MUTATION_RATE - (INITIAL_MUTATION_RATE - FINAL_MUTATION_RATE) * (generation / max_generations)
    
    @staticmethod
    def tournament_selection(population: List[np.ndarray], fitness_scores: List[float],
                            tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
        """Select an individual using tournament selection"""
        selected_indices = random.sample(range(len(population)), tournament_size)
        selected_fitness = [fitness_scores[i] for i in selected_indices]
        winner_idx = selected_indices[np.argmax(selected_fitness)]
        return population[winner_idx].copy()
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray,
                 crossover_rate: float = CROSSOVER_RATE) -> np.ndarray:
        """Perform crossover between two parents"""
        if random.random() > crossover_rate:
            return parent1.copy()
        
        n = len(parent1)
        child = np.zeros_like(parent1)
        
        # Multi-point crossover with better balance
        crossover_points = sorted(random.sample(range(1, n), min(3, n-1)))
        last_point = 0
        use_parent1 = True
        
        for point in crossover_points:
            if use_parent1:
                child[last_point:point, :] = parent1[last_point:point, :]
            else:
                child[last_point:point, :] = parent2[last_point:point, :]
            last_point = point
            use_parent1 = not use_parent1
        
        # Handle final segment
        if use_parent1:
            child[last_point:, :] = parent1[last_point:, :]
        else:
            child[last_point:, :] = parent2[last_point:, :]
        
        # Local refinement
        child = CircleRefiner.refine_solution(child)
        return child
    
    @staticmethod
    def mutate(individual: np.ndarray, mutation_rate: float = 0.2) -> np.ndarray:
        """Mutate an individual with different strategies"""
        mutated = individual.copy()
        n = len(mutated)
        
        # Apply mutations based on probabilities
        for i in range(n):
            if random.random() < mutation_rate:
                # Choose mutation type with preference for position changes
                mutation_type = random.choices(
                    [0, 1, 2, 3], 
                    weights=[0.5, 0.5, 0.2, 0.3]
                )[0]
                
                if mutation_type == 0:  # Mutate x position (larger change)
                    mutated[i, 0] = np.clip(mutated[i, 0] + random.gauss(0, 0.05), 0.05, 0.95)
                elif mutation_type == 1:  # Mutate y position (larger change)
                    mutated[i, 1] = np.clip(mutated[i, 1] + random.gauss(0, 0.05), 0.05, 0.95)
                elif mutation_type == 2:  # Mutate radius (smaller change)
                    mutated[i, 2] = np.clip(mutated[i, 2] + random.gauss(0, 0.01), 0.001, 0.2)
                else:  # Mutate both position and radius
                    mutated[i, 0] = np.clip(mutated[i, 0] + random.gauss(0, 0.02), 0.05, 0.95)
                    mutated[i, 1] = np.clip(mutated[i, 1] + random.gauss(0, 0.02), 0.05, 0.95)
                    mutated[i, 2] = np.clip(mutated[i, 2] + random.gauss(0, 0.005), 0.001, 0.2)
        
        # Local refinement after mutation
        mutated = CircleRefiner.refine_solution(mutated)
        return mutated

class CirclePackingOptimizer:
    """Main optimizer orchestrating the evolutionary process"""
    
    def __init__(self):
        self.evaluator = CircleEvaluator()
        self.initializer = CircleInitializer()
        self.refiner = CircleRefiner()
        self.operator = EvolutionaryOperator()
    
    def evaluate_fitness(self, circles: np.ndarray, generation: int = 0, total_generations: int = GENERATIONS) -> Tuple[float, float, float]:
        """Evaluate the fitness of a solution with adaptive penalty scaling"""
        total_radius = np.sum(circles[:, 2])
        penalty, _, _ = self.evaluator.calculate_penalty(circles)
        
        # Progressive penalty scaling that increases with generation
        penalty_scaling = 1.0 + (generation / total_generations) * 5.0
        penalty *= penalty_scaling
        
        fitness = total_radius - penalty
        return fitness, total_radius, penalty
    
    def evolve_population(self, population: List[np.ndarray], generation: int) -> Tuple[List[np.ndarray], float, float, float]:
        """Evolve the population for one generation"""
        # Evaluate fitness
        fitness_scores = []
        total_radii = []
        penalties = []
        
        for individual in population:
            fitness, total_radius, penalty = self.evaluate_fitness(individual, generation, GENERATIONS)
            fitness_scores.append(fitness)
            total_radii.append(total_radius)
            penalties.append(penalty)
        
        # Track best individual
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_total_radius = total_radii[best_idx]
        best_penalty = penalties[best_idx]
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individuals
        elite_indices = np.argsort(fitness_scores)[-ELITISM_COUNT:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate rest of population
        while len(new_population) < len(population):
            # Selection
            parent1 = self.operator.tournament_selection(population, fitness_scores)
            parent2 = self.operator.tournament_selection(population, fitness_scores)
            
            # Crossover
            child = self.operator.crossover(parent1, parent2)
            
            # Mutation with adaptive rate
            mut_rate = self.operator.adaptive_mutation_rate(len(new_population), POPULATION_SIZE)
            child = self.operator.mutate(child, mut_rate)
            
            new_population.append(child)
        
        return new_population, best_fitness, best_total_radius, best_penalty

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    optimizer = CirclePackingOptimizer()
    n = 26
    population = optimizer.initializer.initialize_population(POPULATION_SIZE, n)
    
    best_total_radius = 0.0
    best_individual = None
    best_penalty = float('inf')
    
    # Evolution loop
    start_time = time.time()
    for generation in range(GENERATIONS):
        population, gen_fitness, gen_radius, gen_penalty = optimizer.evolve_population(population, generation)
        
        if gen_radius > best_total_radius:
            best_total_radius = gen_radius
            best_individual = population[0]  # Keep track of best individual
            best_penalty = gen_penalty
        
        # Print progress every 100 generations
        if generation % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Generation {generation}: Best radius sum = {gen_radius:.6f} (penalty={gen_penalty:.2f}) Time: {elapsed:.2f}s")
    
    elapsed = time.time() - start_time
    print(f"Final result: Best radius sum = {best_total_radius:.6f} (penalty={best_penalty:.2f}) Time: {elapsed:.2f}s")
    print(f"Benchmark ratio: {best_total_radius / 2.6358627564136983:.6f}")
    
    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback to returning first individual if something went wrong
        return population[0]

# EVOLVE-BLOCK-END