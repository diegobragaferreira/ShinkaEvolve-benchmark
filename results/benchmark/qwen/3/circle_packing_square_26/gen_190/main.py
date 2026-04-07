# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List, Dict, Any
import math
from collections import defaultdict
import heapq

# Algorithm configuration
POPULATION_SIZE = 150
GENERATIONS = 200
TOURNAMENT_SIZE = 5
MUTATION_RATE_START = 0.2
MUTATION_RATE_END = 0.005
CROSSOVER_PROB = 0.9
VALIDITY_THRESHOLD = 1e-6
INITIAL_GRID_SIZE = 20
ADAPTIVE_GRID_START = 50
GRID_FINE_RES = 30

class CirclePackingOptimizer:
    def __init__(self):
        random.seed(42)
        np.random.seed(42)
        
    def _compute_voronoi_regions(self, n_points: int) -> List[Tuple[float, float, float]]:
        """Compute Voronoi-like regions with adaptive spacing"""
        regions = []
        grid_size = int(math.ceil(math.sqrt(n_points)))
        
        # Create grid-aligned regions with some randomness
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(regions) >= n_points:
                    break
                base_x = (j + 1) * spacing_x
                base_y = (i + 1) * spacing_y
                # Add perturbation for better distribution
                x = max(0.01, min(0.99, base_x + np.random.normal(0, spacing_x * 0.2)))
                y = max(0.01, min(0.99, base_y + np.random.normal(0, spacing_y * 0.2)))
                regions.append((x, y, 0.0))  # Placeholder for radius
                
        return regions[:n_points]
    
    def _initialize_with_voronoi_distribution(self, n: int) -> np.ndarray:
        """Initialize circles using Voronoi-like distribution with adaptive radii"""
        circles = np.zeros((n, 3))
        
        # Generate Voronoi-like regions
        regions = self._compute_voronoi_regions(n)
        
        # Assign positions and initial radii based on region locations
        for i, (x, y, _) in enumerate(regions):
            # Set initial radius based on proximity to edges
            max_radius = min(x, 1-x, y, 1-y) * 0.8
            if max_radius < 0.01:
                max_radius = 0.01
            
            # Use exponential distribution for interesting initial radii
            radius = np.random.exponential(0.03)
            radius = min(radius, max_radius)
            
            circles[i] = [x, y, radius]
        
        return circles
    
    def _adaptive_poisson_sampling(self, n_points: int) -> List[Tuple[float, float]]:
        """Enhanced Poisson disk sampling with adaptive rejection"""
        points = []
        
        # Start with a random point
        points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
        
        # Define rejection sampling parameters
        k = 30
        radius_min = 0.1
        
        # Sample additional points
        for _ in range(n_points - 1):
            found = False
            attempts = 0
            
            while not found and attempts < k:
                # Choose a random point from existing points
                rand_idx = random.randint(0, len(points) - 1)
                base_x, base_y = points[rand_idx]
                
                # Generate candidate point in annular region
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(radius_min, 2 * radius_min)
                
                new_x = base_x + radius * math.cos(angle)
                new_y = base_y + radius * math.sin(angle)
                
                # Check bounds
                if new_x < 0.05 or new_x > 0.95 or new_y < 0.05 or new_y > 0.95:
                    attempts += 1
                    continue
                
                # Check distance to all existing points
                too_close = False
                for px, py in points:
                    dist = math.sqrt((new_x - px)**2 + (new_y - py)**2)
                    if dist < radius_min:
                        too_close = True
                        break
                
                if not too_close:
                    points.append((new_x, new_y))
                    found = True
                else:
                    attempts += 1
                    
            # If couldn't find valid point, add random point
            if not found:
                points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
        
        return points
    
    def _initialize_population_improved(self, n: int, population_size: int) -> np.ndarray:
        """Improved initialization method using multiple strategies"""
        population = []
        
        # Strategy 1: Voronoi-based initialization
        voronoi_circles = self._initialize_with_voronoi_distribution(n)
        voronoi_circles = self._resolve_initial_overlaps(voronoi_circles)
        population.append(voronoi_circles)
        
        # Strategy 2: Poisson-distributed initialization
        poisson_points = self._adaptive_poisson_sampling(n)
        poisson_circles = np.zeros((n, 3))
        for i, (x, y) in enumerate(poisson_points):
            # Distribute radii evenly at first
            max_radius = min(x, 1-x, y, 1-y) * 0.7
            if max_radius < 0.01:
                max_radius = 0.01
            radius = max_radius * (0.5 + 0.5 * random.random())
            poisson_circles[i] = [x, y, radius]
        poisson_circles = self._resolve_initial_overlaps(poisson_circles)
        population.append(poisson_circles)
        
        # Strategy 3: Hybrid of both
        hybrid_circles = np.zeros((n, 3))
        for i in range(n):
            if random.random() < 0.5:
                # Use Voronoi position
                x, y, r = voronoi_circles[i]
                # Add slight perturbation
                x += np.random.normal(0, 0.02)
                y += np.random.normal(0, 0.02)
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                # Use Voronoi radius
                hybrid_circles[i] = [x, y, r]
            else:
                # Use Poisson position
                x, y, r = poisson_circles[i]
                # Add slight perturbation
                x += np.random.normal(0, 0.02)
                y += np.random.normal(0, 0.02)
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                # Use Poisson radius
                hybrid_circles[i] = [x, y, r]
        hybrid_circles = self._resolve_initial_overlaps(hybrid_circles)
        population.append(hybrid_circles)
        
        # Fill remaining slots with variations
        while len(population) < population_size:
            # Create variations of the existing good configurations
            base_idx = random.randint(0, len(population) - 1)
            base_config = population[base_idx].copy()
            variant_config = base_config.copy()
            
            # Perturb a few circles
            for i in range(random.randint(1, n//3)):
                idx = random.randint(0, n-1)
                # Small random perturbations
                variant_config[idx, 0] = max(0.01, min(0.99, variant_config[idx, 0] + np.random.normal(0, 0.01)))
                variant_config[idx, 1] = max(0.01, min(0.99, variant_config[idx, 1] + np.random.normal(0, 0.01)))
                variant_config[idx, 2] = max(0.001, min(0.49, variant_config[idx, 2] + np.random.normal(0, 0.01)))
            
            # Resolve overlaps in variant
            variant_config = self._resolve_initial_overlaps(variant_config)
            population.append(variant_config)
        
        return np.array(population)
    
    def _resolve_initial_overlaps(self, circles: np.ndarray, max_iter: int = 20) -> np.ndarray:
        """Enhanced overlap resolution with priority queue processing"""
        result = circles.copy()
        
        # Build a priority queue of potential conflicts
        conflicts = []
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                x1, y1, r1 = result[i]
                x2, y2, r2 = result[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2 - VALIDITY_THRESHOLD):
                    # Priority based on overlap severity
                    overlap = (r1 + r2) - dist
                    heapq.heappush(conflicts, (-overlap, i, j))
        
        # Resolve conflicts in order of severity
        for _ in range(max_iter):
            if not conflicts:
                break
                
            # Resolve the most severe conflict first
            _, i, j = heapq.heappop(conflicts)
            x1, y1, r1 = result[i]
            x2, y2, r2 = result[j]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            if dist < (r1 + r2 - VALIDITY_THRESHOLD):
                # Calculate the shift needed
                dx = x2 - x1
                dy = y2 - y1
                distance = max(VALIDITY_THRESHOLD, dist)
                dx /= distance
                dy /= distance
                
                # Move circles apart proportionally to inverse radii
                move_amount = (r1 + r2 - dist) * 0.5
                
                # Apply movement in opposite directions
                scale_factor = min(1.0, r1 / (r1 + r2 + 0.001))
                result[i, 0] -= dx * move_amount * scale_factor * 0.5
                result[i, 1] -= dy * move_amount * scale_factor * 0.5
                result[j, 0] += dx * move_amount * (1 - scale_factor) * 0.5
                result[j, 1] += dy * move_amount * (1 - scale_factor) * 0.5
                
                # Enforce bounds
                x1, y1, r1 = result[i]
                x2, y2, r2 = result[j]
                result[i] = [max(r1, min(1-r1, x1)), max(r1, min(1-r1, y1)), r1]
                result[j] = [max(r2, min(1-r2, x2)), max(r2, min(1-r2, y2)), r2]
                
                # Add updated conflicts back to queue
                for k in range(len(result)):
                    if k != i and k != j:
                        xk, yk, rk = result[k]
                        dist_i = math.sqrt((x1 - xk)**2 + (y1 - yk)**2)
                        dist_j = math.sqrt((x2 - xk)**2 + (y2 - yk)**2)
                        if dist_i < (r1 + rk - VALIDITY_THRESHOLD):
                            overlap = (r1 + rk) - dist_i
                            heapq.heappush(conflicts, (-overlap, i, k))
                        if dist_j < (r2 + rk - VALIDITY_THRESHOLD):
                            overlap = (r2 + rk) - dist_j
                            heapq.heappush(conflicts, (-overlap, j, k))
        
        return result

    def _check_containment(self, circles: np.ndarray) -> bool:
        """Check if all circles are fully contained in the unit square."""
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True

    def _calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _get_grid_cells(self, circles: np.ndarray, grid_size: int = INITIAL_GRID_SIZE) -> dict:
        """Create a spatial grid for fast neighbor lookups."""
        grid = {}
        cell_size = 1.0 / grid_size
        
        for i, (x, y, r) in enumerate(circles):
            # Determine which grid cells this circle might occupy
            min_x_cell = max(0, int((x - r) / cell_size))
            max_x_cell = min(grid_size - 1, int((x + r) / cell_size))
            min_y_cell = max(0, int((y - r) / cell_size))
            max_y_cell = min(grid_size - 1, int((y + r) / cell_size))
            
            for gx in range(min_x_cell, max_x_cell + 1):
                for gy in range(min_y_cell, max_y_cell + 1):
                    if (gx, gy) not in grid:
                        grid[(gx, gy)] = []
                    grid[(gx, gy)].append(i)
        
        return grid

    def _check_overlap_efficient(self, circles: np.ndarray, grid: dict = None) -> bool:
        """Check if any circles overlap using spatial grid indexing."""
        if len(circles) <= 1:
            return False
        
        if grid is None:
            grid = self._get_grid_cells(circles, INITIAL_GRID_SIZE)
        
        # For each cell, check pairs of circles
        for (gx, gy), indices in grid.items():
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    x1, y1, r1 = circles[idx1]
                    x2, y2, r2 = circles[idx2]
                    
                    distance = self._calculate_distance((x1, y1), (x2, y2))
                    if distance < (r1 + r2 - VALIDITY_THRESHOLD):
                        return True
        
        return False

    def _compute_penalty(self, circles: np.ndarray, generation: int = 0, 
                        total_generations: int = 100, diversity_factor: float = 1.0) -> float:
        """Compute penalty based on constraint violations with adaptive scaling."""
        penalty = 0.0
        
        # Dynamic penalty scaling factor with diversity consideration
        penalty_scale = 1.0 + (generation / total_generations) * 5.0 * diversity_factor
        
        # Check containment violations with scaled penalties
        for x, y, r in circles:
            # Boundary violations  
            if x - r < 0:
                penalty += (abs(x - r) ** 2) * 10000 * penalty_scale
            elif x + r > 1:
                penalty += (abs(x + r - 1) ** 2) * 10000 * penalty_scale
            if y - r < 0:
                penalty += (abs(y - r) ** 2) * 10000 * penalty_scale
            elif y + r > 1:
                penalty += (abs(y + r - 1) ** 2) * 10000 * penalty_scale
        
        # Check overlap violations with scaled penalties
        grid = self._get_grid_cells(circles, INITIAL_GRID_SIZE)
        if self._check_overlap_efficient(circles, grid):
            penalty += 10000000.0 * penalty_scale
        
        return penalty

    def _evaluate_fitness(self, circles: np.ndarray, generation: int = 0, 
                         total_generations: int = 100, use_penalty: bool = True) -> float:
        """Evaluate fitness of a circle configuration."""
        # If invalid, heavily penalize
        if not self._check_containment(circles) or self._check_overlap_efficient(circles):
            if use_penalty:
                # Calculate population diversity factor
                diversity_factor = self._calculate_diversity_factor(circles)
                penalty = self._compute_penalty(circles, generation, total_generations, diversity_factor)
                return -penalty
            else:
                return -float('inf')
        
        # Otherwise, return total radius
        total_radius = np.sum(circles[:, 2])
        return total_radius

    def _calculate_diversity_factor(self, circles: np.ndarray) -> float:
        """Calculate a diversity factor to adjust penalty scaling."""
        if len(circles) <= 1:
            return 1.0
            
        # Simple variance-based diversity metric
        radii = circles[:, 2]
        mean_radius = np.mean(radii)
        variance = np.var(radii)
        
        # Normalize variance to [0.5, 1.5] range
        diversity = 1.0 + 0.5 * (variance / (mean_radius**2 + 1e-8))
        return max(0.5, min(1.5, diversity))

    def _mutate(self, circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
        """Mutate a circle configuration with adaptive rates and selective approach."""
        mutated = circles.copy()
        
        # Adaptive mutation rate using sigmoid decay
        mutation_rate = MUTATION_RATE_START + (MUTATION_RATE_END - MUTATION_RATE_START) * \
                       (1 / (1 + math.exp(-10 * (generation / total_generations - 0.5))))
        
        n = len(mutated)
        
        # Mutation type selection based on generation
        mutation_types = ['position', 'radius']
        if generation > total_generations * 0.7:
            mutation_types = ['position']  # Later stages focus on refinement
        
        # Mutate some circles with selective approach
        for i in range(n):
            if random.random() < mutation_rate:
                # Choose mutation type
                mut_type = random.choice(mutation_types)
                
                if mut_type == 'position':
                    # Mutate position with larger steps in early generations
                    step_size = 0.05 if generation < total_generations * 0.3 else 0.02
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, step_size)))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, step_size)))
                else:  # radius
                    # Mutate radius with smaller steps
                    mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + random.gauss(0, 0.02)))
        
        # Ensure valid configuration after mutation
        return self._enforce_constraints(mutated)

    def _enforce_constraints(self, circles: np.ndarray) -> np.ndarray:
        """Enforce constraints on circle positions and radii."""
        result = circles.copy()
        
        # Adjust positions and radii to satisfy bounds
        for i in range(len(result)):
            x, y, r = result[i]
            
            # Ensure circle fits in the unit square
            r = min(r, x, 1-x, y, 1-y)
            r = max(0.001, min(0.49, r))
            
            # Clamp coordinates to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            result[i] = [x, y, r]
        
        return result

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two parent configurations with enhanced recombination."""
        if random.random() > CROSSOVER_PROB:
            # Return one of the parents randomly
            return parent1.copy() if random.random() < 0.5 else parent2.copy()
        
        n = len(parent1)
        child = np.zeros_like(parent1)
        
        # Use uniform crossover for better recombination
        for i in range(n):
            # Uniform crossover for each parameter
            if random.random() < 0.5:
                child[i] = parent1[i].copy()
            else:
                child[i] = parent2[i].copy()
            
            # Add some blending for better exploration (30% chance)
            if random.random() < 0.3:
                alpha = random.random()
                # Blend positions and radii
                child[i][0] = parent1[i][0] + alpha * (parent2[i][0] - parent1[i][0])
                child[i][1] = parent1[i][1] + alpha * (parent2[i][1] - parent1[i][1])
                child[i][2] = parent1[i][2] + alpha * (parent2[i][2] - parent1[i][2])
        
        # Ensure offspring stays within bounds
        child = self._enforce_constraints(child)
        
        # Apply aggressive refinement to ensure validity
        child = self._refine_configuration(child)
        
        return child

    def _refine_configuration(self, circles: np.ndarray) -> np.ndarray:
        """Aggressive refinement to remove overlaps and correct constraints."""
        refined = circles.copy()
        
        # Try to resolve overlaps up to 10 times
        for iteration in range(10):
            grid = self._get_grid_cells(refined, INITIAL_GRID_SIZE)
            
            # Check for overlaps and resolve them
            resolved = False
            for i in range(len(refined)):
                for j in range(i+1, len(refined)):
                    xi, yi, ri = refined[i]
                    xj, yj, rj = refined[j]
                    dist = self._calculate_distance((xi, yi), (xj, yj))
                    
                    if dist < (ri + rj - VALIDITY_THRESHOLD):
                        # Resolve overlap by moving circles apart with force-based approach
                        dx = xj - xi
                        dy = yj - yi
                        distance = max(VALIDITY_THRESHOLD, dist)
                        
                        # Normalize direction vector
                        dx /= distance
                        dy /= distance
                        
                        # Move circles apart based on their relative sizes and distances
                        move_amount = (ri + rj - dist) * 0.7  # Increased factor
                        
                        # Scale by inverse radii to balance movement
                        scale_factor = min(1.0, ri / (ri + rj + 0.001))
                        refined[i, 0] -= dx * move_amount * scale_factor * 0.5
                        refined[i, 1] -= dy * move_amount * scale_factor * 0.5
                        refined[j, 0] += dx * move_amount * (1 - scale_factor) * 0.5
                        refined[j, 1] += dy * move_amount * (1 - scale_factor) * 0.5
                        resolved = True
            
            # Enforce bounds
            for i in range(len(refined)):
                x, y, r = refined[i]
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                refined[i] = [x, y, r]
            
            # Early stopping if no changes made
            if not resolved:
                break
        
        return refined

    def _tournament_selection(self, population: np.ndarray, fitnesses: np.ndarray, 
                             tournament_size: int) -> np.ndarray:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]

    def run_evolution(self) -> np.ndarray:
        """Run the evolutionary algorithm to find optimal circle packing."""
        n = 26
        population = self._initialize_population_improved(n, POPULATION_SIZE)
        
        # Evaluate initial population
        fitnesses = [self._evaluate_fitness(individual) for individual in population]
        
        # Evolution loop
        for gen in range(GENERATIONS):
            # Selection, crossover, and mutation
            new_population = []
            
            for _ in range(POPULATION_SIZE):
                # Tournament selection
                parent1 = self._tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
                parent2 = self._tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                child = self._mutate(child, gen, GENERATIONS)
                
                new_population.append(child)
            
            # Evaluate new population
            population = np.array(new_population)
            fitnesses = [self._evaluate_fitness(individual, gen, GENERATIONS) for individual in population]
            
            # Print progress
            best_fitness = max(fitnesses)
            if gen % 25 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness}")
        
        # Return the best individual
        best_index = np.argmax(fitnesses)
        best_solution = population[best_index]
        
        # Final refinement using more elaborate approach
        best_solution = self._refine_configuration(best_solution)
        
        return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END