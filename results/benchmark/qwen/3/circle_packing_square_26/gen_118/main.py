# EVOLVE-BLOCK-START
import numpy as np
import random
from copy import deepcopy
from typing import Tuple, List
from scipy.spatial import Voronoi, cKDTree

class SpatialGrid:
    """Efficient spatial grid for fast overlap detection"""
    
    def __init__(self, resolution: int = 30):
        self.resolution = resolution
        self.grid = {}
        self.cell_size = 1.0 / resolution
        
    def _get_cell_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Get grid cell coordinates for a point"""
        return (int(x / self.cell_size), int(y / self.cell_size))
    
    def clear(self):
        """Clear the spatial grid"""
        self.grid.clear()
        
    def add_circle(self, idx: int, x: float, y: float, radius: float):
        """Add a circle to the spatial grid"""
        cell_coords = self._get_cell_coords(x, y)
        if cell_coords not in self.grid:
            self.grid[cell_coords] = []
        self.grid[cell_coords].append((idx, x, y, radius))
        
    def get_neighbors(self, x: float, y: float, radius: float) -> List[Tuple[int, float, float, float]]:
        """Get all circles in neighboring cells that could potentially overlap"""
        neighbors = []
        cell_x, cell_y = self._get_cell_coords(x, y)
        
        # Check surrounding cells (3x3 grid around main cell)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in self.grid:
                    for idx, nx, ny, nr in self.grid[neighbor_cell]:
                        # Skip self
                        if idx == -1:  # Placeholder for self-check
                            continue
                        neighbors.append((idx, nx, ny, nr))
        return neighbors

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n_circles = 26
    max_generations = 1000
    population_size = 150
    
    # Global spatial grid instance for validation
    spatial_grid = SpatialGrid(resolution=30)
    
    def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
        """Initialize population with diverse configurations using enhanced Voronoi and Fibonacci methods"""
        population = []
        
        # Pre-compute Golden Ratio for Fibonacci spiral
        phi = (1 + np.sqrt(5)) / 2
        
        for _ in range(pop_size):
            # Create Fibonacci spiral distribution for better point spread
            circles = np.zeros((n_circles, 3))
            points = []
            
            # Generate points using Fibonacci spiral method
            for i in range(n_circles):
                angle = i * 2.399963229728653  # 2*pi*(i*phi) mod 2*pi
                radius = np.sqrt(i) / np.sqrt(n_circles-1) if n_circles > 1 else 0.5
                x = 0.5 + radius * np.cos(angle) * 0.45
                y = 0.5 + radius * np.sin(angle) * 0.45
                
                # Add noise to avoid perfect patterns
                x += np.random.normal(0, 0.015)
                y += np.random.normal(0, 0.015)
                
                # Clamp to unit square with safety margin
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                
                points.append([x, y])
            
            # Enhance distribution using Voronoi if possible
            try:
                # Use the first half of points to create Voronoi diagram
                if len(points) >= n_circles:
                    # Create Voronoi diagram
                    vor_points = points[:min(len(points), n_circles * 2)]
                    vor = Voronoi(vor_points)
                    valid_voronoi_points = []
                    
                    # Extract valid Voronoi vertices 
                    for vertex in vor.vertices:
                        if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                            valid_voronoi_points.append(vertex)
                    
                    # If we have enough valid Voronoi points, use them
                    if len(valid_voronoi_points) >= n_circles:
                        points = valid_voronoi_points[:n_circles]
            except:
                pass  # If Voronoi fails, continue with spiral points
            
            # Create circles with calculated radii
            for i in range(n_circles):
                x, y = points[i]
                
                # Calculate safe radius based on proximity to other circles
                min_dist = float('inf')
                for other_x, other_y in points[:i]:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)
                
                # Set radius with boundary constraints and neighbor distances
                boundary_dist = min(x, 1-x, y, 1-y)
                # Different approach: prefer larger radii for earlier circles
                if i < 8:
                    radius = min(0.08, boundary_dist, min_dist/2)
                else:
                    radius = min(0.05, boundary_dist, min_dist/2)
                    
                if radius <= 0:
                    radius = 0.01
                    
                circles[i] = [x, y, radius]
            
            population.append(circles)
        return population
    
    def is_valid(circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and non-overlapping"""
        n = len(circles)
        
        # Clear spatial grid
        spatial_grid.clear()
        
        # First check boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
            # Add to spatial grid for overlap checking
            spatial_grid.add_circle(i, x, y, r)
        
        # Then check overlap constraints using spatial grid
        for i in range(n):
            x1, y1, r1 = circles[i]
            
            # Get potential overlapping candidates from neighbors
            candidates = spatial_grid.get_neighbors(x1, y1, r1)
            
            # Check actual overlaps with early termination
            for _, x2, y2, r2 in candidates:
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False
                    
        return True
    
    def evaluate_fitness(circles: np.ndarray) -> float:
        """Evaluate fitness of a solution"""
        if not is_valid(circles):
            # Dynamic penalty based on constraint violations with proper weighting
            total_penalty = 0
            
            # Boundary violations with high penalty
            for i in range(len(circles)):
                x, y, r = circles[i]
                boundary_violation = 0
                if x - r < 0:
                    boundary_violation += abs(x - r)
                if x + r > 1:
                    boundary_violation += abs(x + r - 1)
                if y - r < 0:
                    boundary_violation += abs(y - r)
                if y + r > 1:
                    boundary_violation += abs(y + r - 1)
                total_penalty += boundary_violation * 1000000
            
            # Overlap violations with extreme penalty
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    overlap = max(0, r1 + r2 - distance)
                    total_penalty += overlap * 10000000
            
            return -total_penalty
        
        # Valid configuration: maximize sum of radii
        return np.sum(circles[:, 2])
    
    def mutate(circles: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Apply mutation to circles with enhanced strategy"""
        mutated = deepcopy(circles)
        # Adaptive mutation rate with faster decay for later generations
        mutation_rate = 0.25 - (0.2 * generation / max_generations)  
        
        # Mutate each circle with some probability
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Mutate position with larger step size early, smaller later
                pos_mutation_scale = 0.04 * (1 - generation/max_generations) + 0.01
                mutated[i, 0] += np.random.normal(0, pos_mutation_scale)  # x coordinate
                mutated[i, 1] += np.random.normal(0, pos_mutation_scale)  # y coordinate
                
                # Clamp to unit square with safety margin
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))
                
                # Mutate radius with larger variance early, smaller later
                rad_mutation_scale = 0.03 * (1 - generation/max_generations) + 0.005
                mutated[i, 2] += np.random.normal(0, rad_mutation_scale)
                mutated[i, 2] = max(0.001, mutated[i, 2])
                
        return mutated
    
    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring via crossover of two parents"""
        child = deepcopy(parent1)
        n = len(parent1)
        
        # Uniform crossover for better mixing
        mask = np.random.rand(n) > 0.5
        
        for i in range(n):
            if mask[i]:
                child[i] = parent2[i].copy()
            
        return child
    
    def tournament_selection(population: List[np.ndarray], k: int = 5) -> np.ndarray:
        """Select individual using tournament selection"""
        selected = random.sample(population, k)
        return max(selected, key=evaluate_fitness)
    
    def geometric_refinement(circles: np.ndarray) -> np.ndarray:
        """Apply geometric refinement to improve solution quality"""
        refined = deepcopy(circles)
        
        # Phase 1: Fix containment issues efficiently
        for i in range(len(refined)):
            x, y, r = refined[i]
            # Ensure containment with margin
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            refined[i] = [x, y, r]
        
        # Phase 2: Try to slightly increase radii where possible with early termination
        improved_count = 0
        for iteration in range(15):  # Reduced iterations for efficiency
            improved = False
            for i in range(len(refined)):
                x, y, r = refined[i]
                # Try to increase radius slightly while maintaining constraints
                new_r = min(r + 0.0015, x, 1-x, y, 1-y)
                
                # Quick check against boundary constraints
                if new_r <= r:
                    continue
                    
                # Test if we can increase this radius efficiently
                valid = True
                temp_r = new_r
                for j in range(len(refined)):
                    if i != j:
                        x2, y2, r2 = refined[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if distance < temp_r + r2:
                            valid = False
                            break
                
                if valid and new_r > r:
                    refined[i] = [x, y, new_r]
                    improved = True
                    improved_count += 1
            
            # Stop early if no improvement made
            if not improved:
                break
                
        return refined
    
    def refine_solution(circles: np.ndarray) -> np.ndarray:
        """Apply local refinement to fix minor constraint violations"""
        refined = deepcopy(circles)
        
        # Fix containment violations first
        for i in range(len(refined)):
            x, y, r = refined[i]
            # Ensure containment with margin
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            refined[i] = [x, y, r]
            
        # Apply geometric refinement for further improvement
        refined = geometric_refinement(refined)
            
        return refined
    
    # Initialize population
    population = initialize_population(population_size, n_circles)
    
    # Evolve
    best_fitness = float('-inf')
    best_solution = None
    
    for generation in range(max_generations):
        # Evaluate fitness for entire population
        fitness_scores = [evaluate_fitness(individual) for individual in population]
        
        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = deepcopy(population[max_fitness_idx])
        
        # Elitism: keep top 25%
        elite_count = max(1, population_size // 4)
        sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
        elite = [population[i] for i in sorted_indices]
        
        # Create new population
        new_population = deepcopy(elite)
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection for parents
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            mutated_child = mutate(child, generation, max_generations)
            
            # Local refinement
            refined_child = refine_solution(mutated_child)
            
            new_population.append(refined_child)
        
        population = new_population[:population_size]
    
    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to first individual if nothing was found
        return population[0]

# EVOLVE-BLOCK-END
