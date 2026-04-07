# EVOLVE-BLOCK-START
import numpy as np
import random
from copy import deepcopy
from typing import Tuple, List
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import time

class MultiResolutionGrid:
    """Multi-resolution spatial grid for efficient overlap detection"""
    
    def __init__(self):
        self.grids = {}
        self.grid_sizes = [10, 20, 40]  # Different resolutions for different scales
        
    def _get_cell_coords(self, x: float, y: float, resolution: int) -> Tuple[int, int]:
        """Get grid cell coordinates for a point"""
        cell_size = 1.0 / resolution
        return (int(x / cell_size), int(y / cell_size))
    
    def clear(self):
        """Clear all grids"""
        self.grids.clear()
        
    def add_circle(self, idx: int, x: float, y: float, radius: float):
        """Add a circle to all relevant grids"""
        for res in self.grid_sizes:
            if res not in self.grids:
                self.grids[res] = {}
            cell_coords = self._get_cell_coords(x, y, res)
            if cell_coords not in self.grids[res]:
                self.grids[res][cell_coords] = []
            self.grids[res][cell_coords].append((idx, x, y, radius))
    
    def get_neighbors(self, x: float, y: float, radius: float, resolution: int = 20) -> List[Tuple[int, float, float, float]]:
        """Get neighbors from specific resolution grid"""
        if resolution not in self.grids:
            return []
        cell_x, cell_y = self._get_cell_coords(x, y, resolution)
        
        neighbors = []
        # Check surrounding cells (3x3 grid around main cell)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in self.grids[resolution]:
                    for idx, nx, ny, nr in self.grids[resolution][neighbor_cell]:
                        # Skip self
                        if idx == -1:
                            continue
                        neighbors.append((idx, nx, ny, nr))
        return neighbors

class CirclePackHybridEvolution:
    """Hybrid evolutionary algorithm for circle packing optimization"""
    
    def __init__(self, n_circles: int = 26, max_generations: int = 1000, 
                 population_size: int = 50, seed: int = 42):
        self.n_circles = n_circles
        self.max_generations = max_generations
        self.population_size = population_size
        self.seed = seed
        
        # Initialize random seeds
        np.random.seed(seed)
        random.seed(seed)
        
        # Multi-resolution spatial indexing
        self.spatial_grid = MultiResolutionGrid()
        
        # Performance tracking
        self.best_fitness_history = []
        
    def _fibonacci_spiral_layout(self) -> np.ndarray:
        """Generate initial layout using Fibonacci spiral for better distribution"""
        circles = np.zeros((self.n_circles, 3))
        
        # Fibonacci spiral positioning
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(self.n_circles):
            # Distribute points around a circle
            theta = i * 2 * np.pi / golden_ratio
            r = np.sqrt(i / (self.n_circles - 1)) if self.n_circles > 1 else 0
            
            # Map to unit square with padding and slight randomness
            x = 0.1 + 0.8 * r * np.cos(theta) + np.random.normal(0, 0.01)
            y = 0.1 + 0.8 * r * np.sin(theta) + np.random.normal(0, 0.01)
            
            # Clamp to valid range
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            circles[i] = [x, y, 0.02]  # Initial small radius
            
        return circles
    
    def _poisson_disk_sampling(self) -> np.ndarray:
        """Generate points using Poisson disk sampling for high quality distribution"""
        circles = np.zeros((self.n_circles, 3))
        
        # Estimate radius based on area
        max_radius = np.sqrt(1.0 / (self.n_circles * np.pi))
        radius = max_radius * 0.8  # Leave some margin
        
        # Create a simple Poisson disk sampling approximation
        points = []
        cell_size = radius / np.sqrt(2)
        grid_width = int(np.ceil(1.0 / cell_size))
        grid_height = int(np.ceil(1.0 / cell_size))
        grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
        
        # Add first point randomly
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        points.append((x, y))
        grid[int(x / cell_size)][int(y / cell_size)] = (x, y)
        
        # Continue with Poisson sampling approach
        active_list = [(x, y)]
        while len(points) < self.n_circles and active_list:
            # Pick a random point from active list
            idx = np.random.randint(len(active_list))
            px, py = active_list[idx]
            
            # Try up to 30 random points around the selected point
            found_point = False
            for _ in range(30):
                angle = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(radius, 2*radius)
                x = px + r * np.cos(angle)
                y = py + r * np.sin(angle)
                
                # Check if new point is within bounds
                if x < 0 or x >= 1 or y < 0 or y >= 1:
                    continue
                
                # Check if point is close to existing points
                grid_x = int(x / cell_size)
                grid_y = int(y / cell_size)
                
                # Check nearby cells
                valid = True
                for i in range(max(0, grid_x-2), min(grid_width, grid_x+3)):
                    for j in range(max(0, grid_y-2), min(grid_height, grid_y+3)):
                        if grid[i][j] is not None:
                            gx, gy = grid[i][j]
                            dx = x - gx
                            dy = y - gy
                            if dx*dx + dy*dy < radius*radius:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    points.append((x, y))
                    active_list.append((x, y))
                    grid[grid_x][grid_y] = (x, y)
                    found_point = True
                    break
            
            # Remove point from active list if no valid point found
            if not found_point:
                del active_list[idx]
        
        # If we don't have enough points, fill with random ones
        while len(points) < self.n_circles:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            points.append((x, y))
        
        # Take only first n_circles points
        points = points[:self.n_circles]
        
        # Convert to circles array
        for i, (x, y) in enumerate(points):
            circles[i] = [x, y, 0.02]
            
        return circles
    
    def _compute_initial_radii(self, circles: np.ndarray) -> np.ndarray:
        """Compute initial optimal radii for circles based on spacing"""
        result = deepcopy(circles)
        
        # Compute pairwise distances using cKDTree for efficiency
        coords = result[:, :2]
        try:
            tree = cKDTree(coords)
            # Get nearest neighbors for each point (excluding itself)
            distances, indices = tree.query(coords, k=min(10, self.n_circles), p=2)
            
            # For each circle, compute max possible radius
            for i in range(self.n_circles):
                # Radius constrained by boundaries
                boundary_radius = min(
                    result[i, 0],  # left boundary
                    1 - result[i, 0],  # right boundary
                    result[i, 1],  # bottom boundary
                    1 - result[i, 1]  # top boundary
                )
                
                # Radius constrained by neighbors
                neighbor_min_dist = np.inf
                for j in range(len(indices[i])):
                    if indices[i][j] != i:  # Not self
                        dist = distances[i][j]
                        if dist > 0:  # Not identical points
                            neighbor_min_dist = min(neighbor_min_dist, dist)
                
                neighbor_radius = neighbor_min_dist / 2.0 if neighbor_min_dist != np.inf else 0.05
                
                # Set final radius as minimum of all constraints
                result[i, 2] = min(boundary_radius, neighbor_radius, 0.05)
                
        except Exception:
            # Fallback to brute force method if KDTree fails
            # Compute pairwise distances
            distances = cdist(coords, coords)
            np.fill_diagonal(distances, float('inf'))
            
            # For each circle, compute max possible radius
            for i in range(self.n_circles):
                # Radius constrained by boundaries
                boundary_radius = min(
                    result[i, 0],  # left boundary
                    1 - result[i, 0],  # right boundary
                    result[i, 1],  # bottom boundary
                    1 - result[i, 1]  # top boundary
                )
                
                # Radius constrained by neighbors
                neighbor_min_dist = np.min(distances[i]) if self.n_circles > 1 else float('inf')
                neighbor_radius = neighbor_min_dist / 2.0 if neighbor_min_dist != float('inf') else 0.05
                
                # Set final radius as minimum of all constraints
                result[i, 2] = min(boundary_radius, neighbor_radius, 0.05)
            
        return result
    
    def initialize_population(self) -> List[np.ndarray]:
        """Initialize diverse population with hybrid strategies"""
        population = []
        
        # Generate multiple diverse initial configurations
        for _ in range(self.population_size):
            # Alternate between initialization strategies
            strategy = random.choice(['fibonacci', 'poisson'])
            
            if strategy == 'fibonacci':
                circles = self._fibonacci_spiral_layout()
            else:
                circles = self._poisson_disk_sampling()
            
            # Compute optimal initial radii
            circles = self._compute_initial_radii(circles)
            
            # Apply small random modifications
            for i in range(self.n_circles):
                if random.random() < 0.3:  # 30% chance to modify
                    # Small position perturbation
                    circles[i, 0] += np.random.normal(0, 0.01)
                    circles[i, 1] += np.random.normal(0, 0.01)
                    
                    # Clamp positions
                    circles[i, 0] = max(0.01, min(0.99, circles[i, 0]))
                    circles[i, 1] = max(0.01, min(0.99, circles[i, 1]))
            
            # Ensure all circles are valid
            circles = self._ensure_validity(circles)
            population.append(circles)
            
        return population
    
    def _ensure_validity(self, circles: np.ndarray) -> np.ndarray:
        """Ensure circles satisfy all constraints"""
        result = deepcopy(circles)
        
        # Fix containment violations
        for i in range(self.n_circles):
            x, y, r = result[i]
            # Ensure containment with margin
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            result[i] = [x, y, r]
            
        return result
    
    def is_valid(self, circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and non-overlapping"""
        n = len(circles)
        
        # Clear spatial grid
        self.spatial_grid.clear()
        
        # First check boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
            # Add to spatial grid for overlap checking
            self.spatial_grid.add_circle(i, x, y, r)
        
        # Then check overlap constraints using spatial grid
        for i in range(n):
            x1, y1, r1 = circles[i]
            
            # Use different resolution grids for performance
            candidates = self.spatial_grid.get_neighbors(x1, y1, r1, resolution=20)
            
            # Check actual overlaps
            for _, x2, y2, r2 in candidates:
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False
                    
        return True
    
    def calculate_density_fitness(self, circles: np.ndarray, sum_radii: float) -> float:
        """Calculate density-based fitness component"""
        # Calculate actual filled area
        total_area = np.sum(np.pi * circles[:, 2]**2)
        density = total_area / 1.0  # Unit square area = 1
        
        # Normalize by maximum possible area for 26 circles
        max_area = self.n_circles * np.pi * (1.0/(self.n_circles * np.pi))**2
        normalized_density = density / max_area if max_area > 0 else 0
        
        return normalized_density
    
    def evaluate_fitness(self, circles: np.ndarray, generation: int = 0) -> float:
        """Enhanced fitness evaluation with better penalty system"""
        if not self.is_valid(circles):
            # Progressive constraint relaxation: reduce penalty weights in early generations
            penalty_weight = max(0.2, 1.0 - (generation / self.max_generations) * 0.8)

            # Quick penalty calculation - early termination for invalid solutions
            total_penalty = 0
            
            # Boundary violations - weighted more heavily
            for i in range(len(circles)):
                x, y, r = circles[i]
                if x - r < 0:
                    total_penalty += abs(x - r) * 2000 * penalty_weight
                if x + r > 1:
                    total_penalty += abs(x + r - 1) * 2000 * penalty_weight
                if y - r < 0:
                    total_penalty += abs(y - r) * 2000 * penalty_weight
                if y + r > 1:
                    total_penalty += abs(y + r - 1) * 2000 * penalty_weight
            
            # Overlap violations - heavier penalty for severe violations
            self.spatial_grid.clear()
            for i in range(len(circles)):
                x, y, r = circles[i]
                self.spatial_grid.add_circle(i, x, y, r)
                
            # Detect overlaps with early termination
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                candidates = self.spatial_grid.get_neighbors(x1, y1, r1, resolution=20)
                for _, x2, y2, r2 in candidates:
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        overlap = max(0, r1 + r2 - distance)
                        total_penalty += overlap * 20000 * penalty_weight  # Increased penalty
            
            return -total_penalty
        
        # Valid configuration: maximize sum of radii plus bonus for density
        sum_radii = np.sum(circles[:, 2])
        density_bonus = self.calculate_density_fitness(circles, sum_radii)
        
        return sum_radii + density_bonus * 0.1  # Small bonus for density
    
    def _adaptive_mutation(self, circles: np.ndarray, generation: int, 
                          max_generations: int) -> np.ndarray:
        """Apply adaptive mutation with multiple strategies"""
        mutated = deepcopy(circles)
        
        # Adaptive mutation rate with sigmoid decay (but with more aggressive early phase)
        gen_progress = generation / max_generations
        mutation_rate = 0.2 - (0.15 * gen_progress)
        
        # Apply different mutation strategies based on generation
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Choose mutation type based on progress
                mutation_type = random.choices(
                    ['position', 'radius', 'large_shift'],
                    weights=[0.6, 0.3, 0.1]
                )[0]
                
                if mutation_type == 'position':
                    # Small position mutation
                    mutated[i, 0] += np.random.normal(0, 0.01)
                    mutated[i, 1] += np.random.normal(0, 0.01)
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))
                    
                elif mutation_type == 'radius':
                    # Small radius mutation
                    mutated[i, 2] += np.random.normal(0, 0.005)
                    mutated[i, 2] = max(0.001, mutated[i, 2])
                    
                elif mutation_type == 'large_shift':
                    # Large shift for global exploration
                    mutated[i, 0] += np.random.uniform(-0.05, 0.05)
                    mutated[i, 1] += np.random.uniform(-0.05, 0.05)
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))
        
        return mutated
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring via crossover with variable strategy"""
        child = deepcopy(parent1)
        n = len(parent1)
        
        # Use uniform crossover with some random selection
        for i in range(n):
            if random.random() < 0.5:
                child[i] = parent2[i].copy()
            
        return child
    
    def tournament_selection(self, population: List[np.ndarray], k: int = 3, 
                           generation: int = 0) -> np.ndarray:
        """Select individual using tournament selection with fitness sharing"""
        selected = random.sample(population, k)
        # Evaluate with current generation context
        return max(selected, key=lambda ind: self.evaluate_fitness(ind, generation))
    
    def evolve(self) -> np.ndarray:
        """Main evolution loop with multi-stage approach"""
        # Initialize population
        population = self.initialize_population()
        
        best_fitness = float('-inf')
        best_solution = None
        start_time = time.time()
        
        for generation in range(self.max_generations):
            # Evaluate fitness for entire population
            fitness_scores = [self.evaluate_fitness(individual, generation) for individual in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_solution = deepcopy(population[max_fitness_idx])
                self.best_fitness_history.append(best_fitness)
            
            # Early termination check
            if time.time() - start_time > 55:  # Leave 5 seconds for cleanup
                break
            
            # Elitism: keep top 20%
            elite_count = max(1, self.population_size // 5)
            sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
            elite = [population[i] for i in sorted_indices]
            
            # Create new population
            new_population = deepcopy(elite)
            
            # Generate offspring through crossover and mutation
            while len(new_population) < self.population_size:
                # Tournament selection for parents
                parent1 = self.tournament_selection(population, generation=generation)
                parent2 = self.tournament_selection(population, generation=generation)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation with adaptive parameters  
                mutated_child = self._adaptive_mutation(child, generation, self.max_generations)
                
                # Ensure validity
                mutated_child = self._ensure_validity(mutated_child)
                
                new_population.append(mutated_child)
            
            population = new_population[:self.population_size]
            
            # Progress reporting
            if generation % 100 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Return the best solution found
        if best_solution is not None:
            return best_solution
        else:
            # Fallback to first individual if nothing was found
            return population[0]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackHybridEvolution(n_circles=26, max_generations=1000, population_size=50, seed=42)
    return optimizer.evolve()

# EVOLVE-BLOCK-END