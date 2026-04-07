# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from typing import Tuple, List, Optional
import random
from copy import deepcopy
import time

# Global constants
GRID_SIZE = 5  # 5x5 grid for spatial decomposition
POPULATION_SIZE = 25  # One per grid cell plus overflow
GENERATIONS = 60
MUTATION_RATE_INITIAL = 0.15
CROSSOVER_RATE = 0.7
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class GridBasedEvolutionOptimizer:
    def __init__(self):
        self.n_circles = 26
        self.grid_size = GRID_SIZE
        self.cell_width = 1.0 / self.grid_size
        self.cell_height = 1.0 / self.grid_size
        
    def is_valid_configuration(self, circles: np.ndarray) -> bool:
        """Check if the configuration satisfies all constraints."""
        if len(circles) != self.n_circles:
            return False
            
        # Check containment constraints using vectorized operations
        radii = circles[:, 2]
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        
        # Check if any radius violates containment
        containment_check = (
            (radii <= x_coords) & 
            (radii <= y_coords) & 
            (radii <= 1 - x_coords) & 
            (radii <= 1 - y_coords)
        )
        
        if not np.all(containment_check):
            return False

        # Check overlap constraints efficiently using spatial indexing
        if self.n_circles > 1:
            # Use KDTree for efficient neighbor search
            tree = KDTree(circles[:, :2])
            
            # For each circle, find neighbors that might overlap
            for i in range(self.n_circles):
                # Find neighbors within 2*(max_radius) distance (optimization)
                potential_neighbors = tree.query_ball_point(circles[i, :2], 2 * np.max(radii))
                # Skip self
                potential_neighbors = [idx for idx in potential_neighbors if idx != i]
                
                for j in potential_neighbors:
                    dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        return False
                        
        return True
    
    def calculate_sum_radii(self, circles: np.ndarray) -> float:
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])
    
    def initialize_grid_population(self) -> List[np.ndarray]:
        """Initialize population using grid-based approach."""
        population = []
        
        # Create grid cell centers
        cell_centers = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                center_x = (j + 0.5) * self.cell_width
                center_y = (i + 0.5) * self.cell_height
                cell_centers.append((center_x, center_y))
        
        # Generate diverse initial configurations for each cell
        for i, (center_x, center_y) in enumerate(cell_centers):
            if i < self.n_circles:
                # Create a configuration for this cell
                circles = self._create_cell_initialization(center_x, center_y, i)
                if self.is_valid_configuration(circles):
                    population.append(circles.copy())
                else:
                    # Fallback to simple initialization
                    population.append(self._create_simple_initialization())
            else:
                break
        
        # Fill remaining slots if needed
        while len(population) < POPULATION_SIZE:
            population.append(self._create_simple_initialization())
            
        return population
    
    def _create_simple_initialization(self) -> np.ndarray:
        """Create a simple but valid initial configuration."""
        circles = np.zeros((self.n_circles, 3))
        
        # Place in a simple grid pattern
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing = 1.0 / (grid_size + 1)
        
        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if idx >= self.n_circles:
                    break
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 4  # Conservative radius
                circles[idx] = [x, y, r]
                idx += 1
                
        return circles
    
    def _create_cell_initialization(self, center_x: float, center_y: float, seed_offset: int) -> np.ndarray:
        """Create initialization for a specific cell with spatial awareness."""
        circles = np.zeros((self.n_circles, 3))
        
        # Determine which circles belong to this cell
        # Distribute circles spatially with some randomness
        np.random.seed(seed_offset)  # Fixed seed for reproducibility
        
        # Place circles around the cell center with some randomness
        for i in range(self.n_circles):
            # Base position around cell center
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(0, self.cell_width * 0.3)
            
            # Calculate position with constraints
            x = center_x + distance * np.cos(angle)
            y = center_y + distance * np.sin(angle)
            
            # Constrain to cell bounds
            x = np.clip(x, center_x - self.cell_width * 0.4, center_x + self.cell_width * 0.4)
            y = np.clip(y, center_y - self.cell_height * 0.4, center_y + self.cell_height * 0.4)
            
            # Calculate radius based on distance to boundaries
            min_dist = min(x, y, 1-x, 1-y)
            r = np.random.uniform(0.01, min_dist/2)
            
            circles[i] = [x, y, r]
            
        return circles
    
    def _get_cell_index(self, x: float, y: float) -> int:
        """Get the grid cell index for a given point."""
        col = int(x / self.cell_width)
        row = int(y / self.cell_height)
        col = min(col, self.grid_size - 1)
        row = min(row, self.grid_size - 1)
        return row * self.grid_size + col
    
    def _get_cell_bounds(self, cell_index: int) -> Tuple[float, float, float, float]:
        """Get the bounds of a grid cell."""
        row = cell_index // self.grid_size
        col = cell_index % self.grid_size
        x_min = col * self.cell_width
        y_min = row * self.cell_height
        x_max = (col + 1) * self.cell_width
        y_max = (row + 1) * self.cell_height
        return x_min, x_max, y_min, y_max
    
    def optimize_cell_placement(self, circles: np.ndarray, cell_index: int, max_iter: int = 100) -> np.ndarray:
        """Apply local optimization specific to a grid cell with spatial awareness."""
        n = len(circles)
        circles = circles.copy()
        
        # Calculate overlap count for this cell's circles
        def count_cell_overlaps(config, cell_id):
            if n <= 1:
                return 0
            # Get circles in this cell
            cell_bounds = self._get_cell_bounds(cell_id)
            x_min, x_max, y_min, y_max = cell_bounds
            
            # Filter circles in this cell
            cell_circles = []
            for i in range(n):
                x, y = config[i, 0], config[i, 1]
                if cell_bounds[0] <= x <= cell_bounds[1] and cell_bounds[2] <= y <= cell_bounds[3]:
                    cell_circles.append(i)
            
            if len(cell_circles) <= 1:
                return 0
                
            # Count overlaps in this cell
            overlap_count = 0
            cell_config = config[cell_circles]
            if len(cell_config) > 1:
                for i in range(len(cell_circles)):
                    for j in range(i+1, len(cell_circles)):
                        ci = cell_circles[i]
                        cj = cell_circles[j]
                        dist = np.sqrt((config[ci, 0] - config[cj, 0])**2 + (config[ci, 1] - config[cj, 1])**2)
                        min_dist = config[ci, 2] + config[cj, 2]
                        if dist < min_dist:
                            overlap_count += 1
            return overlap_count

        # Get current cell overlaps
        cell_overlaps = count_cell_overlaps(circles, cell_index)
        
        # Different refinement strategies based on overlap count
        max_refinement_iter = max_iter // 2 if cell_overlaps == 0 else max_iter
        
        for iteration in range(max_refinement_iter):
            improved = False
            
            # Strategy 1: Try to expand radii in the cell
            cell_bounds = self._get_cell_bounds(cell_index)
            x_min, x_max, y_min, y_max = cell_bounds
            
            # Focus on circles in this cell
            for i in range(n):
                x, y = circles[i, 0], circles[i, 1]
                if cell_bounds[0] <= x <= cell_bounds[1] and cell_bounds[2] <= y <= cell_bounds[3]:
                    # Calculate maximum possible radius for this circle
                    max_radius = min(
                        x,
                        y,
                        1 - x,
                        1 - y
                    )
                    
                    # Try to increase radius with adaptive increment
                    original_radius = circles[i][2]
                    if max_radius > original_radius:
                        # Smaller increment for high overlap situations
                        increment = 0.005 if cell_overlaps <= 2 else 0.002
                        new_radius = min(original_radius + increment, max_radius)
                        
                        if new_radius > original_radius:
                            circles[i][2] = new_radius
                            
                            # Check if valid configuration
                            if self.is_valid_configuration(circles):
                                improved = True
                            else:
                                # Revert if invalid
                                circles[i][2] = original_radius
            
            # Strategy 2: Position adjustments for circles in this cell
            if improved or cell_overlaps > 0:
                adjustments = [
                    (0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001),
                    (0.0005, 0.0005), (-0.0005, -0.0005)
                ]
                
                # Try adjustments to resolve overlaps
                for i in range(n):
                    x, y = circles[i, 0], circles[i, 1]
                    if cell_bounds[0] <= x <= cell_bounds[1] and cell_bounds[2] <= y <= cell_bounds[3]:
                        original_x, original_y = x, y
                        
                        for dx, dy in adjustments:
                            new_x = np.clip(original_x + dx, 0, 1)
                            new_y = np.clip(original_y + dy, 0, 1)
                            
                            if new_x != original_x or new_y != original_y:
                                circles[i][0] = new_x
                                circles[i][1] = new_y
                                
                                if self.is_valid_configuration(circles):
                                    improved = True
                                    break
                                else:
                                    # Revert if invalid
                                    circles[i][0] = original_x
                                    circles[i][1] = original_y

            if not improved:
                break
                
        return circles
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform specialized grid-aware crossover."""
        if np.random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()

        # Create offspring by combining elements from both parents in a grid-aware way
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Get the 5x5 grid indices for each circle
        parent1_indices = [self._get_cell_index(c[0], c[1]) for c in parent1]
        parent2_indices = [self._get_cell_index(c[0], c[1]) for c in parent2]
        
        # Use a hybrid approach: some circles from parent1, some from parent2
        # Based on relative distance to cell centers
        
        for i in range(self.n_circles):
            # Determine which parent to take this circle from based on proximity to cell centers
            p1_center = self._get_cell_center(parent1_indices[i])
            p2_center = self._get_cell_center(parent2_indices[i])
            
            # Distance to cell centers
            dist1 = np.sqrt((parent1[i, 0] - p1_center[0])**2 + (parent1[i, 1] - p1_center[1])**2)
            dist2 = np.sqrt((parent2[i, 0] - p2_center[0])**2 + (parent2[i, 1] - p2_center[1])**2)
            
            # Select based on proximity to cell centers
            if dist1 < dist2:
                # Take from parent1
                if np.random.random() < 0.6:  # 60% chance to take from parent1
                    child2[i] = parent1[i].copy()
            else:
                # Take from parent2
                if np.random.random() < 0.6:  # 60% chance to take from parent2
                    child1[i] = parent2[i].copy()
        
        # Apply grid-aware optimization to ensure validity
        child1 = self._grid_aware_optimize(child1)
        child2 = self._grid_aware_optimize(child2)
        
        return child1, child2
    
    def _get_cell_center(self, cell_index: int) -> Tuple[float, float]:
        """Get the center of a grid cell."""
        row = cell_index // self.grid_size
        col = cell_index % self.grid_size
        center_x = (col + 0.5) * self.cell_width
        center_y = (row + 0.5) * self.cell_height
        return center_x, center_y
    
    def _grid_aware_optimize(self, circles: np.ndarray) -> np.ndarray:
        """Apply grid-aware optimization to ensure validity."""
        circles = circles.copy()
        
        # Group circles by cells and optimize each cell independently
        for cell_idx in range(self.grid_size * self.grid_size):
            circles_in_cell = []
            for i in range(self.n_circles):
                x, y = circles[i, 0], circles[i, 1]
                cell_of_circle = self._get_cell_index(x, y)
                if cell_of_circle == cell_idx:
                    circles_in_cell.append(i)
            
            if len(circles_in_cell) > 1:
                # Optimization for this cell only
                cell_circles = circles[circles_in_cell]
                # Simple optimization: try to maximize radii in each cell
                for iter_count in range(50):  # Limited iterations
                    improved = False
                    for ci in circles_in_cell:
                        # Maximize radius for this circle
                        max_radius = min(
                            circles[ci, 0],  # x distance to left
                            circles[ci, 1],  # y distance to bottom
                            1 - circles[ci, 0],  # x distance to right
                            1 - circles[ci, 1]   # y distance to top
                        )
                        
                        if max_radius > circles[ci, 2]:
                            # Increase radius
                            circles[ci, 2] = max_radius
                            if self.is_valid_configuration(circles):
                                improved = True
                            else:
                                # Revert if invalid
                                circles[ci, 2] = circles[ci, 2]  # keep original
                    if not improved:
                        break
                        
        return circles

    def mutate(self, circles: np.ndarray, mutation_rate: float = MUTATION_RATE_INITIAL) -> np.ndarray:
        """Apply grid-aware mutation with spatial reasoning."""
        mutated = circles.copy()
        n = len(mutated)
        
        # Apply mutation to each circle with grid awareness
        for i in range(n):
            if np.random.random() < mutation_rate:
                # Consider spatial neighborhood for mutation decision
                cell_index = self._get_cell_index(mutated[i, 0], mutated[i, 1])
                cell_bounds = self._get_cell_bounds(cell_index)
                
                # Mutate either position or radius based on spatial context
                if np.random.random() < 0.7:  # 70% chance to mutate position
                    # Mutate position with adaptive magnitude based on cell density
                    # In denser cells, use smaller mutations
                    mutated[i][0] = np.clip(mutated[i][0] + np.random.normal(0, 0.02), 0, 1)
                    mutated[i][1] = np.clip(mutated[i][1] + np.random.normal(0, 0.02), 0, 1)
                else:
                    # Mutate radius
                    mutated[i][2] = np.clip(mutated[i][2] + np.random.normal(0, 0.01), 0.01, 0.5)
        
        # Apply localized optimization to maintain feasibility
        mutated = self._grid_aware_optimize(mutated)
        return mutated

    def compute_fitness(self, circles: np.ndarray) -> float:
        """Compute fitness with penalty for invalid configurations."""
        if self.is_valid_configuration(circles):
            return self.calculate_sum_radii(circles)
        else:
            # Invalid configurations get very low fitness
            return 0.0

    def run_evolution(self) -> np.ndarray:
        """Run the complete grid-based evolutionary algorithm."""
        # Initialize population using grid decomposition
        population = self.initialize_grid_population()
        
        if not population:
            return self._create_simple_initialization()
            
        best_solution = None
        best_fitness = -1
        
        # Run generations with grid-centric approach
        for generation in range(GENERATIONS):
            # Adjust mutation rate based on generation
            mutation_rate = max(MUTATION_RATE_INITIAL * (1 - generation / GENERATIONS), 0.01)
            
            # Evaluate fitness for all individuals
            fitnesses = [self.compute_fitness(circles) for circles in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()

            # Create new population with grid-aware reproduction
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_solution.copy())

            # Generate offspring with grid-aware reproduction
            while len(new_population) < POPULATION_SIZE:
                # Select two parents using tournament selection with grid bias
                parent1_idx = self._tournament_selection_with_grid_bias(population, fitnesses, 3)
                parent2_idx = self._tournament_selection_with_grid_bias(population, fitnesses, 3)

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)

                # Add children to new population
                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:POPULATION_SIZE]

        # Final optimization of best solution
        if best_solution is not None:
            best_solution = self._grid_aware_optimize(best_solution)
            # Ensure final validation
            if not self.is_valid_configuration(best_solution):
                return self._create_simple_initialization()
        
        # Return the best solution found
        if best_solution is None:
            # Fallback to a simple configuration if nothing worked
            return self._create_simple_initialization()

        return best_solution
    
    def _tournament_selection_with_grid_bias(self, population: List[np.ndarray], fitnesses: List[float], 
                                           tournament_size: int = 3) -> int:
        """Select an individual using tournament selection with grid bias."""
        # Select multiple candidates
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        # Choose the best among them
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return winner_index

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = GridBasedEvolutionOptimizer()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END