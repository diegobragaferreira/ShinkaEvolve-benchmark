# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
import time
from collections import defaultdict

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Rectangle dimensions: optimized width=1.5, height=0.5 (perimeter = 4)
    rect_width = 1.5
    rect_height = 0.5
    
    # Number of circles
    n_circles = 21
    
    # Initialize container bounds
    x_min, x_max = 0, rect_width
    y_min, y_max = 0, rect_height
    
    # Create toolbox for evolutionary algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define bounds for each variable (x, y, radius) for all circles
    # Each circle has 3 variables: x, y, radius
    # For x: [radius, rect_width-radius]
    # For y: [radius, rect_height-radius] 
    # For radius: [0.001, min(rect_width, rect_height)/2]
    
    max_radius = min(rect_width, rect_height) / 2
    
    def create_individual():
        individual = []
        for _ in range(n_circles):
            # Random x position
            x = random.uniform(0.001, rect_width - 0.001)
            # Random y position
            y = random.uniform(0.001, rect_height - 0.001)
            # Random radius (smaller for better packing)
            r = random.uniform(0.001, max_radius * 0.8)
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    def evaluate(individual):
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)
        
        # Check constraints
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for invalid positions
        penalty = 0
        
        # Check boundary constraints efficiently using grid-based approach
        grid = SpatialGrid(rect_width, rect_height, max_radius)
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Boundary constraint check
            if x - r < x_min or x + r > x_max or y - r < y_min or y + r > y_max:
                penalty += 1000000  # Large penalty for boundary violation
                continue
            
            # Add circle to grid for overlap checking
            grid.add_circle(i, x, y, r)
        
        # Check overlap constraints using spatial grid for efficiency
        if len(circles) > 1:
            # Use grid-based neighbors for efficient overlap detection
            for i in range(len(circles)):
                x, y, r = circles[i]
                nearby_circles = grid.get_neighbors(i, x, y, r)
                
                for j in nearby_circles:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        # Distance between centers
                        dist = np.sqrt((x2-x)**2 + (y2-y)**2)
                        
                        # If circles overlap, add penalty
                        if dist < (r + r2):
                            # Penalty proportional to overlap amount
                            overlap = (r + r2) - dist
                            penalty += 100000 * overlap
        
        # Return fitness (negative because we want to maximize, but DEAP minimizes)
        return (total_radius - penalty,),
    
    def mutate(individual):
        # Mutate each gene with small changes
        for i in range(len(individual)):
            if random.random() < 0.1:  # 10% chance to mutate
                if i % 3 == 0:  # x coordinate
                    individual[i] += random.gauss(0, 0.05)
                    individual[i] = max(0.001, min(rect_width - 0.001, individual[i]))
                elif i % 3 == 1:  # y coordinate
                    individual[i] += random.gauss(0, 0.05)
                    individual[i] = max(0.001, min(rect_height - 0.001, individual[i]))
                else:  # radius
                    individual[i] += random.gauss(0, 0.01)
                    individual[i] = max(0.001, min(max_radius * 0.8, individual[i]))
        return individual,
    
    def crossover(ind1, ind2):
        # Uniform crossover
        if random.random() < 0.8:  # 80% chance of crossover
            for i in range(len(ind1)):
                if random.random() < 0.5:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    # Register functions in toolbox
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Multi-start optimization with different initial populations
    best_score = 0
    best_solution = None
    
    # Run multiple independent evolutions with different random seeds
    for run in range(3):
        random.seed(42 + run * 100)
        np.random.seed(42 + run * 100)
        
        # Create fresh population for this run
        population = toolbox.population(n=50)
        
        # Statistics to track evolution
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run evolution
        try:
            population, logbook = algorithms.eaSimple(population, toolbox, 
                                                    cxpb=0.8, mutpb=0.3, 
                                                    ngen=50, stats=stats, 
                                                    verbose=False)
        except Exception as e:
            # Fallback to simple heuristic if evolution fails
            current_solution = heuristic_initialization(rect_width, rect_height)
            current_score = np.sum(current_solution[:, 2])
        else:
            # Get best solution from this run
            best_ind = tools.selBest(population, 1)[0]
            current_solution = np.array(best_ind).reshape(-1, 3)
            current_score = np.sum(current_solution[:, 2])
        
        # Apply boundary optimization to improve edge circles
        if current_solution is not None and current_score > best_score:
            final_solution = boundary_optimization(current_solution, rect_width, rect_height)
            final_score = np.sum(final_solution[:, 2])
            if final_score > best_score:
                best_solution = final_solution
                best_score = final_score

    # If we didn't get a solution, fallback to heuristic
    if best_solution is None:
        best_solution = heuristic_initialization(rect_width, rect_height)
    
    return best_solution

class SpatialGrid:
    """Spatial grid for efficient neighbor lookups"""
    
    def __init__(self, width, height, min_radius):
        self.width = width
        self.height = height
        self.cell_size = min_radius * 2
        self.grid = defaultdict(list)
        self.circle_positions = {}
    
    def _get_cell_coords(self, x, y):
        """Get cell coordinates for given position"""
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        return (col, row)
    
    def add_circle(self, idx, x, y, r):
        """Add circle to spatial grid"""
        # Store the circle's center and radius
        self.circle_positions[idx] = (x, y, r)
        
        # Get all cells that this circle intersects
        min_col, min_row = self._get_cell_coords(x - r, y - r)
        max_col, max_row = self._get_cell_coords(x + r, y + r)
        
        # Add to all intersected cells
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                self.grid[(col, row)].append(idx)
    
    def get_neighbors(self, idx, x, y, r):
        """Get all neighboring circles efficiently"""
        neighbors = set()
        
        # Get all cells that this circle intersects
        min_col, min_row = self._get_cell_coords(x - r, y - r)
        max_col, max_row = self._get_cell_coords(x + r, y + r)
        
        # Check all intersected cells for neighbors
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                if (col, row) in self.grid:
                    for other_idx in self.grid[(col, row)]:
                        if other_idx != idx:
                            neighbors.add(other_idx)
        
        return list(neighbors)

def boundary_optimization(circles, rect_width, rect_height):
    """Refine solution to better handle boundary constraints"""
    # Parameters for local optimization
    max_iter = 30
    learning_rate = 0.05
    
    # Container bounds
    x_min, x_max = 0, rect_width
    y_min, y_max = 0, rect_height
    
    # Precompute pairwise distances for collision detection
    n_circles = len(circles)
    
    for iteration in range(max_iter):
        # Calculate forces between circles
        forces = np.zeros_like(circles)
        
        # Calculate mutual repulsion forces using vectorized operations
        if n_circles > 1:
            centers = circles[:, :2]
            radii = circles[:, 2]
            
            # Compute all pairwise center differences and distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            
            # Create mask for non-diagonal elements
            mask = ~np.eye(n_circles, dtype=bool)
            
            # Check for overlaps
            overlap_mask = (distances < (radii[:, np.newaxis] + radii[np.newaxis, :])) & mask
            
            # Only compute forces for overlapping pairs
            overlap_indices = np.where(overlap_mask)
            if len(overlap_indices[0]) > 0:
                for i, j in zip(overlap_indices[0], overlap_indices[1]):
                    dx = diff[i, j, 0]
                    dy = diff[i, j, 1]
                    dist = distances[i, j]
                    
                    if dist > 0:
                        force_magnitude = 1.0 / (dist * dist)
                        forces[i, 0] -= force_magnitude * dx / dist
                        forces[i, 1] -= force_magnitude * dy / dist
                        forces[j, 0] += force_magnitude * dx / dist
                        forces[j, 1] += force_magnitude * dy / dist
        
        # Apply boundary forces for circles near edges
        for i in range(n_circles):
            x, y, r = circles[i]
            
            # Boundary force to push circles away from edges
            boundary_force_x = 0
            boundary_force_y = 0
            
            # Left edge
            if x - r < x_min:
                boundary_force_x += (x_min - (x - r)) * 50
            # Right edge  
            elif x + r > x_max:
                boundary_force_x += (x_max - (x + r)) * 50
                
            # Bottom edge
            if y - r < y_min:
                boundary_force_y += (y_min - (y - r)) * 50
            # Top edge
            elif y + r > y_max:
                boundary_force_y += (y_max - (y + r)) * 50
            
            forces[i, 0] += boundary_force_x
            forces[i, 1] += boundary_force_y
            
            # Move based on force and learning rate
            new_x = x + learning_rate * forces[i, 0]
            new_y = y + learning_rate * forces[i, 1]
            
            # Boundary constraints
            new_x = max(r, min(rect_width - r, new_x))
            new_y = max(r, min(rect_height - r, new_y))
            
            # Update circle position
            circles[i, 0] = new_x
            circles[i, 1] = new_y
    
    return circles

def heuristic_initialization(rect_width, rect_height):
    """Initialize with a good heuristic placement"""
    # Use hexagonal packing for initial layout
    circles = []
    
    # Number of rows and columns for hexagonal arrangement
    rows = 3
    cols = 7
    
    # Calculate spacing
    spacing_x = rect_width / cols
    spacing_y = rect_height / rows
    
    # Hexagonal offset
    hex_offset = spacing_x * 0.5
    
    # Place circles in hexagonal pattern
    for row in range(rows):
        for col in range(cols):
            # Add offset to odd rows
            x_offset = hex_offset if row % 2 == 1 else 0
            x = (col * spacing_x) + x_offset + spacing_x/2
            y = (row * spacing_y) + spacing_y/2
            
            # Ensure we're within bounds
            if x >= spacing_x/2 and x <= rect_width - spacing_x/2:
                if y >= spacing_y/2 and y <= rect_height - spacing_y/2:
                    # Radius is determined by spacing but small enough to allow some growth
                    max_radius = min(spacing_x, spacing_y) / 3
                    r = max_radius * 0.8
                    circles.append([x, y, r])
    
    # Fill remaining slots with random placements
    remaining = 21 - len(circles)
    for _ in range(remaining):
        x = random.uniform(0.001, rect_width - 0.001)
        y = random.uniform(0.001, rect_height - 0.001)
        r = random.uniform(0.001, min(rect_width, rect_height) / 4)
        circles.append([x, y, r])
    
    return np.array(circles)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
