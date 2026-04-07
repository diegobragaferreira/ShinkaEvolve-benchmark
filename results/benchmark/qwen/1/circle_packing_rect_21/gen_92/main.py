# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Dict, Set

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class GridBasedCirclePacker:
    def __init__(self, rect_width: float = 1.0, rect_height: float = 1.0, cell_size: float = 0.2):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.cell_size = cell_size
        self.grid_width = int(np.ceil(rect_width / cell_size))
        self.grid_height = int(np.ceil(rect_height / cell_size))
        self.grid = {}
        
    def clear_grid(self):
        self.grid = {}
        
    def get_grid_coords(self, x: float, y: float, r: float) -> Set[Tuple[int, int]]:
        """Get all grid cells that might contain this circle."""
        min_col = max(0, int((x - r) / self.cell_size))
        max_col = min(self.grid_width - 1, int((x + r) / self.cell_size))
        min_row = max(0, int((y - r) / self.cell_size))
        max_row = min(self.grid_height - 1, int((y + r) / self.cell_size))
        
        coords = set()
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                coords.add((col, row))
        return coords
    
    def add_circle_to_grid(self, index: int, x: float, y: float, r: float):
        """Add a circle to the grid."""
        coords = self.get_grid_coords(x, y, r)
        for coord in coords:
            if coord not in self.grid:
                self.grid[coord] = []
            self.grid[coord].append(index)
    
    def remove_circle_from_grid(self, index: int, x: float, y: float, r: float):
        """Remove a circle from the grid."""
        coords = self.get_grid_coords(x, y, r)
        for coord in coords:
            if coord in self.grid and index in self.grid[coord]:
                self.grid[coord].remove(index)
    
    def check_collision_with_grid(self, x: float, y: float, r: float) -> bool:
        """Check collision with existing circles in grid with O(n) complexity."""
        coords = self.get_grid_coords(x, y, r)
        for coord in coords:
            if coord in self.grid:
                for idx in self.grid[coord]:
                    # Check actual distance for this specific circle
                    # This assumes we have access to the circle data somewhere
                    pass
        return False
    
    def get_neighbors_in_grid(self, x: float, y: float, r: float) -> List[int]:
        """Get all circles that might collide with the given circle."""
        neighbors = []
        coords = self.get_grid_coords(x, y, r)
        for coord in coords:
            if coord in self.grid:
                neighbors.extend(self.grid[coord])
        return neighbors

def check_constraints(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """Efficiently check if all circles satisfy the constraints with early termination."""
    n = len(circles)
    
    # Check boundary constraints first
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            return False
    
    # Use grid-based spatial indexing for efficient overlap checking
    if n > 1:
        packer = GridBasedCirclePacker(rect_width, rect_height)
        
        # Place all circles in grid
        for i in range(n):
            x, y, r = circles[i]
            packer.add_circle_to_grid(i, x, y, r)
            
        # Check for overlaps
        for i in range(n):
            x1, y1, r1 = circles[i]
            neighbors = packer.get_neighbors_in_grid(x1, y1, r1)
            
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < (r1 + r2):
                        return False
    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as the sum of radii with constraint validation."""
    if not check_constraints(circles):
        return -np.inf
    
    return np.sum(circles[:, 2])

def create_initial_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Create a high-quality initial solution using a hybrid approach."""
    circles = np.zeros((21, 3))
    
    # Strategy 1: Hexagonal pattern for dense core
    rows = 5
    cols = 5
    grid_width = rect_width * 0.8
    grid_height = rect_height * 0.8
    cell_width = grid_width / cols
    cell_height = grid_height / rows
    hex_radius = min(cell_width, cell_height) * 0.3
    
    placed = 0
    
    # Place circles in hexagonal pattern
    for row in range(rows):
        if placed >= 21:
            break
        for col in range(cols):
            if placed >= 21:
                break
            offset = (row % 2) * (cell_width / 2)
            x = offset + col * cell_width + cell_width / 2 + (rect_width - grid_width) / 2
            y = row * cell_height + cell_height / 2 + (rect_height - grid_height) / 2
            
            # Adjust radius to prevent boundary issues
            max_radius = min(x, y, rect_width - x, rect_height - y)
            r = min(hex_radius, max_radius * 0.8)
            
            circles[placed] = [x, y, r]
            placed += 1
    
    # Strategy 2: Fill remaining positions with strategic placement
    # Place some near edges for better boundary utilization
    edge_positions = [
        (rect_width * 0.1, rect_height * 0.1),
        (rect_width * 0.9, rect_height * 0.1),
        (rect_width * 0.1, rect_height * 0.9),
        (rect_width * 0.9, rect_height * 0.9),
        (rect_width * 0.5, rect_height * 0.1),
        (rect_width * 0.5, rect_height * 0.9),
        (rect_width * 0.1, rect_height * 0.5),
        (rect_width * 0.9, rect_height * 0.5)
    ]
    
    for x, y in edge_positions:
        if placed >= 21:
            break
        r = min(x, y, rect_width - x, rect_height - y) * 0.1
        circles[placed] = [x, y, r]
        placed += 1
    
    # Strategy 3: Fill remainder with random but reasonable placement
    for i in range(placed, 21):
        attempts = 0
        while attempts < 1000:
            x = np.random.uniform(0.05, rect_width - 0.05)
            y = np.random.uniform(0.05, rect_height - 0.05)
            r = np.random.uniform(0.01, min(x, y, rect_width - x, rect_height - y) * 0.2)
            
            # Check if this circle would overlap with any existing circle
            valid_placement = True
            for j in range(i):
                existing_x, existing_y, existing_r = circles[j]
                distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if distance < (r + existing_r):
                    valid_placement = False
                    break
            
            if valid_placement:
                circles[i] = [x, y, r]
                break
            attempts += 1
    
    return circles

def mutate(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0, 
           generation: int = 0, max_generations: int = 100) -> np.ndarray:
    """Mutation operator with boundary-aware strategy and variable intensity."""
    mutated = circles.copy()
    
    # Variable mutation intensity based on generation (reduce later)
    mutation_intensity = max(0.1, 0.3 * (1.0 - generation / max_generations))
    
    for i in range(21):
        if np.random.random() < 0.3:  # 30% mutation rate
            # Bias toward position for early generations, radius later
            mutation_type = np.random.choice(['position', 'radius'], 
                                            p=[0.7 + 0.2 * (generation / max_generations), 
                                               0.3 - 0.2 * (generation / max_generations)])
            
            if mutation_type == 'position':
                # Boundary-aware position mutation
                step_size = 0.05 * mutation_intensity
                
                # Increase chance of expansion moves for edge circles
                x, y, r = mutated[i]
                edge_distance = min(x, y, rect_width - x, rect_height - y)
                if edge_distance < r * 2:  # Near boundary
                    step_size *= 2.0  # Larger moves for boundary circles
                
                mutated[i, 0] += np.random.normal(0, step_size)
                mutated[i, 1] += np.random.normal(0, step_size)
                
                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], r, rect_width - r)
                mutated[i, 1] = np.clip(mutated[i, 1], r, rect_height - r)
                
            else:
                # Radius mutation with boundary awareness
                scale_factor = np.exp(np.random.normal(0, 0.2 * mutation_intensity))
                mutated[i, 2] *= scale_factor
                mutated[i, 2] = max(0.001, mutated[i, 2])
                
                # Boundary awareness for radius
                x, y, r = mutated[i]
                max_possible_radius = min(x, y, rect_width - x, rect_height - y)
                mutated[i, 2] = min(mutated[i, 2], max_possible_radius * 0.9)
    
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Improved crossover operator with strategic swapping."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Strategy-based crossover
    for i in range(21):
        # For 50% of circles, swap entire circle data  
        if np.random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
        else:
            # Blend positions with strategic weights
            pos_blend = np.random.random() * 0.3 + 0.35  # [0.35, 0.65]
            rad_blend = np.random.random() * 0.3 + 0.35  # [0.35, 0.65]
            
            child1[i, :2] = parent1[i, :2] * pos_blend + parent2[i, :2] * (1 - pos_blend)
            child2[i, :2] = parent1[i, :2] * (1 - pos_blend) + parent2[i, :2] * pos_blend
            
            child1[i, 2] = parent1[i, 2] * rad_blend + parent2[i, 2] * (1 - rad_blend)
            child2[i, 2] = parent1[i, 2] * (1 - rad_blend) + parent2[i, 2] * rad_blend
            
            # Ensure positive radii
            child1[i, 2] = max(0.001, child1[i, 2])
            child2[i, 2] = max(0.001, child2[i, 2])
    
    return child1, child2

def repair_solution(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Repair solution with smart conflict resolution."""
    repaired = circles.copy()
    
    # Fix boundary violations first
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        x = np.clip(x, r, rect_width - r)
        y = np.clip(y, r, rect_height - r)
        repaired[i] = [x, y, r]
    
    # Resolve overlaps using a faster approach
    packer = GridBasedCirclePacker(rect_width, rect_height)
    
    # Rebuild grid
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        packer.add_circle_to_grid(i, x, y, r)
    
    # Iterative conflict resolution
    for iteration in range(50):
        conflicts = []
        # Check for potential conflicts using grid
        for i in range(len(repaired)):
            x1, y1, r1 = repaired[i]
            neighbors = packer.get_neighbors_in_grid(x1, y1, r1)
            
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < (r1 + r2):
                        conflicts.append((i, j))
        
        if not conflicts:
            break
            
        # Resolve conflicts with prioritization
        for i, j in conflicts:
            x1, y1, r1 = repaired[i]
            x2, y2, r2 = repaired[j]
            
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                # Move circles away from each other
                overlap = (r1 + r2) - distance
                move_distance = overlap / 2
                
                dx_norm = dx / distance
                dy_norm = dy / distance
                
                # Apply movement (prioritize boundary preservation)
                repaired[i, 0] -= dx_norm * move_distance * 0.5
                repaired[i, 1] -= dy_norm * move_distance * 0.5
                repaired[j, 0] += dx_norm * move_distance * 0.5
                repaired[j, 1] += dy_norm * move_distance * 0.5
                
                # Keep within bounds
                repaired[i, 0] = np.clip(repaired[i, 0], r1, rect_width - r1)
                repaired[i, 1] = np.clip(repaired[i, 1], r1, rect_height - r1)
                repaired[j, 0] = np.clip(repaired[j, 0], r2, rect_width - r2)
                repaired[j, 1] = np.clip(repaired[j, 1], r2, rect_height - r2)
                
                # Rebuild grid for next iteration
                packer.clear_grid()
                for k in range(len(repaired)):
                    x, y, r = repaired[k]
                    packer.add_circle_to_grid(k, x, y, r)
    
    return repaired

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    rect_width = 1.5
    rect_height = 0.5

    # Parameters for evolutionary algorithm
    population_size = 40
    generations = 120
    elite_size = 10
    tournament_size = 5
    max_stagnation = 25

    # Create two populations for hybrid approach
    pop1 = []  # Exploration population
    pop2 = []  # Refinement population
    
    # Initialize both populations
    for _ in range(population_size // 2):
        sol1 = create_initial_solution(rect_width, rect_height)
        pop1.append(sol1)
        
        sol2 = create_initial_solution(rect_width, rect_height)
        pop2.append(sol2)
    
    # Track best fitness for convergence detection
    previous_best = -np.inf
    stagnation_count = 0
    
    # Evolutionary algorithm
    for generation in range(generations):
        # Evaluate fitness for both populations
        fitness_scores1 = [evaluate_fitness(individual) for individual in pop1]
        fitness_scores2 = [evaluate_fitness(individual) for individual in pop2]
        
        # Sort populations
        sorted_indices1 = np.argsort(fitness_scores1)[::-1]
        pop1 = [pop1[i] for i in sorted_indices1]
        fitness_scores1 = [fitness_scores1[i] for i in sorted_indices1]
        
        sorted_indices2 = np.argsort(fitness_scores2)[::-1]
        pop2 = [pop2[i] for i in sorted_indices2]
        fitness_scores2 = [fitness_scores2[i] for i in sorted_indices2]
        
        # Merge populations with elitism
        merged_population = pop1[:elite_size] + pop2[:elite_size]
        
        # Add some diversity from best of each population
        merged_population.append(pop1[0])
        merged_population.append(pop2[0])
        
        # Create offspring using mixed strategy
        new_population = merged_population[:]
        
        # Generate offspring until we have enough
        while len(new_population) < population_size:
            # Select from both populations based on fitness
            if np.random.random() < 0.7:  # 70% from population 1
                parent1_idx = np.random.choice(min(10, len(pop1)))
                parent2_idx = np.random.choice(min(10, len(pop2)))
                parent1 = pop1[parent1_idx].copy()
                parent2 = pop2[parent2_idx].copy()
            else:  # 30% from population 2
                parent1_idx = np.random.choice(min(10, len(pop2)))
                parent2_idx = np.random.choice(min(10, len(pop1)))
                parent1 = pop2[parent1_idx].copy()
                parent2 = pop1[parent2_idx].copy()
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutate with generation-aware parameters
            child1 = mutate(child1, rect_width, rect_height, generation, generations)
            child2 = mutate(child2, rect_width, rect_height, generation, generations)
            
            # Repair
            child1 = repair_solution(child1, rect_width, rect_height)
            child2 = repair_solution(child2, rect_width, rect_height)
            
            new_population.extend([child1, child2])
        
        # Keep only the required population size
        pop1 = new_population[:population_size//2]
        pop2 = new_population[population_size//2:population_size]
        
        # Combine populations for final evaluation
        all_solutions = pop1 + pop2
        all_fitness = [evaluate_fitness(sol) for sol in all_solutions]
        
        # Keep track of best solution
        best_idx = np.argmax(all_fitness)
        current_best = all_fitness[best_idx]
        
        # Convergence detection
        if abs(current_best - previous_best) < 1e-6:
            stagnation_count += 1
        else:
            stagnation_count = 0
        previous_best = current_best
        
        # Early stopping if stagnated too long
        if stagnation_count > max_stagnation:
            print(f"Early stopping at generation {generation} due to convergence")
            break
            
        # Print progress
        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {current_best:.6f}")
    
    # Return the best solution
    all_solutions = pop1 + pop2
    all_fitness = [evaluate_fitness(sol) for sol in all_solutions]
    best_idx = np.argmax(all_fitness)
    best_solution = all_solutions[best_idx]
    
    # Final validation
    final_fitness = evaluate_fitness(best_solution)
    if final_fitness == -np.inf:
        print("Warning: Final solution violated constraints. Returning fallback.")
        # Fallback to best valid solution found
        for sol in all_solutions:
            if evaluate_fitness(sol) > -np.inf:
                return sol
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")