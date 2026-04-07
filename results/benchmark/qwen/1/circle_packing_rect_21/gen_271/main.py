# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional
import time
import random

# Set seeds for determinism
random.seed(42)
np.random.seed(42)

class CirclePackingProblem:
    def __init__(self, n_circles: int = 21, rect_width: float = 1.0, rect_height: float = 1.0):
        self.n_circles = n_circles
        self.rect_width = rect_width
        self.rect_height = rect_height
        
    def is_valid_solution(self, circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and non-overlapping using efficient spatial hashing."""
        n = len(circles)
        
        if n == 0:
            return True

        # Check bounds - vectorized
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        if np.any(x_coords - radii < 0) or np.any(x_coords + radii > self.rect_width) or \
           np.any(y_coords - radii < 0) or np.any(y_coords + radii > self.rect_height):
            return False

        # Use spatial indexing to detect overlaps efficiently
        if len(radii) > 0:
            avg_radius = np.mean(radii)
            # Dynamic cell size based on average radius - more efficient than min radius
            cell_size = max(0.001, avg_radius * 1.2)  # Slightly larger than average radius for efficiency
        else:
            cell_size = 0.01  # Fallback

        # Grid dimensions
        grid_width = int(np.ceil(self.rect_width / cell_size)) + 1
        grid_height = int(np.ceil(self.rect_height / cell_size)) + 1

        # Initialize grid with lists for each cell
        grid = {}

        # Place circles into grid cells
        for i in range(n):
            x, y, r = circles[i]
            # Calculate grid bounds for this circle
            min_cell_x = max(0, int((x - r) / cell_size))
            max_cell_x = min(grid_width - 1, int((x + r) / cell_size))
            min_cell_y = max(0, int((y - r) / cell_size))
            max_cell_y = min(grid_height - 1, int((y + r) / cell_size))

            # Add circle to all relevant grid cells
            for gx in range(min_cell_x, max_cell_x + 1):
                for gy in range(min_cell_y, max_cell_y + 1):
                    if (gx, gy) not in grid:
                        grid[(gx, gy)] = []
                    grid[(gx, gy)].append(i)

        # Check for overlaps within each grid cell and adjacent cells
        for i in range(n):
            x1, y1, r1 = circles[i]
            # Get grid cell coordinates
            cell_x = int(x1 / cell_size)
            cell_y = int(y1 / cell_size)

            # Check nearby cells (3x3 neighborhood)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cell_x + dx, cell_y + dy
                    if (nx, ny) in grid:
                        for j in grid[(nx, ny)]:
                            # Skip self-comparison and ensure we don't double-check
                            if i >= j:
                                continue
                            x2, y2, r2 = circles[j]
                            # Check if circles overlap using squared distances for efficiency
                            dx_squared = (x1 - x2)**2
                            dy_squared = (y1 - y2)**2
                            distance_squared = dx_squared + dy_squared
                            radii_sum = r1 + r2
                            if distance_squared < radii_sum * radii_sum:
                                return False

        return True

    def compute_voronoi_density(self, circles: np.ndarray) -> np.ndarray:
        """Compute Voronoi cell areas for each circle to estimate constraint density."""
        if len(circles) == 0:
            return np.array([])
            
        # Add boundary points for proper Voronoi calculation
        points = circles[:, :2].copy()

        # Add boundary points to make Voronoi more meaningful
        boundary_points = [
            [0, 0], [self.rect_width, 0], [0, self.rect_height], [self.rect_width, self.rect_height],
            [self.rect_width/2, 0], [self.rect_width/2, self.rect_height],
            [0, self.rect_height/2], [self.rect_width, self.rect_height/2]
        ]
        points = np.vstack([points, boundary_points])

        try:
            vor = Voronoi(points)

            # For each original point, compute Voronoi cell area
            areas = []
            for i in range(len(circles)):
                region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1

                if region_idx != -1 and region_idx < len(vor.regions):
                    region = vor.regions[region_idx]
                    if -1 not in region and len(region) >= 3:
                        # Compute area of polygon using shoelace formula
                        vertices = np.array([vor.vertices[i] for i in region])
                        if len(vertices) >= 3:
                            x = vertices[:, 0]
                            y = vertices[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            areas.append(area)
                        else:
                            areas.append(1.0)
                    else:
                        areas.append(1.0)
                else:
                    areas.append(1.0)

            return np.array(areas)
        except:
            # Fallback to uniform distribution if Voronoi fails
            return np.ones(len(circles))

    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate fitness as sum of radii."""
        return np.sum(circles[:, 2])

    def generate_initial_patterns(self) -> List[np.ndarray]:
        """Generate multiple initial patterns for multi-start optimization."""
        patterns = []
        
        # Hexagonal pattern - more compact arrangement
        patterns.append(self._generate_hexagonal_pattern())
        
        # Triangular pattern  
        patterns.append(self._generate_triangular_pattern())
        
        # Square pattern
        patterns.append(self._generate_square_pattern())
        
        # Random pattern with better initialization
        patterns.append(self._generate_improved_random_pattern())
        
        # Dense hexagonal pattern for better packing
        patterns.append(self._generate_dense_hexagonal_pattern())
        
        return patterns

    def _generate_hexagonal_pattern(self) -> np.ndarray:
        """Generate initial hexagonal pattern."""
        circles = np.zeros((self.n_circles, 3))
        
        # Hexagonal packing parameters - more efficient arrangement
        rows = int(np.sqrt(self.n_circles * 1.2))  # Slightly increase rows for better packing
        cols = int(np.ceil(self.n_circles / rows))
        
        spacing_x = self.rect_width / (cols + 0.5)
        spacing_y = self.rect_height / (rows + 0.5)
        
        min_radius = 0.03  # Larger initial radius for better early packing
        
        for i in range(self.n_circles):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * spacing_x
            y = (row + 0.5) * spacing_y
            
            # Offset every other row for hexagonal arrangement
            if row % 2 == 1:
                x += spacing_x / 2
                
            circles[i] = [x, y, min_radius]
            
        return circles

    def _generate_triangular_pattern(self) -> np.ndarray:
        """Generate triangular pattern."""
        circles = np.zeros((self.n_circles, 3))
        
        # Arrange in triangular pattern
        sqrt_n = int(np.ceil(np.sqrt(self.n_circles)))
        spacing_x = self.rect_width / (sqrt_n + 1.5)
        spacing_y = self.rect_height / (sqrt_n + 1.5)
        
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= self.n_circles:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                # Slight offset for triangular pattern
                if i % 2 == 1:
                    x += spacing_x / 2
                circles[idx] = [x, y, 0.03]
                idx += 1
            if idx >= self.n_circles:
                break
                
        return circles

    def _generate_square_pattern(self) -> np.ndarray:
        """Generate square grid pattern."""
        circles = np.zeros((self.n_circles, 3))
        
        sqrt_n = int(np.ceil(np.sqrt(self.n_circles)))
        spacing_x = self.rect_width / (sqrt_n + 1.2)
        spacing_y = self.rect_height / (sqrt_n + 1.2)
        
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= self.n_circles:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                circles[idx] = [x, y, 0.03]
                idx += 1
            if idx >= self.n_circles:
                break
                
        return circles

    def _generate_improved_random_pattern(self) -> np.ndarray:
        """Generate random initial pattern with better distribution."""
        circles = np.zeros((self.n_circles, 3))
        
        # Generate random positions and reasonable initial radii
        for i in range(self.n_circles):
            x = np.random.uniform(0.05, self.rect_width - 0.05)
            y = np.random.uniform(0.05, self.rect_height - 0.05)
            r = np.random.uniform(0.02, 0.06)  # More varied initial radii
            circles[i] = [x, y, r]
            
        return circles

    def _generate_dense_hexagonal_pattern(self) -> np.ndarray:
        """Generate a densely packed hexagonal pattern."""
        circles = np.zeros((self.n_circles, 3))
        
        # Even more compact hexagonal arrangement
        rows = int(np.sqrt(self.n_circles * 1.5))
        cols = int(np.ceil(self.n_circles / rows))
        
        spacing_x = self.rect_width / (cols + 0.3)
        spacing_y = self.rect_height / (rows + 0.3)
        
        min_radius = 0.035
        
        for i in range(self.n_circles):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * spacing_x
            y = (row + 0.5) * spacing_y
            
            # Offset every other row for hexagonal arrangement
            if row % 2 == 1:
                x += spacing_x / 2
                
            circles[i] = [x, y, min_radius]
            
        return circles

class VoronoiAdaptiveMutationEngine:
    """Handles all circle mutation operations with adaptive strategies based on Voronoi analysis."""
    
    @staticmethod
    def mutate_radius(circles: np.ndarray, idx: int, density_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """Mutate radius with adaptive delta based on Voronoi density."""
        new_circles = circles.copy()
        old_r = new_circles[idx, 2]
        
        # Adaptive delta based on density - more sophisticated approach
        max_delta = 0.015  # Increased max delta for more aggressive search
        delta = max_delta
        
        if density_scores is not None and len(density_scores) > idx:
            # High density means more constraints, use smaller deltas but allow larger for exploration
            normalized_density = min(1.0, density_scores[idx] / 15.0)  # Different cap for sensitivity
            # For dense regions: reduce delta significantly; for sparse regions: allow larger changes
            delta_factor = 0.1 + 0.9 * (1.0 - normalized_density)  # Range: 0.1 to 1.0
            delta = max_delta * delta_factor
            
        # Use different probability distributions based on density
        if np.random.random() < 0.6:  # 60% chance of small change
            delta_r = np.random.uniform(-delta*0.3, delta*0.3)
        else:  # 40% chance of medium change
            delta_r = np.random.uniform(-delta, delta)
            
        new_r = old_r + delta_r
        
        # Ensure positive radius
        new_r = max(0.001, new_r)
        new_circles[idx, 2] = new_r
        
        return new_circles

    @staticmethod
    def mutate_position(circles: np.ndarray, idx: int, density_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """Mutate position with small random perturbation, adapted to Voronoi density."""
        new_circles = circles.copy()
        old_x, old_y = new_circles[idx, 0], new_circles[idx, 1]
        
        # Adaptive step size based on density and boundary proximity
        max_delta = 0.06  # Increased delta for better exploration
        delta = max_delta
        
        if density_scores is not None and len(density_scores) > idx:
            # Reduce step size in dense regions more aggressively
            normalized_density = min(1.0, density_scores[idx] / 20.0)  # Different sensitivity
            delta_factor = 0.2 + 0.8 * (1.0 - normalized_density)  # Range: 0.2 to 1.0
            delta = max_delta * delta_factor
            
        # Consider proximity to boundary with stronger factor
        x, y, r = new_circles[idx]
        boundary_distance = min(x, 1.0 - x, y, 1.0 - y)
        boundary_factor = min(1.0, boundary_distance / (r * 1.5))  # Stronger boundary consideration
        boundary_delta = delta * boundary_factor
        
        # Use minimum of adaptive and boundary delta
        actual_delta = min(delta, boundary_delta)
        
        # Use more aggressive perturbation distribution
        if np.random.random() < 0.5:  # 50% chance of larger step
            delta_x = np.random.uniform(-actual_delta*1.5, actual_delta*1.5)
            delta_y = np.random.uniform(-actual_delta*1.5, actual_delta*1.5)
        else:
            delta_x = np.random.uniform(-actual_delta, actual_delta)
            delta_y = np.random.uniform(-actual_delta, actual_delta)
        
        new_x = old_x + delta_x
        new_y = old_y + delta_y
        
        # Ensure within bounds (with some margin)
        margin = 0.01
        new_x = np.clip(new_x, r + margin, 1.0 - r - margin)
        new_y = np.clip(new_y, r + margin, 1.0 - r - margin)
        
        new_circles[idx, 0] = new_x
        new_circles[idx, 1] = new_y
        
        return new_circles

class LocalOptimizer:
    """Performs local optimization with Voronoi-aware strategies."""
    
    def __init__(self, problem: CirclePackingProblem, mutation_engine: VoronoiAdaptiveMutationEngine):
        self.problem = problem
        self.mutation_engine = mutation_engine
    
    def optimize(self, circles: np.ndarray, max_iter: int = 80, patience: int = 15) -> np.ndarray:
        """Perform enhanced local optimization."""
        current_circles = circles.copy()
        best_circles = current_circles.copy()
        best_fitness = self.problem.evaluate_fitness(current_circles)
        
        patience_counter = 0
        last_improvement = 0
        
        # Track improvement history for adaptive behavior
        recent_improvements = []
        
        for iteration in range(max_iter):
            improved = False
            
            # Compute Voronoi densities once per iteration for consistency
            density_scores = self.problem.compute_voronoi_density(current_circles)
            
            # Mutate each circle in a specific order - prioritize critical circles
            # Sort by density (low density = more unconstrained) to mutate those first
            if len(density_scores) > 0:
                sorted_indices = np.argsort(density_scores)
            else:
                sorted_indices = range(len(current_circles))
            
            for i in sorted_indices:
                # Try position mutation with adaptive parameters
                mutated_pos = self.mutation_engine.mutate_position(current_circles, i, density_scores)
                
                # Try radius mutation with adaptive parameters
                mutated_rad = self.mutation_engine.mutate_radius(current_circles, i, density_scores)
                
                # Evaluate both mutations
                pos_fitness = self.problem.evaluate_fitness(mutated_pos)
                rad_fitness = self.problem.evaluate_fitness(mutated_rad)
                
                # Choose the better one
                if pos_fitness > rad_fitness:
                    if self.problem.is_valid_solution(mutated_pos):
                        current_circles = mutated_pos
                        improved = True
                else:
                    if self.problem.is_valid_solution(mutated_rad):
                        current_circles = mutated_rad
                        improved = True
                        
            # Update best solution
            current_fitness = self.problem.evaluate_fitness(current_circles)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_circles = current_circles.copy()
                patience_counter = 0
                last_improvement = iteration
                recent_improvements.append(current_fitness)
            else:
                patience_counter += 1
                
            # Early stopping if no improvement
            if patience_counter >= patience or (iteration - last_improvement > max_iter // 2):
                # Allow for some additional iterations if there were recent improvements
                if len(recent_improvements) > 0 and iteration - max(recent_improvements) < 5:
                    continue
                break
                
        return best_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create problem instance 
    problem = CirclePackingProblem(n_circles=21, rect_width=1.0, rect_height=1.0)
    
    # Create optimizer
    mutation_engine = VoronoiAdaptiveMutationEngine()
    optimizer = LocalOptimizer(problem, mutation_engine)
    
    # Generate initial patterns
    initial_patterns = problem.generate_initial_patterns()
    
    best_solution = None
    best_score = -float('inf')
    
    # Multi-start optimization with different strategies
    for seed_pattern in initial_patterns:
        # Apply local optimization to get better starting points
        optimized_pattern = optimizer.optimize(seed_pattern, max_iter=40)
        
        # Further refine using a few rounds of local search
        final_circles = optimizer.optimize(optimized_pattern, max_iter=30)
        
        score = problem.evaluate_fitness(final_circles)
        if score > best_score and problem.is_valid_solution(final_circles):
            best_score = score
            best_solution = final_circles.copy()
    
    # Final comprehensive fine-tuning with longer search
    if best_solution is not None:
        # Apply extensive local optimization with higher patience
        best_solution = optimizer.optimize(best_solution, max_iter=100, patience=20)
    
    # Ensure final validity
    if best_solution is None:
        # Fallback to random initialization with more iterations
        best_solution = problem._generate_improved_random_pattern()
        best_solution = optimizer.optimize(best_solution, max_iter=120)
        
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")