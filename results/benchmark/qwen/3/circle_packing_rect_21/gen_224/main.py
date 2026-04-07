# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
import math
from typing import Tuple, List, Optional

class CirclePackingConfig:
    """Configuration manager for circle packing parameters"""
    
    def __init__(self):
        self.rect_width = 1.5
        self.rect_height = 0.5
        self.n_circles = 21
        self.seed = 42
        
class CircleValidator:
    """Validates circle configurations and constraints"""
    
    @staticmethod
    def validate_boundaries(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Check if all circles are within rectangle boundaries"""
        return not (np.any(circles[:, 0] - circles[:, 2] < 0) or
                   np.any(circles[:, 0] + circles[:, 2] > rect_width) or
                   np.any(circles[:, 1] - circles[:, 2] < 0) or
                   np.any(circles[:, 1] + circles[:, 2] > rect_height))
    
    @staticmethod
    def validate_overlaps(circles: np.ndarray) -> bool:
        """Vectorized overlap validation"""
        if len(circles) < 2:
            return True
            
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, float('inf'))
        
        min_distances = np.min(dist_matrix, axis=1)
        radii_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        overlap_mask = min_distances < np.min(radii_sums, axis=0)
        return not np.any(overlap_mask)
    
    @staticmethod
    def is_valid_configuration(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Complete validation of circle configuration"""
        return (CircleValidator.validate_boundaries(circles, rect_width, rect_height) and
                CircleValidator.validate_overlaps(circles))

class CircleInitializer:
    """Handles circle initialization strategies"""
    
    @staticmethod
    def create_voronoi_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create initial configuration using Voronoi-based spatial distribution"""
        circles = np.zeros((n_circles, 3))
        
        # Strategic corner and edge positions
        initial_points = [
            (width * 0.1, height * 0.1),    # bottom-left
            (width * 0.9, height * 0.1),    # bottom-right
            (width * 0.1, height * 0.9),    # top-left
            (width * 0.9, height * 0.9),    # top-right
            (width * 0.5, height * 0.1),    # bottom-middle
            (width * 0.5, height * 0.9),    # top-middle
            (width * 0.1, height * 0.5),    # left-middle
            (width * 0.9, height * 0.5),    # right-middle
        ]
        
        # Add random points for coverage
        additional_points = []
        for _ in range(n_circles - len(initial_points)):
            x = random.uniform(0.05 * width, 0.95 * width)
            y = random.uniform(0.05 * height, 0.95 * height)
            additional_points.append([x, y])
        
        all_points = initial_points + additional_points[:n_circles - len(initial_points)]
        
        # Create Voronoi diagram
        try:
            vor = Voronoi(all_points)
            
            # Get valid Voronoi vertices
            valid_vertices = []
            for vertex in vor.vertices:
                if (0 <= vertex[0] <= width) and (0 <= vertex[1] <= height):
                    valid_vertices.append(vertex)
            
            # Use Voronoi vertices as center points
            chosen_centers = valid_vertices[:min(n_circles, len(valid_vertices))]
            if len(chosen_centers) < n_circles:
                # Fill with random points
                for i in range(len(chosen_centers), n_circles):
                    x = random.uniform(0.05 * width, 0.95 * width)
                    y = random.uniform(0.05 * height, 0.95 * height)
                    chosen_centers.append([x, y])
            
            # Assign positions with initial small radii
            for i, (x, y) in enumerate(chosen_centers[:n_circles]):
                circles[i] = [x, y, 0.02]
                
        except Exception:
            # Fallback initialization
            for i in range(n_circles):
                x = random.uniform(0.05 * width, 0.95 * width)
                y = random.uniform(0.05 * height, 0.95 * height)
                circles[i] = [x, y, 0.02]
        
        return circles
    
    @staticmethod
    def create_hexagonal_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create hexagonal grid initialization"""
        circles = np.zeros((n_circles, 3))
        rows = int(math.ceil(math.sqrt(n_circles)))
        cols = int(math.ceil(n_circles / rows))
        cell_width = width / (cols + 1)
        cell_height = height / (rows + 1)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                x_offset = 0.0 if i % 2 == 0 else 0.5
                x = (j + 1 + x_offset) * cell_width
                y = (i + 1) * cell_height
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                circles[idx] = [x, y, 0.02]
                idx += 1
                if idx >= n_circles:
                    break
        return circles
    
    @staticmethod
    def create_random_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create random initialization"""
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            circles[i] = [x, y, 0.02]
        return circles

class CircleRadiusCalculator:
    """Calculates maximum possible radius for a circle at given position"""
    
    @staticmethod
    def compute_max_radius_at_position(x: float, y: float, existing_circles: np.ndarray,
                                     rect_width: float, rect_height: float) -> float:
        """Compute maximum possible radius with early termination"""
        # Distance to boundaries
        min_bound = min(x, rect_width - x, y, rect_height - y)

        # Distance to other circles with early termination
        min_dist = float('inf')
        for i in range(len(existing_circles)):
            ex, ey, er = existing_circles[i]
            dx = x - ex
            dy = y - ey
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0.0001:  # Avoid self-distance
                dist_to_edge = dist - er
                min_dist = min(min_dist, dist_to_edge)
                if min_dist < 0.001:  # Early termination
                    break

        # Return minimum of boundary and other-circle distances
        max_radius = min(min_bound, min_dist if min_dist < float('inf') else float('inf'))
        return max(0.001, max_radius)

class MultiScaleOptimizer:
    """Implements multi-scale optimization with hierarchical refinement"""
    
    def __init__(self, config: CirclePackingConfig):
        self.config = config
    
    def scale_search(self, circles: np.ndarray, scales: List[Tuple[float, int]]) -> np.ndarray:
        """Perform optimization across multiple scales"""
        current = circles.copy()
        
        for scale_step, iterations in scales:
            # Adapt step size based on current configuration
            for _ in range(iterations):
                # Process circles in random order for better exploration
                indices = list(range(len(current)))
                random.shuffle(indices)
                
                for i in indices:
                    original_x, original_y, original_r = current[i]
                    best_x, best_y, best_r = original_x, original_y, original_r
                    best_sum = np.sum(current[:, 2])  # Current sum
                    
                    # Generate moves for this scale
                    moves = self._generate_scaled_moves(scale_step, original_x, original_y)
                    
                    # Try each move
                    for dx, dy in moves:
                        test_x = max(0.001, min(self.config.rect_width - 0.001, original_x + dx))
                        test_y = max(0.001, min(self.config.rect_height - 0.001, original_y + dy))
                        
                        # Compute max radius at new position
                        temp_circles = current.copy()
                        temp_circles[i] = [test_x, test_y, 0.01]
                        max_r = CircleRadiusCalculator.compute_max_radius_at_position(
                            test_x, test_y, temp_circles, self.config.rect_width, self.config.rect_height)
                        test_r = min(max_r, max(0.001, original_r + random.uniform(-0.03, 0.03)))
                        
                        # Apply adjustment
                        temp_circles[i] = [test_x, test_y, test_r]
                        
                        # Validate and evaluate
                        if CircleValidator.is_valid_configuration(temp_circles, 
                                                                self.config.rect_width, 
                                                                self.config.rect_height):
                            new_sum = np.sum(temp_circles[:, 2])
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_x, best_y, best_r = test_x, test_y, test_r
                    
                    # Update if improvement found
                    if best_sum > np.sum(current[:, 2]):
                        current[i] = [best_x, best_y, best_r]
        
        return current
    
    def _generate_scaled_moves(self, scale_step: float, x: float, y: float) -> List[Tuple[float, float]]:
        """Generate move candidates based on scale"""
        moves = []
        
        # Base moves with scale-dependent step sizes
        step_sizes = [scale_step * 0.5, scale_step, scale_step * 2]
        
        for step in step_sizes:
            moves.extend([
                (step * random.gauss(0, 1), step * random.gauss(0, 1)),
                (step * random.uniform(-1, 1), 0),
                (0, step * random.uniform(-1, 1)),
                (step * random.choice([-1, 1]), 0),
                (0, step * random.choice([-1, 1])),
                (random.uniform(-step/2, step/2), random.uniform(-step/2, step/2)),
                (0, 0)
            ])
        
        return moves

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Configuration setup
    config = CirclePackingConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize components
    validator = CircleValidator()
    initializer = CircleInitializer()
    optimizer = MultiScaleOptimizer(config)
    
    def calculate_radius_sum(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])
    
    def multi_stage_optimization() -> np.ndarray:
        """Multi-stage optimization with progressive refinement"""
        best_circles = None
        best_sum = -float('inf')
        
        # Different initialization strategies
        init_strategies = [
            ('voronoi', lambda: initializer.create_voronoi_initialization(
                config.n_circles, config.rect_width, config.rect_height)),
            ('hexagonal', lambda: initializer.create_hexagonal_initialization(
                config.n_circles, config.rect_width, config.rect_height)),
            ('random', lambda: initializer.create_random_initialization(
                config.n_circles, config.rect_width, config.rect_height))
        ]
        
        # Run multiple optimization starts
        for start_num in range(6):
            # Select initialization strategy
            strategy_name, strategy_func = init_strategies[start_num % len(init_strategies)]
            
            # Create initial circles
            circles = strategy_func()
            
            # Scale-based refinement in stages
            scales = [
                (0.2, 30),   # Coarse scale
                (0.1, 50),   # Medium scale
                (0.05, 70),  # Fine scale
                (0.02, 100)  # Ultra-fine scale
            ]
            
            refined_circles = optimizer.scale_search(circles, scales)
            
            final_sum = calculate_radius_sum(refined_circles)
            
            if final_sum > best_sum:
                best_sum = final_sum
                best_circles = refined_circles.copy()
        
        return best_circles
    
    # Main optimization workflow
    final_circles = multi_stage_optimization()
    
    # Final validation and cleanup
    if final_circles is not None:
        # Ensure final configuration is valid
        max_attempts = 5
        attempts = 0
        
        while not validator.is_valid_configuration(final_circles, 
                                                 config.rect_width, 
                                                 config.rect_height) and attempts < max_attempts:
            # Re-initialize if invalid
            final_circles = initializer.create_voronoi_initialization(
                config.n_circles, config.rect_width, config.rect_height)
            attempts += 1
        
        # Final refinement if needed
        if attempts < max_attempts:
            scales = [(0.01, 50)]  # Fine tuning
            final_circles = optimizer.scale_search(final_circles, scales)
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")