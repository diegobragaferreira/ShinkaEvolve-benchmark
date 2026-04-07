# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import time
from typing import Tuple, List, Optional

class CirclePackingOptimizer:
    """Multi-phase optimizer for 21 circle packing in a rectangle"""
    
    def __init__(self, width: float = 1.2, height: float = 0.8, seed: int = 42):
        self.width = width
        self.height = height
        self.seed = seed
        np.random.seed(seed)
        
    def _calculate_max_radius_fast(self, circles: np.ndarray, index: int) -> float:
        """Fast calculation of maximum radius for circle at given index without overlapping others."""
        x, y, current_radius = circles[index]
        
        # Maximum radius based on container boundaries
        max_radius_bound = min(x, y, self.width - x, self.height - y)
        
        # Vectorized overlap checking for efficiency
        if len(circles) > 1:
            # Get other circles' positions and radii
            other_positions = circles[[i for i in range(len(circles)) if i != index], :2]
            other_radii = circles[[i for i in range(len(circles)) if i != index], 2]
            
            # Calculate distances to all other circles
            distances = np.sqrt(np.sum((other_positions - [x, y])**2, axis=1))
            
            # Maximum radius that avoids overlap with all other circles
            max_radius_overlap = np.min(distances - other_radii)
            
            max_radius = min(max_radius_bound, max_radius_overlap)
        else:
            max_radius = max_radius_bound
            
        return max(max_radius, 0.001)
    
    def _calculate_max_radius_at_position_fast(self, circles: np.ndarray, index: int, x: float, y: float) -> float:
        """Fast calculation of maximum radius for circle at given position without overlapping others."""
        # Maximum radius based on container boundaries
        max_radius_bound = min(x, y, self.width - x, self.height - y)
        
        # Vectorized overlap checking for efficiency
        if len(circles) > 1:
            # Get other circles' positions and radii
            other_positions = circles[[i for i in range(len(circles)) if i != index], :2]
            other_radii = circles[[i for i in range(len(circles)) if i != index], 2]
            
            # Calculate distances to all other circles
            distances = np.sqrt(np.sum((other_positions - [x, y])**2, axis=1))
            
            # Maximum radius that avoids overlap with all other circles
            max_radius_overlap = np.min(distances - other_radii)
            
            max_radius = min(max_radius_bound, max_radius_overlap)
        else:
            max_radius = max_radius_bound
            
        return max(max_radius, 0.001)
    
    def _initialize_with_strategic_placement(self, n_circles: int) -> np.ndarray:
        """Initialize circles using strategic placement with corner/edge points and hexagonal grid"""
        # Start with corner points for boundary coverage
        corner_points = [
            [0.1, 0.1],           # Bottom-left
            [self.width-0.1, 0.1],     # Bottom-right
            [0.1, self.height-0.1],    # Top-left
            [self.width-0.1, self.height-0.1], # Top-right
        ]
        
        # Add edge midpoints for better boundary coverage
        edge_points = [
            [self.width/2, 0.1],       # Bottom-middle
            [self.width/2, self.height-0.1], # Top-middle
            [0.1, self.height/2],      # Left-middle
            [self.width-0.1, self.height/2], # Right-middle
        ]
        
        # Combine corner and edge points
        init_points = corner_points + edge_points
        
        # Fill remaining slots with hexagonal grid pattern for interior coverage
        rows = 5
        cols = 5
        estimated_radius = 0.08  # Adjusted for container size
        
        # Generate hexagonal grid points
        hex_points = []
        for i in range(rows):
            for j in range(cols):
                if len(hex_points) >= (n_circles - len(init_points)):
                    break
                x = j * 2 * estimated_radius + (i % 2) * estimated_radius  # Offset every other row
                y = i * np.sqrt(3) * estimated_radius
                
                # Add point if it's within bounds
                if 0.01 <= x <= self.width - 0.01 and 0.01 <= y <= self.height - 0.01:
                    hex_points.append([x, y])
            if len(hex_points) >= (n_circles - len(init_points)):
                break

        # Combine all initial points
        all_init_points = init_points + hex_points[:n_circles-len(init_points)]
        
        # Add any remaining points randomly if needed
        while len(all_init_points) < n_circles:
            x = np.random.uniform(0.05, self.width - 0.05)
            y = np.random.uniform(0.05, self.height - 0.05)
            all_init_points.append([x, y])
        
        # Initialize with small radii
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            circles[i] = [all_init_points[i][0], all_init_points[i][1], 0.01]

        return circles
    
    def _adaptive_optimization_phase(self, circles: np.ndarray) -> np.ndarray:
        """Phase 2: Multi-scale adaptive optimization with momentum"""
        max_iterations = 1500
        best_sum = np.sum(circles[:, 2])
        best_circles = circles.copy()
        
        # Track improvement for early termination
        last_improvement_iter = 0
        improvement_count = 0

        for iteration in range(max_iterations):
            improved = False
            
            # Adaptive parameter adjustment based on iteration
            if iteration < 500:
                # Phase 1: Broad exploration with large steps
                step_size = 0.15
                radius_update_factor = 0.95
            elif iteration < 1000:
                # Phase 2: Focused refinement with medium steps
                step_size = 0.08
                radius_update_factor = 0.8
            else:
                # Phase 3: Fine-tuning with small steps
                step_size = 0.03
                radius_update_factor = 0.6

            # Try to increase each circle's radius
            for i in range(len(circles)):
                # Find maximum possible radius for circle i
                max_radius = self._calculate_max_radius_fast(circles, i)
                
                if max_radius > circles[i][2]:
                    # Apply adaptive radius update with momentum
                    new_radius = min(max_radius, circles[i][2] * radius_update_factor + 0.005)
                    circles[i][2] = new_radius
                    improved = True

            # Track improvement for early stopping
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
                improvement_count = 0
                last_improvement_iter = iteration
            else:
                improvement_count += 1

            # Early termination for stagnation
            if improvement_count > 100 and iteration > 500:
                break

        return best_circles
    
    def _local_search_phase(self, circles: np.ndarray) -> np.ndarray:
        """Phase 3: Progressive local search with adaptive neighborhood"""
        local_search_iterations = 1000
        last_improvement = 0
        tolerance = 1e-6

        for refinement_iteration in range(local_search_iterations):
            # Adaptive step size reduction
            if refinement_iteration < 300:
                step_size = 0.1
            elif refinement_iteration < 600:
                step_size = 0.05
            else:
                step_size = 0.02

            # Try moving each circle slightly to see if we can improve the configuration
            for i in range(len(circles)):
                current_x, current_y, current_r = circles[i]

                # Track best improvement for this iteration
                best_pos = [current_x, current_y, current_r]
                best_radius = current_r

                # Examine grid around current position with adaptive step size
                if refinement_iteration < 200:
                    # Coarse search in early stages
                    search_grid = [-step_size*2, -step_size, 0, step_size, step_size*2]
                elif refinement_iteration < 400:
                    # Medium search in middle stages
                    search_grid = [-step_size, 0, step_size]
                else:
                    # Fine search in later stages
                    search_grid = [-step_size/2, 0, step_size/2]

                # Also add diagonal searches for better exploration in later stages
                if refinement_iteration > 400:
                    search_grid.extend([-step_size*1.5, step_size*1.5])

                # Examine grid around current position
                for dx in search_grid:
                    for dy in search_grid:
                        new_x, new_y = current_x + dx, current_y + dy

                        # Check if new position is within bounds
                        if 0 <= new_x <= self.width and 0 <= new_y <= self.height:
                            # Calculate max radius at new position
                            max_radius = self._calculate_max_radius_at_position_fast(
                                circles, i, new_x, new_y
                            )

                            if max_radius > best_radius:
                                best_radius = max_radius
                                best_pos = [new_x, new_y, max_radius]

                # Update if we found a better position
                if best_pos[2] > circles[i][2]:
                    circles[i] = best_pos

            # Check for convergence
            new_sum = np.sum(circles[:, 2])
            if new_sum > np.sum(circles[:, 2]):  # Only update if there was an improvement
                last_improvement = refinement_iteration
            elif refinement_iteration - last_improvement > 50:
                break

        return circles
    
    def optimize(self) -> np.ndarray:
        """Main optimization driver"""
        # Phase 1: Strategic initialization for better spatial distribution
        n_circles = 21
        circles = self._initialize_with_strategic_placement(n_circles)

        # Phase 2: Adaptive optimization
        circles = self._adaptive_optimization_phase(circles)

        # Phase 3: Local search refinement
        circles = self._local_search_phase(circles)

        # Final validation and cleanup
        for i in range(n_circles):
            # Ensure minimum radius
            circles[i][2] = max(circles[i][2], 0.001)
            
            # Ensure circles stay within bounds
            circles[i][0] = np.clip(circles[i][0], 0.001, self.width - 0.001)
            circles[i][1] = np.clip(circles[i][1], 0.001, self.height - 0.001)

        return circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer()
    return optimizer.optimize()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")