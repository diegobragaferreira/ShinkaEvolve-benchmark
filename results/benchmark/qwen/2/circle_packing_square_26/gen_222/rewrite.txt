# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, Delaunay
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import distance
import random
import time
from typing import Tuple, List, Optional

# Global constants
SEED = 42
MAX_ITER = 1000
TOLERANCE = 1e-6

random.seed(SEED)
np.random.seed(SEED)

class VoronoiBasedOptimizer:
    def __init__(self):
        self.n_circles = 26
        
    def generate_voronoi_initialization(self) -> np.ndarray:
        """Generate initial configuration using Voronoi-based approach with strategic point placement."""
        # Generate points using hexagonal grid pattern for good initial distribution
        points = self._generate_hexagonal_grid(self.n_circles + 20)
        
        # Create Voronoi diagram
        try:
            vor = Voronoi(points)
        except:
            # Fallback to simple initialization
            return self._create_simple_initialization()
        
        # Select valid points for circle centers
        circles = np.zeros((self.n_circles, 3))
        
        # Filter points that are inside the unit square
        valid_indices = []
        for i, point in enumerate(vor.points):
            if 0 <= point[0] <= 1 and 0 <= point[1] <= 1:
                valid_indices.append(i)
                
        # Take first n_circles valid points, or fallback to grid
        if len(valid_indices) >= self.n_circles:
            selected_indices = valid_indices[:self.n_circles]
        else:
            # Fallback to grid if not enough valid points
            return self._create_simple_initialization()
        
        # Estimate initial radii based on Voronoi cell areas and boundary constraints
        for i, idx in enumerate(selected_indices):
            center = vor.points[idx]
            x, y = center
            
            # Calculate minimum distance to nearest Voronoi neighbors
            min_neighbor_dist = float('inf')
            for j, other_point in enumerate(vor.points):
                if i != j:
                    dist = np.sqrt((center[0] - other_point[0])**2 + (center[1] - other_point[1])**2)
                    min_neighbor_dist = min(min_neighbor_dist, dist)
            
            # Estimate radius based on surrounding points and boundary constraints
            if min_neighbor_dist != float('inf'):
                estimated_radius = min_neighbor_dist / 4.0
            else:
                estimated_radius = 0.1
                
            # Respect boundary constraints
            min_dist_to_boundary = min(x, y, 1-x, 1-y)
            final_radius = min(estimated_radius, min_dist_to_boundary * 0.8)
            final_radius = max(0.01, min(final_radius, 0.2))  # Reasonable bounds
            
            circles[i] = [x, y, final_radius]
        
        return circles
    
    def _generate_hexagonal_grid(self, n_points: int) -> np.ndarray:
        """Generate hexagonal grid points."""
        points = []
        rows = int(np.ceil(np.sqrt(n_points)))
        cols = int(np.ceil(n_points / rows))
        
        # Hexagonal spacing 
        spacing = 1.0 / (max(rows, cols) + 2)
        hex_height = spacing * np.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = (j + 0.5 + (i % 2) * 0.5) * spacing
                y = (i + 0.5) * hex_height
                if x <= 1 and y <= 1:
                    points.append([x, y])
        
        # Trim to exact number needed and add jitter
        points = points[:n_points]
        for point in points:
            point[0] += np.random.uniform(-spacing/8, spacing/8)
            point[1] += np.random.uniform(-spacing/8, spacing/8)
        
        # Ensure bounds
        points = [[max(0.01, min(0.99, p[0])), max(0.01, min(0.99, p[1]))] for p in points]
        return np.array(points)
    
    def _create_simple_initialization(self) -> np.ndarray:
        """Create a simple but valid initial configuration."""
        circles = np.zeros((self.n_circles, 3))
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing = 1.0 / (grid_size + 1)
        
        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if idx >= self.n_circles:
                    break
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 4
                circles[idx] = [x, y, r]
                idx += 1
        return circles
    
    def validate_constraints(self, circles: np.ndarray) -> bool:
        """Check if all circles satisfy containment and overlap constraints."""
        if len(circles) != self.n_circles:
            return False

        # Check containment constraints
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

        # Check overlap constraints with optimized distance calculation
        if self.n_circles > 1:
            distances = cdist(circles[:, :2], circles[:, :2])
            mask = np.triu(np.ones((self.n_circles, self.n_circles), dtype=bool), k=1)
            min_distances = (circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]) * mask
            overlaps = distances < min_distances
            if np.any(overlaps):
                return False
                
        return True
    
    def calculate_sum_radii(self, circles: np.ndarray) -> float:
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])
    
    def solve_circle_packing(self, initial_config: np.ndarray) -> np.ndarray:
        """Main optimization procedure using Voronoi-Delaunay approach."""
        # Start with initial configuration
        circles = initial_config.copy()
        
        # Optimization using constrained minimization on negative sum of radii
        def objective(params):
            # Convert params back to circles array
            temp_circles = circles.copy()
            for i in range(self.n_circles):
                temp_circles[i, 0] = params[i*3]
                temp_circles[i, 1] = params[i*3 + 1]
                temp_circles[i, 2] = params[i*3 + 2]
            
            # If invalid, penalize heavily
            if not self.validate_constraints(temp_circles):
                return 1000000.0
                
            # Return negative sum (since we're minimizing)
            return -self.calculate_sum_radii(temp_circles)
        
        def constraint_func(params):
            # Convert params back to circles array
            temp_circles = circles.copy()
            for i in range(self.n_circles):
                temp_circles[i, 0] = params[i*3]
                temp_circles[i, 1] = params[i*3 + 1]
                temp_circles[i, 2] = params[i*3 + 2]
            
            # Constraint violation measure
            if not self.validate_constraints(temp_circles):
                return -1000000.0  # Invalid configuration
            
            # Return positive value if constraint satisfied
            return self.calculate_sum_radii(temp_circles)
            
        # Prepare initial parameters 
        initial_params = []
        for i in range(self.n_circles):
            initial_params.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
        
        # Define bounds for [x, y, r] for each circle
        bounds = []
        for i in range(self.n_circles):
            # x bounds: [r, 1-r]
            bounds.append((circles[i, 2], 1 - circles[i, 2]))
            # y bounds: [r, 1-r]  
            bounds.append((circles[i, 2], 1 - circles[i, 2]))
            # r bounds: [0.01, 0.5] (reasonable bounds)
            bounds.append((0.01, 0.5))
        
        # Constraint function: must have positive value for valid solutions
        cons = {'type': 'ineq', 'fun': lambda x: constraint_func(x)}
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(objective, 
                            initial_params,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': MAX_ITER, 'ftol': TOLERANCE},
                            callback=None)
            
            # Extract results
            if result.success:
                # Convert back to circles format
                optimized_circles = circles.copy()
                for i in range(self.n_circles):
                    optimized_circles[i, 0] = result.x[i*3]
                    optimized_circles[i, 1] = result.x[i*3 + 1]
                    optimized_circles[i, 2] = result.x[i*3 + 2]
                
                # Final validation
                if self.validate_constraints(optimized_circles):
                    return optimized_circles
                else:
                    # Fallback to local search if optimization didn't work
                    return self._local_search_optimization(circles)
            else:
                # Optimization failed, use fallback
                return self._local_search_optimization(circles)
                
        except Exception:
            # If optimization fails, use local search
            return self._local_search_optimization(circles)
    
    def _local_search_optimization(self, circles: np.ndarray) -> np.ndarray:
        """Fallback local search optimization to refine solution."""
        current_circles = circles.copy()
        best_circles = current_circles.copy()
        best_sum = self.calculate_sum_radii(current_circles)
        
        # Simulated annealing style local search
        max_iter = 500
        temperature = 1.0
        cooling_rate = 0.999
        
        for i in range(max_iter):
            # Create candidate solution by perturbing one circle
            candidate = current_circles.copy()
            
            # Choose a random circle to modify
            circle_idx = random.randint(0, self.n_circles - 1)
            
            # Perturb position and radius
            candidate[circle_idx, 0] += np.random.normal(0, 0.01)
            candidate[circle_idx, 1] += np.random.normal(0, 0.01)
            candidate[circle_idx, 2] += np.random.normal(0, 0.005)
            
            # Apply boundary constraints
            candidate[circle_idx, 0] = np.clip(candidate[circle_idx, 0], 
                                              candidate[circle_idx, 2], 
                                              1 - candidate[circle_idx, 2])
            candidate[circle_idx, 1] = np.clip(candidate[circle_idx, 1], 
                                              candidate[circle_idx, 2], 
                                              1 - candidate[circle_idx, 2])
            candidate[circle_idx, 2] = np.clip(candidate[circle_idx, 2], 0.01, 0.5)
            
            # Validate new solution
            if self.validate_constraints(candidate):
                candidate_sum = self.calculate_sum_radii(candidate)
                if candidate_sum > best_sum:
                    best_circles = candidate.copy()
                    best_sum = candidate_sum
                elif random.random() < np.exp(-(best_sum - candidate_sum) / (temperature + 1e-10)):
                    # Accept worse solution with probability
                    current_circles = candidate.copy()
            
            temperature *= cooling_rate
            
        return best_circles
    
    def run_optimization(self) -> np.ndarray:
        """Run complete optimization pipeline."""
        # Step 1: Generate initial Voronoi-based configuration
        initial_config = self.generate_voronoi_initialization()
        
        # Step 2: Optimize using gradient-based approach
        optimized_result = self.solve_circle_packing(initial_config)
        
        # Step 3: Final refinement using local search
        final_result = self._local_search_optimization(optimized_result)
        
        return final_result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = VoronoiBasedOptimizer()
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END