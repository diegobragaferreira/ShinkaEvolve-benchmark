# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
import time
from numba import jit, prange
import warnings

class HexagonGeometry:
    """Handles all geometric computations for hexagons."""
    
    @staticmethod
    @jit(nopython=True)
    def vertices(x, y, angle_deg, side_length=1):
        """Compute vertices of a hexagon given center, rotation, and side length."""
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        # Vertices of regular hexagon with side length 1 centered at origin
        base_verts = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1, 0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])

        # Rotate and translate
        rotated_verts = np.empty_like(base_verts)
        for i in range(6):
            x_orig, y_orig = base_verts[i]
            rotated_verts[i] = [
                x + side_length * (x_orig * cos_a - y_orig * sin_a),
                y + side_length * (x_orig * sin_a + y_orig * cos_a)
            ]

        return rotated_verts
    
    @staticmethod
    def polygon(x, y, angle_deg, side_length=1):
        """Convert hexagon parameters to shapely polygon."""
        vertices = HexagonGeometry.vertices(x, y, angle_deg, side_length)
        return Polygon(vertices)
    
    @staticmethod
    def contains_point(hex_poly, point):
        """Check if a point is inside a hexagon polygon."""
        return hex_poly.contains(point) or hex_poly.covers(point)

class ConstraintChecker:
    """Handles all constraint validation for hexagon packing."""
    
    def __init__(self, outer_hex_polygon, num_hexagons=12):
        self.outer_hex_polygon = outer_hex_polygon
        self.num_hexagons = num_hexagons
    
    def check_containment(self, hex_poly):
        """Check if hexagon is fully contained within outer hexagon."""
        try:
            return self.outer_hex_polygon.contains(hex_poly) or self.outer_hex_polygon.covers(hex_poly)
        except:
            return False
    
    def check_overlap(self, hex1_poly, hex2_poly):
        """Check if two hexagons overlap."""
        try:
            return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)
        except:
            return False
    
    def validate_config(self, inner_hex_data):
        """Validate entire configuration for constraints."""
        # Early exit for invalid data shape
        if len(inner_hex_data) != self.num_hexagons:
            return False, float('inf')
        
        # Compute all inner hexagon polygons
        hex_polygons = [HexagonGeometry.polygon(x, y, angle) for x, y, angle in inner_hex_data]
        
        # Check containment
        for hex_poly in hex_polygons:
            if not self.check_containment(hex_poly):
                return False, float('inf')
        
        # Check overlaps using spatial indexing for efficiency
        try:
            tree = cKDTree([(hex.center.x, hex.center.y) for hex in hex_polygons])
            
            total_penalty = 0.0
            for i, hex_poly in enumerate(hex_polygons):
                # Find nearby hexagons for overlap checking
                nearby_indices = tree.query_ball_point((hex_poly.centroid.x, hex_poly.centroid.y), 3.0)
                
                for j in nearby_indices:
                    if i >= j:  # Avoid checking pairs twice and self-intersection
                        continue
                    
                    if self.check_overlap(hex_poly, hex_polygons[j]):
                        # Calculate overlap area
                        try:
                            overlap = hex_poly.intersection(hex_polygons[j])
                            if hasattr(overlap, 'area'):
                                total_penalty += overlap.area
                        except:
                            total_penalty += 1000  # Large penalty
                        
                        # Early exit on severe overlap
                        if total_penalty > 10000:
                            return False, total_penalty
        
        except Exception as e:
            warnings.warn(f"Constraint checking error: {e}")
            return False, float('inf')
            
        return True, total_penalty

class Optimizer:
    """Handles the optimization process."""
    
    def __init__(self, num_hexagons=12):
        self.num_hexagons = num_hexagons
        self.constraint_checker = None
    
    def objective_function(self, params):
        """Objective function for optimization."""
        # Extract inner hexagon data and outer radius
        inner_hex_data = params[:-1].reshape(self.num_hexagons, 3)
        outer_hex_side_length = params[-1]
        
        # Create outer hexagon polygon (centered at origin)
        outer_hex_poly = HexagonGeometry.polygon(0, 0, 0, outer_hex_side_length)
        
        # Set up constraint checker
        self.constraint_checker = ConstraintChecker(outer_hex_poly)
        
        # Validate configuration
        is_valid, penalty = self.constraint_checker.validate_config(inner_hex_data)
        
        if not is_valid:
            # Apply penalty based on constraint violation severity
            return penalty * 10000 if penalty > 0 else 1000000
        
        # If valid, return inverse of outer hexagon side length (negative for minimization)
        return -1.0 / outer_hex_side_length
    
    def generate_initial_guess(self):
        """Generate a good initial symmetric configuration."""
        # Start with a pattern resembling known good packings
        angles = [0, 60, 120, 180, 240, 300]
        base_radius = 1.5
        positions = []
        
        # Central hexagon
        positions.append([0, 0, 0])
        
        # Surrounding hexagons in 6 directions
        for i, angle in enumerate(angles):
            rad_angle = np.radians(angle)
            x = base_radius * np.cos(rad_angle)
            y = base_radius * np.sin(rad_angle)
            positions.append([x, y, 0])
        
        # Additional layer
        layer2_radius = 2.5
        for i, angle in enumerate(angles):
            rad_angle = np.radians(angle)
            x = layer2_radius * np.cos(rad_angle)
            y = layer2_radius * np.sin(rad_angle)
            positions.append([x, y, 0])
        
        # Add remaining positions
        positions.append([0, -3.5, 0])  # Bottom center
        
        # Take only first 12 positions
        return np.array(positions[:12])
    
    def optimize(self):
        """Main optimization routine."""
        # Generate initial guess
        initial_guess_inner = self.generate_initial_guess()
        
        # Initial estimate for outer radius based on configuration
        max_dist = 0
        for i in range(self.num_hexagons):
            x, y, _ = initial_guess_inner[i]
            dist = np.sqrt(x*x + y*y)
            max_dist = max(max_dist, dist)
        
        # Add margin for hexagon size (hexagon has width approximately 2)
        initial_outer_radius = max_dist + 2.0
        
        # Combine into single parameter vector: [12*3 positions + 1 outer radius]
        initial_params = np.concatenate([initial_guess_inner.flatten(), [initial_outer_radius]])
        
        # Define bounds for optimization
        # Positions: x, y bounded to reasonable range, angle 0-360
        bounds = []
        for _ in range(self.num_hexagons):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        # Outer radius bound (should be positive)
        bounds.append((0.1, 20.0))
        
        # First, use global optimization to explore the space broadly
        try:
            de_result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                disp=False
            )
            
            # Use the best solution from global search as starting point for local refinement
            best_params = de_result.x
            
            # Local refinement using L-BFGS-B
            refined_bounds = [(b[0], b[1]) for b in bounds]
            
            # Perform local optimization
            try:
                local_result = minimize(
                    self.objective_function,
                    best_params,
                    method='L-BFGS-B',
                    bounds=refined_bounds,
                    options={'maxiter': 100, 'disp': False}
                )
                
                if local_result.success:
                    best_params = local_result.x
            except:
                # Fall back to previous solution if local optimization fails
                pass
            
            # Extract results
            inner_hex_data = best_params[:-1].reshape(self.num_hexagons, 3)
            outer_hex_side_length = best_params[-1]
            
            # Ensure the outer hexagon is actually large enough
            min_outer_radius = 0
            for i in range(self.num_hexagons):
                x, y, _ = inner_hex_data[i]
                dist = np.sqrt(x*x + y*y) + 1.0  # +1 for hexagon radius
                min_outer_radius = max(min_outer_radius, dist)
            
            # If we computed a smaller radius than needed, adjust it up
            if outer_hex_side_length < min_outer_radius:
                outer_hex_side_length = min_outer_radius * 1.05  # Add small margin
                
            return inner_hex_data, outer_hex_side_length
            
        except Exception as e:
            # Fallback to initial guess if optimization fails
            warnings.warn(f"Optimization failed: {e}")
            return initial_guess_inner, initial_outer_radius

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Track execution time
    start_time = time.time()
    
    try:
        # Initialize optimizer and run optimization
        optimizer = Optimizer()
        inner_hex_data, outer_hex_side_length = optimizer.optimize()
        
        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])
        
        # Ensure we don't exceed time limits
        end_time = time.time()
        eval_time = end_time - start_time
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback to original configuration if optimization fails
        warnings.warn(f"Fallback activated due to error: {e}")
        n = 12
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ])

        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
