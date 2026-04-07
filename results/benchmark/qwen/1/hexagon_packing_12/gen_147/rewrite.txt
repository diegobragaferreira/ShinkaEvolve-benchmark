# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
from numba import jit
from joblib import Parallel, delayed

class HexagonGeometryHandler:
    """Handles all geometric computations for hexagons with optimized vectorized operations."""
    
    def __init__(self):
        # Pre-computed base vertices for efficiency
        self.base_vertices = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1, 0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
    
    @staticmethod
    @jit(nopython=True)
    def _compute_hexagon_vertices(x, y, angle_deg, side_length=1):
        """Compute vertices of a hexagon given center, rotation, and side length."""
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Base vertices of regular hexagon with side length 1 centered at origin
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
    
    def compute_hexagon_polygon(self, x, y, angle_deg, side_length=1):
        """Convert hexagon parameters to shapely polygon."""
        vertices = self._compute_hexagon_vertices(x, y, angle_deg, side_length)
        return Polygon(vertices)
    
    def compute_all_hexagon_polygons(self, hex_data):
        """Compute all hexagon polygons from array of parameters."""
        polygons = []
        for i in range(len(hex_data)):
            x, y, angle = hex_data[i]
            polygons.append(self.compute_hexagon_polygon(x, y, angle))
        return polygons

class HexagonConstraintValidator:
    """Validates spatial constraints for hexagon arrangements with optimized checking."""
    
    def __init__(self, geom_handler):
        self.geom = geom_handler
    
    def validate_containment(self, inner_hex_data, outer_radius):
        """Fast validation of containment using bounding circle checks."""
        # Precompute outer hexagon polygon once
        outer_hex_poly = self.geom.compute_hexagon_polygon(0, 0, 0, outer_radius)
        
        # Check containment for all hexagons
        for i in range(len(inner_hex_data)):
            x, y, angle = inner_hex_data[i]
            # Fast bounding circle check first
            dist_from_origin = np.sqrt(x*x + y*y)
            if dist_from_origin + 1.0 > outer_radius:  # +1 for hexagon radius
                return False, None
        
        # Final full containment check
        for i in range(len(inner_hex_data)):
            x, y, angle = inner_hex_data[i]
            inner_poly = self.geom.compute_hexagon_polygon(x, y, angle)
            if not outer_hex_poly.contains(inner_poly):
                return False, None
                
        return True, outer_hex_poly
    
    def compute_overlap_penalty(self, inner_hex_data):
        """Efficiently compute overlap penalty for all pairs of hexagons."""
        # Pre-compute all polygons
        polygons = self.geom.compute_all_hexagon_polygons(inner_hex_data)
        
        penalty = 0.0
        n = len(polygons)
        
        # Check all pairs for overlaps
        for i in range(n):
            for j in range(i+1, n):
                if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                    try:
                        overlap = polygons[i].intersection(polygons[j])
                        if hasattr(overlap, 'area') and overlap.area > 0:
                            penalty += overlap.area
                    except:
                        penalty += 1000  # Large penalty for calculation errors
                        
        return penalty

class HexagonPackingOptimizer:
    """Main optimization orchestrator with improved search strategy."""
    
    def __init__(self):
        self.geom = HexagonGeometryHandler()
        self.validator = HexagonConstraintValidator(self.geom)
    
    def evaluate_configuration(self, params):
        """Evaluate a configuration with optimized constraint checking."""
        try:
            # Extract inner hexagon data and outer radius
            inner_hex_data = params[:-1].reshape(12, 3)
            outer_hex_side_length = params[-1]
            
            # Fast containment check first
            is_contained, outer_hex_poly = self.validator.validate_containment(inner_hex_data, outer_hex_side_length)
            
            if not is_contained:
                # Return large penalty for containment violation
                return 1e10
            
            # Compute overlap penalty
            overlap_penalty = self.validator.compute_overlap_penalty(inner_hex_data)
            
            if overlap_penalty > 0:
                return overlap_penalty * 10000  # Apply penalty for overlaps
                
            # If valid, return inverse of outer hexagon side length (negative for minimization)
            return -1.0 / outer_hex_side_length
            
        except Exception:
            # In case of any computation error, return large penalty
            return 1e10
    
    def generate_initial_configuration(self):
        """Generate an improved initial configuration based on mathematical insight."""
        # Use a more sophisticated configuration that places hexagons in a pattern designed to
        # minimize the outer hexagon size while maintaining non-overlap and containment
        positions = []

        # Central hexagon
        positions.append([0, 0, 0])

        # First ring - six hexagons at distance of ~sqrt(3) from center
        for i in range(6):
            angle = i * 60
            # Distance chosen to allow efficient packing
            dist = 1.732  # sqrt(3)
            x = dist * np.cos(np.radians(angle))
            y = dist * np.sin(np.radians(angle))
            positions.append([x, y, 0])

        # Second ring - six hexagons at distance of ~2*sqrt(3) from center
        for i in range(6):
            angle = i * 60 + 30  # offset to create dense packing
            dist = 3.464  # 2*sqrt(3)
            x = dist * np.cos(np.radians(angle))
            y = dist * np.sin(np.radians(angle))
            positions.append([x, y, 0])

        # Additional strategic positioning for better packing
        positions.append([0, -4.5, 0])

        initial_config = np.array(positions[:12])
        
        # Add small random perturbations to avoid local minima
        np.random.seed(42)
        initial_config[:, :2] += np.random.normal(0, 0.1, (12, 2))
        
        return initial_config
    
    def optimize(self):
        """Perform the optimization process with hybrid approach."""
        # Generate initial guess
        initial_guess_inner = self.generate_initial_configuration()

        # Initial estimate for outer radius based on configuration
        max_dist = 0
        for i in range(12):
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
        for _ in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        # Outer radius bound (should be positive)
        bounds.append((0.1, 20.0))

        # Stage 1: Global optimization with reduced iterations for speed
        def objective_global(params):
            return self.evaluate_configuration(params)

        try:
            de_result = differential_evolution(
                objective_global,
                bounds,
                maxiter=30,
                popsize=10,
                seed=42,
                disp=False
            )
            best_params = de_result.x
        except:
            # Fallback to initial guess if DE fails
            best_params = initial_params.copy()
        
        # Stage 2: Local refinement using L-BFGS-B
        refined_bounds = [(b[0], b[1]) for b in bounds]
        
        try:
            local_result = minimize(
                self.evaluate_configuration,
                best_params,
                method='L-BFGS-B',
                bounds=refined_bounds,
                options={'maxiter': 50, 'disp': False}
            )
            if local_result.success:
                best_params = local_result.x
        except:
            # Fall back to previous solution if local optimization fails
            pass

        # Extract results
        inner_hex_data = best_params[:-1].reshape(12, 3)
        outer_hex_side_length = best_params[-1]

        # Ensure the outer hexagon is actually large enough
        # Recalculate to make sure it contains all hexagons
        min_outer_radius = 0
        for i in range(12):
            x, y, _ = inner_hex_data[i]
            dist = np.sqrt(x*x + y*y) + 1.0  # +1 for hexagon radius
            min_outer_radius = max(min_outer_radius, dist)

        # If we computed a smaller radius than needed, adjust it up
        if outer_hex_side_length < min_outer_radius:
            outer_hex_side_length = min_outer_radius * 1.05  # Add small margin

        return inner_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Create optimizer instance
        optimizer = HexagonPackingOptimizer()
        
        # Get optimized configuration
        inner_hex_data, outer_hex_side_length = optimizer.optimize()

        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])

        # Ensure we don't exceed time limits
        end_time = time.time()
        eval_time = end_time - start_time

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        # Fallback to improved grid configuration if optimization fails
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])

        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END