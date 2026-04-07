# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from numba import jit
import warnings

# Geometric computation module
@jit(nopython=True)
def hexagon_vertices_numba(x, y, angle_deg, side_length=1):
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

@jit(nopython=True)
def point_in_hexagon_numba(px, py, hx, hy, angle_deg, side_length=1):
    """Fast point-in-hexagon test."""
    vertices = hexagon_vertices_numba(hx, hy, angle_deg, side_length)
    # Ray casting method
    n = len(vertices)
    inside = False
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def create_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices_numba(x, y, angle_deg, side_length)
    return Polygon(vertices)

# Constraint validation module
class ConstraintValidator:
    """Handles all constraint checking operations."""
    
    def __init__(self):
        self.cache = {}
        
    def validate_containment(self, hex_poly, outer_hex_poly):
        """Check if hexagon is fully contained within outer hexagon."""
        # Quick centroid check first
        if not outer_hex_poly.contains(hex_poly.centroid):
            return False
        
        # Verify all vertices are inside
        for point_coords in list(hex_poly.exterior.coords):
            if not outer_hex_poly.contains(Point(point_coords)):
                return False
        return True
    
    def compute_overlaps(self, hexagon_configs, outer_radius):
        """Compute overlap penalties using efficient spatial indexing."""
        n = len(hexagon_configs)
        if n <= 1:
            return 0.0, []
            
        # Precompute all polygons for reuse
        polygons = [create_hexagon_polygon(x, y, angle) for x, y, angle in hexagon_configs]
        
        # Create spatial index of centers
        centroids = np.array([(x, y) for x, y, _ in hexagon_configs])
        tree = cKDTree(centroids)
        
        # Find candidate pairs within 2.5 unit distance
        pairs = tree.query_pairs(2.5, output_type='ndarray')
        overlap_penalty = 0.0
        overlap_pairs = []
        
        # Process pairs efficiently
        for i, j in pairs:
            if i >= j:
                continue
                
            # Quick distance check
            dist = np.sqrt((hexagon_configs[i][0] - hexagon_configs[j][0])**2 + 
                          (hexagon_configs[i][1] - hexagon_configs[j][1])**2)
            
            if dist > 2.5:  # Max possible distance for overlap
                continue
                
            # Detailed overlap check
            poly_i, poly_j = polygons[i], polygons[j]
            
            if poly_i.intersects(poly_j) and not poly_i.touches(poly_j):
                try:
                    overlap = poly_i.intersection(poly_j)
                    if hasattr(overlap, 'area') and overlap.area > 0:
                        overlap_penalty += overlap.area
                        overlap_pairs.append((min(i,j), max(i,j)))
                except:
                    overlap_penalty += 1000
                    
        return overlap_penalty, overlap_pairs

# Optimization engine module
class HexagonPackingOptimizer:
    """Main optimization engine for hexagon packing."""
    
    def __init__(self):
        self.validator = ConstraintValidator()
        self.initial_config = None
        self.best_result = None
        
    def create_outer_hexagon(self, side_length):
        """Create outer hexagon polygon."""
        vertices = hexagon_vertices_numba(0, 0, 0, side_length)
        return Polygon(vertices)
    
    def evaluate_fitness(self, params):
        """Evaluate fitness of a configuration."""
        try:
            # Parse parameters
            inner_configs = params[:-1].reshape(12, 3)
            outer_radius = params[-1]
            
            # Create outer hexagon
            outer_poly = self.create_outer_hexagon(outer_radius)
            
            # Check containment
            containment_valid = True
            total_penetration = 0.0
            
            for i, (x, y, angle) in enumerate(inner_configs):
                hex_poly = create_hexagon_polygon(x, y, angle)
                
                if not self.validator.validate_containment(hex_poly, outer_poly):
                    containment_valid = False
                    # Estimate penetration
                    try:
                        diff = outer_poly.difference(hex_poly)
                        if hasattr(diff, 'area'):
                            total_penetration += diff.area
                    except:
                        total_penetration += 1000
                    break
                    
            if not containment_valid:
                return 1e10 + total_penetration * 10000
            
            # Check overlaps
            overlap_penalty, _ = self.validator.compute_overlaps(inner_configs, outer_radius)
            
            if overlap_penalty > 0:
                return overlap_penalty * 10000
            
            # Valid configuration - return negative inverse side length
            return -1.0 / outer_radius
            
        except Exception as e:
            warnings.warn(f"Evaluation error: {e}")
            return 1e10

    def generate_initial_config(self):
        """Generate initial configuration based on geometric principles."""
        # Known good starting configuration
        positions = []
        
        # Central hexagon
        positions.append([0, 0, 0])
        
        # First ring - 6 hexagons
        for i in range(6):
            angle = i * 60
            x = 1.732 * np.cos(np.radians(angle))  # ~= sqrt(3)
            y = 1.732 * np.sin(np.radians(angle))
            positions.append([x, y, 0])
            
        # Second ring - 6 hexagons
        for i in range(6):
            angle = i * 60 + 30
            x = 3.464 * np.cos(np.radians(angle))  # ~= 2*sqrt(3)
            y = 3.464 * np.sin(np.radians(angle))
            positions.append([x, y, 0])
        
        initial_config = np.array(positions[:12])
        
        # Add small random perturbations
        np.random.seed(42)
        initial_config[:, :2] += np.random.normal(0, 0.1, (12, 2))
        
        return initial_config

    def optimize(self, max_iter=100):
        """Run optimization process."""
        # Generate initial configuration
        initial_config = self.generate_initial_config()
        
        # Estimate initial outer radius
        max_dist = 0
        for i in range(12):
            x, y, _ = initial_config[i]
            dist = np.sqrt(x*x + y*y)
            max_dist = max(max_dist, dist)
        initial_radius = max_dist + 2.0
        
        # Combine into parameter vector
        initial_params = np.concatenate([initial_config.flatten(), [initial_radius]])
        
        # Define bounds
        bounds = []
        for _ in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        bounds.append((0.1, 20.0))
        
        # Optimization with fallback strategies
        result = None
        try:
            # Try trust-constr first (often better for smooth problems)
            result = minimize(
                self.evaluate_fitness,
                initial_params,
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False}
            )
        except:
            pass
            
        if result is None or not result.success:
            try:
                # Fallback to L-BFGS-B
                result = minimize(
                    self.evaluate_fitness,
                    initial_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'disp': False}
                )
            except:
                pass
        
        if result is None or not result.success:
            # Final fallback to initial configuration
            final_params = initial_params
        else:
            final_params = result.x
            
        # Extract results
        inner_hex_data = final_params[:-1].reshape(12, 3)
        outer_hex_side_length = final_params[-1]
        
        return inner_hex_data, outer_hex_side_length

# Main execution module
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
        # Initialize optimizer
        optimizer = HexagonPackingOptimizer()
        
        # Run optimization
        inner_hex_data, outer_hex_side_length = optimizer.optimize()
        
        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback to simple configuration
        warnings.warn(f"Fallback due to error: {e}")
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
        outer_hex_side_length = 8
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END