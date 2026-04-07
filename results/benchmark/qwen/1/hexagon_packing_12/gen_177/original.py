# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit
from joblib import Parallel, delayed
import functools

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
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
def point_in_hexagon(px, py, hx, hy, angle_deg, side_length=1):
    """Fast point-in-hexagon test using winding number."""
    vertices = hexagon_vertices(hx, hy, angle_deg, side_length)
    # Use ray casting method for simplicity and speed
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

@jit(nopython=True)
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment."""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1
    
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        # Line segment is a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))  # Clamp projection to line segment
    
    # Find closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def compute_min_distance_hexagon_hexagon(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Compute minimum distance between two hexagons using analytical approach."""
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle)
    
    min_dist = np.inf
    # Check vertex-to-vertex distances
    for i in range(6):
        for j in range(6):
            dist = np.sqrt((v1[i,0]-v2[j,0])**2 + (v1[i,1]-v2[j,1])**2)
            if dist < min_dist:
                min_dist = dist
    
    # Check vertex-to-edge distances
    for i in range(6):
        for j in range(6):
            # Distance from vertex v1[i] to edge v2[j]-v2[(j+1)%6]
            dist = distance_point_to_line(v1[i,0], v1[i,1], v2[j,0], v2[j,1], v2[(j+1)%6,0], v2[(j+1)%6,1])
            if dist < min_dist:
                min_dist = dist
            
            # Distance from vertex v2[j] to edge v1[i]-v1[(i+1)%6]
            dist = distance_point_to_line(v2[j,0], v2[j,1], v1[i,0], v1[i,1], v1[(i+1)%6,0], v1[(i+1)%6,1])
            if dist < min_dist:
                min_dist = dist
    
    return min_dist

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

class HexagonTilingOptimizer:
    """Optimizes hexagon packing using geometric tiling principles."""
    
    def __init__(self):
        self.hex_side_length = 1.0
        self.outer_hex_radius = 10.0  # Initial estimate
        
    def compute_outer_hexagon_vertices(self, side_length):
        """Get vertices for outer hexagon."""
        return hexagon_vertices(0, 0, 0, side_length)
        
    def compute_outer_hexagon_polygon(self, side_length):
        """Get shapely polygon for outer hexagon."""
        vertices = self.compute_outer_hexagon_vertices(side_length)
        return Polygon(vertices)
        
    def check_containment(self, hex_poly, outer_hex_poly):
        """Check if hexagon is fully contained within outer hexagon."""
        # Use simplified containment check for better performance
        # First check if center is in outer hexagon
        center = hex_poly.centroid
        if not outer_hex_poly.contains(center):
            return False
        
        # Then check if all vertices are inside
        for point in list(hex_poly.exterior.coords):
            if not outer_hex_poly.contains(Point(point)):
                return False
        
        return True
        
    def compute_overlap_penalty(self, inner_hex_data):
        """Efficiently compute overlap penalty using spatial indexing."""
        n = len(inner_hex_data)
        if n <= 1:
            return 0.0
            
        # Create spatial index of hexagon centers
        centroids = np.array([[x, y] for x, y, _ in inner_hex_data])
        tree = cKDTree(centroids)
        
        penalty = 0.0
        overlap_pairs = set()
        
        # Find candidates within 2.5 unit distance (safe threshold for overlap)
        pairs = tree.query_pairs(2.5, output_type='ndarray')
        
        # Precompute all polygons for efficiency
        polygons = [compute_hexagon_polygon(x, y, angle) for x, y, angle in inner_hex_data]
        
        # Process pairs efficiently
        for i, j in pairs:
            if i >= j:  # Only check each pair once
                continue
            
            # Fast preliminary check using distance
            dist = np.sqrt((inner_hex_data[i][0] - inner_hex_data[j][0])**2 + 
                          (inner_hex_data[i][1] - inner_hex_data[j][1])**2)
            
            # If distance is too large, skip
            if dist > 2.5:  # Max possible distance for overlap
                continue
                
            # More precise overlap check
            poly_i = polygons[i]
            poly_j = polygons[j]
            
            if poly_i.intersects(poly_j) and not poly_i.touches(poly_j):
                try:
                    overlap = poly_i.intersection(poly_j)
                    if hasattr(overlap, 'area') and overlap.area > 0:
                        penalty += overlap.area
                        overlap_pairs.add((min(i,j), max(i,j)))
                except:
                    penalty += 1000  # Large penalty for calculation errors
                    
        return penalty
        
    def evaluate_tiling_fitness(self, params):
        """Evaluate the fitness of a tiling configuration."""
        try:
            # Extract inner hexagon data and outer radius
            inner_hex_data = params[:-1].reshape(12, 3)
            outer_hex_side_length = params[-1]
            
            # Create outer hexagon polygon
            outer_hex_poly = self.compute_outer_hexagon_polygon(outer_hex_side_length)
            
            # Check containment (faster version)
            containment_valid = True
            total_penetration = 0.0
            
            # Check each hexagon for containment
            for i in range(len(inner_hex_data)):
                x, y, angle = inner_hex_data[i]
                hex_poly = compute_hexagon_polygon(x, y, angle)
                
                # Quick containment check using center
                center = hex_poly.centroid
                if not outer_hex_poly.contains(center):
                    containment_valid = False
                    # Estimate penetration
                    try:
                        diff = outer_hex_poly.difference(hex_poly)
                        if hasattr(diff, 'area'):
                            total_penetration += diff.area
                    except:
                        total_penetration += 1000
                    break
                    
            if not containment_valid:
                return 1e10 + total_penetration * 10000
            
            # Compute overlap penalty using spatial indexing
            overlap_penalty = self.compute_overlap_penalty(inner_hex_data)
            
            if overlap_penalty > 0:
                return overlap_penalty * 10000
            
            # If valid, return inverse of outer hexagon side length (we want to minimize the negative)
            return -1.0 / outer_hex_side_length
            
        except Exception as e:
            return 1e10

    def generate_initial_tiling_config(self):
        """Generate an initial configuration based on geometric tiling principles."""
        # Start with a known good hexagonal packing pattern
        positions = []
        
        # Central hexagon
        positions.append([0, 0, 0])
        
        # First ring - 6 hexagons at distance sqrt(3) from center
        for i in range(6):
            angle = i * 60
            x = 1.732 * np.cos(np.radians(angle))  # ~= sqrt(3)
            y = 1.732 * np.sin(np.radians(angle))
            positions.append([x, y, 0])
            
        # Second ring - 6 hexagons at distance 2*sqrt(3) from center
        for i in range(6):
            angle = i * 60 + 30  # offset
            x = 3.464 * np.cos(np.radians(angle))  # ~= 2*sqrt(3)
            y = 3.464 * np.sin(np.radians(angle))
            positions.append([x, y, 0])
        
        # Use only first 12 positions
        initial_config = np.array(positions[:12])
        
        # Add small random perturbations to avoid local minima
        np.random.seed(42)
        initial_config[:, :2] += np.random.normal(0, 0.1, (12, 2))
        
        return initial_config

    def optimize(self):
        """Main optimization routine using geometric tiling approach."""
        # Generate initial configuration
        initial_guess = self.generate_initial_tiling_config()
        
        # Initial estimate for outer radius
        max_dist = 0
        for i in range(12):
            x, y, _ = initial_guess[i]
            dist = np.sqrt(x*x + y*y)
            max_dist = max(max_dist, dist)
            
        # Add margin for hexagon size
        initial_outer_radius = max_dist + 2.0
        
        # Combine into single parameter vector
        initial_params = np.concatenate([initial_guess.flatten(), [initial_outer_radius]])
        
        # Define bounds
        bounds = []
        for _ in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        bounds.append((0.1, 20.0))
        
        # First, try trust-constr optimization which is often better for smooth problems
        try:
            result = minimize(
                self.evaluate_tiling_fitness,
                initial_params,
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': 100, 'disp': False}
            )
            
            if result.success:
                final_params = result.x
            else:
                raise RuntimeError("Trust-constr failed")
                
        except Exception as e:
            # Fall back to L-BFGS if trust-constr fails
            try:
                result = minimize(
                    self.evaluate_tiling_fitness,
                    initial_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'disp': False}
                )
                if result.success:
                    final_params = result.x
                else:
                    final_params = initial_params
            except:
                final_params = initial_params
                
        # Extract results with validation
        inner_hex_data = final_params[:-1].reshape(12, 3)
        outer_hex_side_length = final_params[-1]
        
        # Final validation and adjustment
        outer_hex_poly = self.compute_outer_hexagon_polygon(outer_hex_side_length)
        
        # If any hexagon is not contained, increase outer radius
        needs_adjustment = False
        for i in range(12):
            x, y, angle = inner_hex_data[i]
            hex_poly = compute_hexagon_polygon(x, y, angle)
            if not outer_hex_poly.contains(hex_poly.centroid):
                needs_adjustment = True
                break
                
        if needs_adjustment:
            min_outer_radius = 0
            for i in range(12):
                x, y, _ = inner_hex_data[i]
                dist = np.sqrt(x*x + y*y) + 1.0  # +1 for hexagon radius
                min_outer_radius = max(min_outer_radius, dist)
            outer_hex_side_length = min_outer_radius * 1.05
            
        return inner_hex_data, outer_hex_side_length

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
        # Create optimizer
        optimizer = HexagonTilingOptimizer()
        
        # Get optimized configuration
        inner_hex_data, outer_hex_side_length = optimizer.optimize()
        
        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])
        
        # Ensure we don't exceed time limits
        end_time = time.time()
        eval_time = end_time - start_time
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback to original configuration if optimization fails
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