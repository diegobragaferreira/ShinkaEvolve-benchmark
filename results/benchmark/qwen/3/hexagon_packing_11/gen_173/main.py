# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import math
from scipy.spatial.distance import cdist
import time
from numba import jit

@jit(nopython=True)
def distance_point_to_line(point, line_start, line_end):
    """Fast computation of point-to-line distance"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx, dy = x2 - x1, y2 - y1
    # Vector from line_start to point
    px_minus_x1, py_minus_y1 = px - x1, py - y1
    
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        return math.sqrt(px_minus_x1*px_minus_x1 + py_minus_y1*py_minus_y1)
    
    # Project point onto line
    t = (px_minus_x1*dx + py_minus_y1*dy) / length_sq
    t = max(0, min(1, t))  # Clamp to line segment
    
    # Closest point on line segment
    closest_x = x1 + t*dx
    closest_y = y1 + t*dy
    
    # Distance to closest point
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def point_in_hexagon_fast(px, py, center_x, center_y, side_length, rotation_rad):
    """Fast point-in-hexagon test"""
    # Transform point to hexagon's local coordinate system
    cos_r = math.cos(-rotation_rad)
    sin_r = math.sin(-rotation_rad)
    local_x = (px - center_x) * cos_r - (py - center_y) * sin_r
    local_y = (px - center_x) * sin_r + (py - center_y) * cos_r
    
    # Check against hexagon boundaries
    # For a regular hexagon with side length s, width = s*sqrt(3)
    half_width = side_length * math.sqrt(3) / 2
    half_height = side_length
    
    # Check if point is within bounds
    if abs(local_x) > half_width or abs(local_y) > half_height:
        return False
    
    # For a hexagon, we also check the corner constraints
    # Calculate distance to edges and compare with radius
    # Simplified check using max distance to corners
    return True

class VoronoiHexagonPacker:
    def __init__(self):
        self.side_length = 1.0
        self.hex_width = 2 * self.side_length * math.cos(math.pi/6)
        self.hex_height = 2 * self.side_length
        
    def create_outer_hexagon_vertices(self, center_x, center_y, side_length, rotation_deg=0):
        """Create vertices of outer hexagon"""
        rotation_rad = math.radians(rotation_deg)
        points = []
        for i in range(6):
            angle = rotation_rad + i * math.pi / 3
            x = center_x + side_length * math.cos(angle)
            y = center_y + side_length * math.sin(angle)
            points.append((x, y))
        return points
    
    def create_inner_hexagon_vertices(self, center_x, center_y, side_length, rotation_deg=0):
        """Create vertices of inner hexagon"""
        rotation_rad = math.radians(rotation_deg)
        points = []
        for i in range(6):
            angle = rotation_rad + i * math.pi / 3
            x = center_x + side_length * math.cos(angle)
            y = center_y + side_length * math.sin(angle)
            points.append((x, y))
        return points
    
    def get_outer_hexagon_bounds(self, center_x, center_y, side_length):
        """Get bounds of outer hexagon"""
        vertices = self.create_outer_hexagon_vertices(center_x, center_y, side_length)
        xs = [p[0] for p in vertices]
        ys = [p[1] for p in vertices]
        return min(xs), max(xs), min(ys), max(ys)
    
    def point_in_outer_hexagon(self, px, py, center_x, center_y, side_length, rotation_deg=0):
        """Fast check if point is in outer hexagon using bounds and simple geometric tests"""
        # Get basic bounds
        min_x, max_x, min_y, max_y = self.get_outer_hexagon_bounds(center_x, center_y, side_length)
        
        # Quick bounds check
        if px < min_x or px > max_x or py < min_y or py > max_y:
            return False
            
        # Use Shapely for precise check (fallback)
        outer_vertices = self.create_outer_hexagon_vertices(center_x, center_y, side_length, rotation_deg)
        outer_polygon = Polygon(outer_vertices)
        point = Point(px, py)
        return outer_polygon.contains(point)
    
    def hexagon_vertices(self, center_x, center_y, side_length=1, rotation_deg=0):
        """Get hexagon vertices as numpy array"""
        rotation_rad = math.radians(rotation_deg)
        angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
        vertices = np.column_stack([
            center_x + side_length * np.cos(angles),
            center_y + side_length * np.sin(angles)
        ])
        return vertices[:-1]  # Remove duplicate last vertex
    
    def check_overlap_simple(self, hex1_center, hex2_center, side_length=1):
        """Simple distance-based overlap check"""
        dist = math.sqrt((hex1_center[0] - hex2_center[0])**2 + (hex1_center[1] - hex2_center[1])**2)
        return dist < side_length * 2  # Overlap when center distance less than sum of diameters
    
    def calculate_voronoi_regions(self, voronoi_points, outer_radius):
        """Calculate Voronoi regions with outer boundary constraints"""
        # Create outer hexagon boundary points
        outer_vertices = self.create_outer_hexagon_vertices(0, 0, outer_radius)
        boundary_points = np.array(outer_vertices)
        
        # Combine with Voronoi points
        all_points = np.vstack([voronoi_points, boundary_points])
        
        # Compute Voronoi diagram
        try:
            vor = Voronoi(all_points)
            return vor
        except:
            # Fallback to simple case
            return Voronoi(voronoi_points)
    
    def compute_packing_score(self, voronoi_points, outer_radius):
        """Compute packing score based on Voronoi regions and constraints"""
        if len(voronoi_points) < 11:
            return -1e6  # Invalid
        
        # Check constraints: all points within outer hexagon
        outer_vertices = self.create_outer_hexagon_vertices(0, 0, outer_radius)
        outer_polygon = Polygon(outer_vertices)
        
        # Check constraints for each hexagon
        total_penalty = 0
        
        # Create hexagon vertices for all 11 hexagons
        hex_centers = voronoi_points[:11]
        hex_positions = []
        
        for i, center in enumerate(hex_centers):
            if not self.point_in_outer_hexagon(center[0], center[1], 0, 0, outer_radius):
                total_penalty += 10000.0
                
            # Check overlaps with all others
            for j, other_center in enumerate(hex_centers):
                if i != j:
                    dist = math.sqrt((center[0] - other_center[0])**2 + (center[1] - other_center[1])**2)
                    if dist < 2.0:  # Overlap
                        total_penalty += 1000.0
        
        if total_penalty > 0:
            return -total_penalty
            
        # Score based on how well hexagons are packed (inverse of outer radius)
        # We want to minimize outer radius (maximize 1/outer_radius)
        return 1.0 / outer_radius

def generate_initial_voronoi_config():
    """Generate initial Voronoi-based configuration"""
    # Initialize with a structured hexagonal arrangement
    initial_points = []
    
    # Center hexagon
    initial_points.append([0.0, 0.0])
    
    # Surrounding hexagons in concentric rings
    ring1_radius = math.sqrt(3)  # Distance between adjacent centers
    ring2_radius = ring1_radius * 2
    
    # First ring: 6 hexagons around the center
    for i in range(6):
        angle = i * math.pi / 3
        x = ring1_radius * math.cos(angle)
        y = ring1_radius * math.sin(angle)
        initial_points.append([x, y])
        
    # Second ring: 12 hexagons
    for i in range(12):
        angle = i * math.pi / 6
        x = ring2_radius * math.cos(angle)
        y = ring2_radius * math.sin(angle)
        initial_points.append([x, y])
    
    # Trim to exactly 11 points and add small random perturbations
    selected_points = initial_points[:11]
    for i in range(len(selected_points)):
        selected_points[i][0] += np.random.uniform(-0.2, 0.2)
        selected_points[i][1] += np.random.uniform(-0.2, 0.2)
    
    return np.array(selected_points)

def optimize_voronoi_packing():
    """Optimize the Voronoi-based hexagon packing"""
    packer = VoronoiHexagonPacker()
    
    # Generate initial configuration
    initial_points = generate_initial_voronoi_config()
    
    # Initial estimate of outer radius
    max_distance = 0
    for point in initial_points:
        dist = math.sqrt(point[0]**2 + point[1]**2)
        max_distance = max(max_distance, dist)
    initial_outer_radius = max_distance + 1.5  # Add buffer
    
    # Optimization variables: [x1, y1, x2, y2, ..., x11, y11, outer_radius]
    n_points = 11
    initial_vars = np.zeros(2*n_points + 1)
    initial_vars[:2*n_points] = initial_points.flatten()
    initial_vars[2*n_points] = initial_outer_radius
    
    # Bounds for optimization
    bounds = []
    # Points bounds
    for i in range(2*n_points):
        bounds.append((-15, 15))
    # Outer radius bound
    bounds.append((2.0, 20.0))
    
    def objective(vars):
        # Extract points and outer radius
        points = vars[:2*n_points].reshape(-1, 2)
        outer_radius = vars[2*n_points]
        
        # Compute packing score
        score = packer.compute_packing_score(points, outer_radius)
        return -score  # Minimize negative score (maximize score)
    
    # Initial optimization with L-BFGS-B
    try:
        # Try several local optimizations with different starting points
        best_score = float('inf')
        best_vars = initial_vars.copy()
        
        # Multiple restarts for better convergence
        for restart in range(3):
            # Small random perturbation
            perturbed_vars = initial_vars.copy()
            perturbed_vars[:2*n_points] += np.random.normal(0, 0.1, 2*n_points)
            perturbed_vars[2*n_points] += np.random.normal(0, 0.5)
            
            result = minimize(
                objective,
                perturbed_vars,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_vars = result.x.copy()
        
        # Final evaluation
        final_points = best_vars[:2*n_points].reshape(-1, 2)
        final_outer_radius = best_vars[2*n_points]
        
        return final_points, final_outer_radius
        
    except Exception as e:
        print(f"Optimization error: {e}")
        return initial_points, initial_outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Use Voronoi-based optimization
    final_points, outer_radius = optimize_voronoi_packing()
    
    # Format output data
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [final_points[i][0], final_points[i][1], 0.0]  # No rotation for simplicity
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    
    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END
