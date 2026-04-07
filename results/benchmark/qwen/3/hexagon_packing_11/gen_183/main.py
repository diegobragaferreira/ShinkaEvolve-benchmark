# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time

class HexagonTilingOptimizer:
    def __init__(self):
        self.side_length = 1.0
        self.hex_width = math.sqrt(3)  # distance between parallel sides
        self.hex_height = 2.0          # distance between opposite vertices
        self.hex_spacing = math.sqrt(3) # center-to-center distance in hexagonal lattice
    
    def create_hexagon(self, center_x, center_y, angle_deg=0):
        """Create a regular hexagon polygon"""
        angle_rad = math.radians(angle_deg)
        points = []
        for i in range(6):
            angle = angle_rad + i * math.pi / 3
            x = center_x + self.side_length * math.cos(angle)
            y = center_y + self.side_length * math.sin(angle)
            points.append((x, y))
        return Polygon(points)
    
    def get_hexagon_bounds(self, center_x, center_y, angle_deg=0):
        """Get axis-aligned bounding box for hexagon"""
        # Approximate bounds for faster collision detection
        # For unit hexagon with side length 1
        half_width = self.hex_width / 2
        half_height = self.hex_height / 2
        
        # Apply rotation
        cos_a = math.cos(math.radians(angle_deg))
        sin_a = math.sin(math.radians(angle_deg))
        
        # Rotated bounds
        r_width = abs(half_width * cos_a) + abs(half_height * sin_a)
        r_height = abs(half_width * sin_a) + abs(half_height * cos_a)
        
        return (
            center_x - r_width, 
            center_x + r_width,
            center_y - r_height,
            center_y + r_height
        )
    
    def check_collision_bounds(self, bounds1, bounds2):
        """Fast bounds-based collision check"""
        x1_min, x1_max, y1_min, y1_max = bounds1
        x2_min, x2_max, y2_min, y2_max = bounds2
        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)
    
    def check_containment_and_overlap(self, inner_hexagons, outer_hexagon):
        """Check containment and overlaps using fast bounds and exact Shapely operations"""
        # Check containment with bounds first
        for hex_poly in inner_hexagons:
            # Quick bounds check
            if hasattr(hex_poly, 'bounds'):
                bounds = hex_poly.bounds
                if not (bounds[0] >= outer_hexagon.bounds[0] and 
                       bounds[2] >= outer_hexagon.bounds[2] and
                       bounds[1] <= outer_hexagon.bounds[1] and
                       bounds[3] <= outer_hexagon.bounds[3]):
                    return False

        # Exact containment and overlap checks
        for i, hex_poly in enumerate(inner_hexagons):
            if not outer_hexagon.contains(hex_poly):
                return False
        
        # Pairwise overlap checking
        for i in range(len(inner_hexagons)):
            for j in range(i+1, len(inner_hexagons)):
                if inner_hexagons[i].intersects(inner_hexagons[j]):
                    return False
        
        return True
    
    def compute_outer_radius_from_positions(self, positions, angles):
        """Compute minimal outer hexagon radius from hexagon positions"""
        all_vertices = []
        for i, (pos, angle) in enumerate(zip(positions, angles)):
            hex_poly = self.create_hexagon(pos[0], pos[1], angle)
            all_vertices.extend(list(hex_poly.exterior.coords))
        
        if not all_vertices:
            return 1.0
            
        # Find center of all vertices
        xs = [p[0] for p in all_vertices]
        ys = [p[1] for p in all_vertices]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        
        # Compute max distance from center to any vertex
        max_dist = 0
        for x, y in all_vertices:
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = max(max_dist, dist)
        
        return max_dist + 0.01  # Small padding

def generate_hexagonal_layout():
    """Generate initial hexagonal layout based on symmetry principles"""
    # Create a honeycomb-like pattern that naturally minimizes gaps
    positions = []
    angles = []
    
    # Central hexagon
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Surrounding hexagons in a hexagonal pattern
    hex_spacing = math.sqrt(3)  # distance between centers
    
    # First ring around center
    for i in range(6):
        angle = i * 60
        rad = math.radians(angle)
        x = hex_spacing * math.cos(rad)
        y = hex_spacing * math.sin(rad)
        positions.append([x, y])
        angles.append(0.0)
    
    # Second ring
    for i in range(6):
        angle = i * 60 + 30  # offset for second ring
        rad = math.radians(angle)
        x = hex_spacing * 2 * math.cos(rad)
        y = hex_spacing * 2 * math.sin(rad)
        positions.append([x, y])
        angles.append(0.0)
    
    # Trim to exactly 11 hexagons
    return np.array(positions[:11]), np.array(angles[:11])

def hexagon_tiling_optimization():
    """Main optimization using hexagonal tiling approach"""
    optimizer = HexagonTilingOptimizer()
    
    # Generate initial layout
    positions, angles = generate_hexagonal_layout()
    
    # Precompute initial outer radius
    initial_radius = optimizer.compute_outer_radius_from_positions(positions, angles)
    
    # Objective function: minimize outer radius with penalties for violations
    def objective(params):
        # params: [x0,y0,a0,x1,y1,a1,...,x10,y10,a10,radius]
        n_hexes = 11
        coords = params[:-1].reshape(n_hexes, 3)  # [[x0,y0,a0], ...]
        radius = params[-1]
        
        # Extract positions and angles
        pos_array = coords[:, :2]
        ang_array = coords[:, 2]
        
        # Create hexagons
        inner_hexagons = []
        for i in range(n_hexes):
            hex_poly = optimizer.create_hexagon(pos_array[i][0], pos_array[i][1], ang_array[i])
            inner_hexagons.append(hex_poly)
        
        # Create outer hexagon
        outer_hexagon = optimizer.create_hexagon(0, 0, 0, radius)
        
        # Check constraints
        valid = optimizer.check_containment_and_overlap(inner_hexagons, outer_hexagon)
        
        # Penalty for invalid configurations
        if not valid:
            return 1000000 + radius  # Large penalty for constraint violations
        
        # Return radius as objective (we want to minimize it)
        return radius
    
    # Optimization with bounds
    # Parameters: 11 hexagons' (x,y,angle) + 1 outer radius
    n_params = 11 * 3 + 1
    initial_params = np.zeros(n_params)
    
    # Fill positions
    for i in range(11):
        initial_params[i*3] = positions[i][0]      # x
        initial_params[i*3+1] = positions[i][1]    # y  
        initial_params[i*3+2] = angles[i]          # angle
    
    # Initial radius guess
    initial_params[-1] = initial_radius
    
    # Bounds for positions (reasonable range)
    bounds = []
    for i in range(11):
        bounds.extend([(-20, 20), (-20, 20), (-180, 180)])  # x, y, angle
    bounds.append((1.0, 20.0))  # outer radius
    
    # Use L-BFGS-B for optimization
    try:
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=lambda x: None  # No callback for now
        )
        
        if result.success:
            final_params = result.x
            final_coords = final_params[:-1].reshape(11, 3)
            final_radius = final_params[-1]
        else:
            # Fallback to initial if optimization fails
            final_coords = positions
            final_radius = initial_radius
            
    except Exception as e:
        # Even if optimization fails, return something reasonable
        final_coords = positions
        final_radius = initial_radius
    
    # Convert results to required format
    inner_hex_data = final_coords.copy()
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = final_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Run the tile-based optimization approach
    return hexagon_tiling_optimization()

# EVOLVE-BLOCK-END
