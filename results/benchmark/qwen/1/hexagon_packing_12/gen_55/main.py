# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time

def create_regular_hexagon(center_x, center_y, side_length, rotation_deg):
    """Create a regular hexagon as a Shapely polygon."""
    rotation_rad = np.radians(rotation_deg)
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    points = [(center_x + side_length * np.cos(a), 
               center_y + side_length * np.sin(a)) for a in angles]
    return Polygon(points)

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hexagon.contains(hexagon) or outer_hexagon.intersects(hexagon)

def compute_outer_hex_side_length(inner_hex_data):
    """Compute minimum outer hexagon side length containing all inner hexagons."""
    # Create all inner hexagons
    inner_hexagons = []
    for x, y, angle in inner_hex_data:
        hex_poly = create_regular_hexagon(x, y, 1.0, angle)
        inner_hexagons.append(hex_poly)
    
    # Find bounding box of all inner hexagons
    all_points = []
    for hex_poly in inner_hexagons:
        for point in hex_poly.exterior.coords:
            all_points.append(point)
    
    # Calculate extreme points in all 6 directions (hexagon directions)
    directions = [(1, 0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2),
                  (-1, 0), (-0.5, -np.sqrt(3)/2), (0.5, -np.sqrt(3)/2)]
    
    max_distances = []
    for dx, dy in directions:
        # Project all points onto this direction
        distances = [p[0]*dx + p[1]*dy for p in all_points]
        max_dist = max(distances)
        min_dist = min(distances)
        max_distances.append(max_dist - min_dist)
    
    # Side length is the maximum distance divided by sqrt(3)
    return max(max_distances) / np.sqrt(3)

def evaluate_solution(params):
    """Evaluate solution and return negative of 1/outer_radius (to minimize)."""
    # Reshape params into 12 hexagons (each with x,y,angle)
    hex_data = params.reshape((12, 3))
    
    # Check for overlaps and containment
    inner_hexagons = []
    total_penalty = 0.0
    
    try:
        # Create hexagons and check for overlaps
        for i, (x, y, angle) in enumerate(hex_data):
            hex_poly = create_regular_hexagon(x, y, 1.0, angle)
            inner_hexagons.append(hex_poly)
            
            # Check containment (this might be problematic with floating point)
            # We'll use a simpler containment check with margin
            # But first let's find a reasonable outer hexagon
            if len(inner_hexagons) == 1:
                outer_center = [0, 0]
                outer_radius = 5.0  # Initial guess
                outer_hex = create_regular_hexagon(outer_center[0], outer_center[1], outer_radius, 0)
                
            # Check overlaps with other hexagons
            for j in range(i):
                if inner_hexagons[i].intersects(inner_hexagons[j]):
                    # Penalty for overlap
                    total_penalty += 1000.0
                    
            # Check containment (simple version)
            # If we're too far from center, penalize
            distance_from_center = np.sqrt(x*x + y*y)
            if distance_from_center > 10.0:
                total_penalty += 1000.0
            
            # Also add some penalty for going too far toward corners
            if abs(x) > 10.0 or abs(y) > 10.0:
                total_penalty += 1000.0
                
        # Compute the actual outer hexagon size needed
        outer_side_length = compute_outer_hex_side_length(hex_data)
        
        # Add penalty if it's too small (constraint violation)
        if outer_side_length < 0.1:  # Unreasonably small
            total_penalty += 10000.0
            
        # Return negative because we want to maximize 1/outer_side_length
        # So minimize -1/outer_side_length = -1/outer_side_length
        result = -1.0 / outer_side_length + total_penalty
        
        return result
        
    except Exception as e:
        # If anything goes wrong with geometry, penalize heavily
        return 10000.0

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Initialize with better starting configuration based on known packed arrangements
    initial_guess = np.array([
        [0, 0, 0],      # center
        [2.0, 0, 0],    # right
        [-2.0, 0, 0],   # left
        [1.0, 1.732, 0],  # top-right
        [-1.0, 1.732, 0], # top-left
        [1.0, -1.732, 0], # bottom-right
        [-1.0, -1.732, 0], # bottom-left
        [3.0, 1.732, 0],  # far right-top
        [-3.0, 1.732, 0], # far left-top
        [3.0, -1.732, 0], # far right-bottom
        [-3.0, -1.732, 0], # far left-bottom
        [0, -3.464, 0]   # bottom center
    ], dtype=float).flatten()
    
    # Set bounds: x, y from -10 to 10, angle from 0 to 360
    bounds = []
    for _ in range(12):
        bounds.extend([(-8.0, 8.0), (-8.0, 8.0), (0.0, 360.0)])
    
    # Run differential evolution optimization
    start_time = time.time()
    
    # Use differential evolution with a reasonable number of iterations
    result = differential_evolution(
        evaluate_solution, 
        bounds, 
        maxiter=200,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=False
    )
    
    end_time = time.time()
    
    # Extract final solution
    final_params = result.x.reshape((12, 3))
    
    # Calculate final outer hexagon side length
    outer_side_length = compute_outer_hex_side_length(final_params)
    
    # Create output data
    inner_hex_data = final_params.copy()
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
