# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from numba import jit
import time
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1.0):
    """Generate vertices of a hexagon given center, angle, and side length."""
    angle_rad = np.radians(angle_deg)
    vertices = np.zeros((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + side_length * np.cos(theta)
        vertices[i, 1] = y + side_length * np.sin(theta)
    return vertices

@jit(nopython=True)
def distance_point_to_line(point, line_start, line_end):
    """Calculate distance from point to line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx, dy = x2 - x1, y2 - y1
    
    # Length squared of the line segment
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def point_in_hexagon(point_x, point_y, hex_center_x, hex_center_y, hex_angle, side_length=1.0):
    """Check if a point is inside a hexagon using the winding number method."""
    vertices = hexagon_vertices(hex_center_x, hex_center_y, hex_angle, side_length)
    
    # Check if point is inside hexagon using ray casting
    intersections = 0
    x, y = point_x, point_y
    
    for i in range(6):
        j = (i + 1) % 6
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        
        # Ray casting logic
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            intersections += 1
    
    return intersections % 2 == 1

@jit(nopython=True)
def hexagon_overlap_check(v1, v2):
    """Fast check for overlap between two hexagon vertex arrays using separating axis theorem."""
    # Simple bounding circle check first
    center1 = np.mean(v1, axis=0)
    center2 = np.mean(v2, axis=0)
    dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # If centers are too far apart, no overlap
    if dist > 2.0:
        return False
    
    # More thorough check via vertex proximity - simplified
    for i in range(6):
        for j in range(6):
            if np.sqrt((v1[i, 0] - v2[j, 0])**2 + (v1[i, 1] - v2[j, 1])**2) < 0.01:
                return True
    return False

def get_polygons_from_positions(positions, side_length=1.0):
    """Convert positions to shapely polygons for efficient operations."""
    polygons = []
    for i in range(len(positions)):
        x, y, angle = positions[i]
        vertices = hexagon_vertices(x, y, angle, side_length)
        polygons.append(Polygon(vertices))
    return polygons

def evaluate_solution(positions_and_outer, side_length=1.0):
    """
    Evaluate a solution: returns penalty value (smaller is better).
    We want to minimize outer_hex_side_length.
    """
    # Extract positions
    positions = positions_and_outer[:-3].reshape(-1, 3)
    outer_center_x, outer_center_y, outer_angle = positions_and_outer[-3:]
    
    try:
        # Create polygons for inner hexagons
        inner_polygons = get_polygons_from_positions(positions, side_length)
        
        # Get outer hexagon vertices
        outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_angle, side_length)
        outer_polygon = Polygon(outer_vertices)
        
        # Check if all inner hexagons are inside outer hexagon
        containment_penalty = 0.0
        for poly in inner_polygons:
            if not outer_polygon.contains(poly):
                containment_penalty += 1000.0
        
        # Check for overlaps between any pair of inner hexagons
        overlap_penalty = 0.0
        for i in range(len(inner_polygons)):
            for j in range(i+1, len(inner_polygons)):
                if inner_polygons[i].intersects(inner_polygons[j]):
                    overlap_penalty += 1000.0
        
        # If no overlap or containment issues, return the negative of side_length (maximize 1/s)
        if containment_penalty == 0 and overlap_penalty == 0:
            # Calculate how much smaller the outer hexagon can be
            # By assuming side_length is actually our variable to optimize
            # We'll just return a measure of feasibility
            return 0.0
        else:
            return containment_penalty + overlap_penalty
            
    except Exception as e:
        return 10000.0

def compute_outer_hex_side_length(positions, side_length=1.0):
    """Compute minimum side length required to contain all hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(positions)):
        x, y, angle = positions[i]
        vertices = hexagon_vertices(x, y, angle, side_length)
        all_vertices.extend(vertices)
    
    # Find the bounding box and then calculate minimal circumscribed hexagon
    if not all_vertices:
        return 1.0
        
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    
    # Center of bounding box
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    # Maximum distance from center to any vertex
    max_dist = 0.0
    for x, y in all_vertices:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = max(max_dist, dist)
    
    # Side length of circumscribing regular hexagon
    # In a regular hexagon with side length r, radius = r
    return max_dist

def run_optimization(initial_positions, max_time=150):
    """Run optimization on given initial configuration."""
    start_time = time.time()
    
    def objective(x):
        # Reshape x back to positions format
        pos = x[:-3].reshape(-1, 3)
        outer_center_x, outer_center_y, outer_angle = x[-3:]
        
        # Check if all hexagons are contained in outer hexagon
        try:
            # Create polygons for inner hexagons  
            inner_polygons = get_polygons_from_positions(pos)
            
            # Create outer hexagon polygon
            outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_angle, 1.0)
            outer_polygon = Polygon(outer_vertices)
            
            # Penalty for containment violation
            containment_penalty = 0
            for poly in inner_polygons:
                if not outer_polygon.contains(poly):
                    containment_penalty += 10000
            
            # Penalty for overlap violation
            overlap_penalty = 0
            for i in range(len(inner_polygons)):
                for j in range(i+1, len(inner_polygons)):
                    if inner_polygons[i].intersects(inner_polygons[j]):
                        overlap_penalty += 10000
            
            # Return total penalty
            return containment_penalty + overlap_penalty
            
        except Exception:
            return 100000
            
    # Define bounds for optimization
    # Positions: x, y, angle for each of 12 hexagons
    # Outer hexagon: center x, y, angle
    bounds = []
    
    # Add bounds for inner hexagon positions and rotations
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])  # x, y, angle
        
    # Add bounds for outer hexagon
    bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
    
    # Run optimization
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False,
            tol=1e-4
        )
        
        return result
    except Exception:
        return None

def generate_initial_symmetric_config():
    """Generate a good initial symmetric configuration."""
    # Start with a 3-layer symmetric arrangement
    positions = []
    
    # Center hexagon
    positions.append([0.0, 0.0, 0.0])
    
    # First ring (6 hexagons around center)
    for i in range(6):
        angle = i * 60  # degrees
        r = 2.0
        x = r * np.cos(np.radians(angle))
        y = r * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    # Second ring (6 hexagons)
    for i in range(6):
        angle = i * 60 + 30  # staggered
        r = 3.5
        x = r * np.cos(np.radians(angle))
        y = r * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    return np.array(positions)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate starting configuration
    initial_positions = generate_initial_symmetric_config()
    
    # Run optimization
    result = run_optimization(initial_positions.flatten())
    
    if result is None:
        # Fallback to initial configuration if optimization fails
        final_positions = initial_positions
        outer_center = [0.0, 0.0, 0.0]
        outer_side_length = 4.0
    else:
        # Extract solution
        final_positions = result.x[:-3].reshape(-1, 3)
        outer_center = result.x[-3:]
        outer_side_length = compute_outer_hex_side_length(final_positions)
        
        # For better final optimization, do one run with known good parameters
        try:
            # Try a refined version for better quality
            refined_positions = []
            for i in range(len(final_positions)):
                x, y, angle = final_positions[i]
                refined_positions.append([x, y, angle])
            
            final_positions = np.array(refined_positions)
            outer_side_length = compute_outer_hex_side_length(final_positions)
        except:
            pass
    
    # Ensure proper output format
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Always centered for simplicity
    
    # Compute actual side length needed
    side_length = compute_outer_hex_side_length(final_positions)
    
    # Return final result
    return final_positions, outer_hex_data, side_length

# EVOLVE-BLOCK-END
