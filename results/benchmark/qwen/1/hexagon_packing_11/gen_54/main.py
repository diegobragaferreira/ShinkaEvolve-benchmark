# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.validation import make_valid
import time
import math

def create_regular_hexagon(center=(0, 0), side_length=1, rotation=0):
    """Create a regular hexagon as a shapely polygon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + side_length * np.cos(a),
               center[1] + side_length * np.sin(a)) for a in angles]
    return Polygon(points)

def get_hexagon_vertices(center, side_length, rotation):
    """Get all vertices of a hexagon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return [(center[0] + side_length * np.cos(a),
             center[1] + side_length * np.sin(a)) for a in angles]

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon"""
    try:
        return outer_hex_poly.contains(hexagon_poly)
    except:
        try:
            valid_outer = make_valid(outer_hex_poly)
            valid_hex = make_valid(hexagon_poly)
            return valid_outer.contains(valid_hex)
        except:
            return False

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap using Shapely"""
    try:
        poly1 = Polygon(hex1)
        poly2 = Polygon(hex2)
        return poly1.intersects(poly2)
    except:
        try:
            valid_poly1 = make_valid(Polygon(hex1))
            valid_poly2 = make_valid(Polygon(hex2))
            return valid_poly1.intersects(valid_poly2)
        except:
            return True

def compute_hexagon_radius(side_length):
    """Compute the circumradius of a regular hexagon"""
    return side_length

def compute_min_outer_radius(inner_hexes):
    """Compute minimum outer hexagon radius that contains all inner hexes"""
    # Get all vertices of all inner hexes
    all_vertices = []
    for hex_vertices in inner_hexes:
        all_vertices.extend(hex_vertices)
    
    # Convert to numpy array for easier computation
    vertices_array = np.array(all_vertices)
    
    # Find the centroid
    centroid = np.mean(vertices_array, axis=0)
    
    # Compute distances from centroid to all vertices
    distances = np.sqrt(np.sum((vertices_array - centroid)**2, axis=1))
    
    # Return maximum distance plus some margin (we'll use 1.1 to ensure safety)
    return np.max(distances) * 1.1

def calculate_penalties(inner_positions, inner_rotations, outer_radius):
    """Calculate penalties for containment and overlap issues"""
    # Create outer hexagon
    outer_hex = create_regular_hexagon((0, 0), outer_radius, 0)
    
    # Create all inner hexagon polygons
    inner_hexes = []
    for i in range(len(inner_positions)):
        pos = tuple(inner_positions[i])
        rot = inner_rotations[i]
        vertices = get_hexagon_vertices(pos, 1, rot)
        inner_hexes.append(vertices)
    
    # Check containment and overlaps
    total_penalty = 0
    
    # Check containment
    for i, hex_vertices in enumerate(inner_hexes):
        hex_poly = Polygon(hex_vertices)
        if not check_containment(hex_poly, outer_hex):
            total_penalty += 10000  # Large penalty for containment violations
    
    # Check overlap (optimized version)
    for i in range(len(inner_hexes)):
        for j in range(i+1, len(inner_hexes)):
            if check_overlap(inner_hexes[i], inner_hexes[j]):
                total_penalty += 100000  # Even larger penalty for overlaps
    
    return total_penalty

def objective_function(params, fixed_outer_radius=None):
    """Objective function to minimize (negative of 1/outer_radius + penalties)"""
    # Parse parameters - we will use a reduced parameter space approach
    n_hex = 11
    positions = params[:2*n_hex].reshape(-1, 2)
    rotations = params[2*n_hex:3*n_hex]
    
    if fixed_outer_radius is not None:
        outer_radius = fixed_outer_radius
    else:
        # Compute outer radius based on current configuration
        outer_radius = compute_min_outer_radius(
            [get_hexagon_vertices(pos, 1, rot) for pos, rot in zip(positions, rotations)]
        )
    
    # Calculate penalties
    penalties = calculate_penalties(positions, rotations, outer_radius)
    
    # Return negative of 1/outer_radius plus penalties for minimization
    inv_radius = 1.0 / outer_radius if outer_radius > 0 else 0
    return -inv_radius + penalties

def generate_initial_config():
    """Generate an initial configuration inspired by optimal hexagonal packings"""
    # Place 11 hexagons in a pattern that mimics hexagonal close packing
    positions = []
    rotations = []
    
    # Central hexagon
    positions.append([0.0, 0.0])
    rotations.append(0.0)
    
    # First ring: 6 hexagons at distance 2 from center (centers spaced by 2 units)
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    for angle in angles:
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle)
        positions.append([x, y])
        rotations.append(0.0)
    
    # Second ring: 4 additional hexagons in a more compact arrangement
    # For a 11-hexagon packing, we can place them strategically
    angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6)[1::2]  # Skip every other angle
    for i, angle in enumerate(angles[:4]):  # Only take 4 to make 11 total
        x = 3 * np.cos(angle)
        y = 3 * np.sin(angle)
        positions.append([x, y])
        rotations.append(0.0)
    
    # Remove extra elements to have exactly 11
    positions = positions[:11]
    rotations = rotations[:11]
    
    # Flatten parameters for optimization
    flat_params = []
    for pos in positions:
        flat_params.extend(pos)
    flat_params.extend(rotations)
    
    return np.array(flat_params)

def optimized_hexagonal_packaging():
    """Main optimization function using a hybrid approach"""
    # Start with an initial configuration
    initial_params = generate_initial_config()
    
    # Stage 1: Coarse global optimization to find good general arrangement
    # Using scipy minimize with L-BFGS-B for faster convergence
    bounds = []
    # Position bounds (-10, 10) for each of 11 hexagons
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10)])
    # Rotation bounds (0, 360) for each of 11 hexagons
    for _ in range(11):
        bounds.append((0, 360))
    
    # First optimization run with L-BFGS-B
    result1 = minimize(
        objective_function,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6},
        tol=1e-6
    )
    
    # Stage 2: Refinement using a different optimizer for better convergence
    # Use a trust-constr method which is often more robust
    result2 = minimize(
        objective_function,
        result1.x,
        method='trust-constr',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8},
        tol=1e-8
    )
    
    # Final post-processing to extract results
    final_params = result2.x
    n_hex = 11
    final_positions = final_params[:2*n_hex].reshape(-1, 2)
    final_rotations = final_params[2*n_hex:3*n_hex]
    
    # Compute final outer radius
    final_hexes = [get_hexagon_vertices(pos, 1, rot) for pos, rot in zip(final_positions, final_rotations)]
    final_outer_radius = compute_min_outer_radius(final_hexes)
    
    # Verify final solution
    penalties = calculate_penalties(final_positions, final_rotations, final_outer_radius)
    
    # Build output data
    inner_hex_data = np.column_stack([
        final_positions[:, 0],
        final_positions[:, 1],
        final_rotations
    ])
    
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    return inner_hex_data, outer_hex_data, final_outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses hybrid optimization approach combining global and local optimization techniques.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Run optimized hexagonal packaging
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimized_hexagonal_packaging()
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END