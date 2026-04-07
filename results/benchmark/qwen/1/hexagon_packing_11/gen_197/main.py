# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.validation import make_valid
import time

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
            return True  # if we can't validate, assume they overlap

def build_spatial_grid(hexagons, cell_size=3.0):
    """Build a simple spatial grid for fast collision detection"""
    grid = {}
    for i, hex_vertices in enumerate(hexagons):
        # Get bounding box of hexagon
        min_x = min(v[0] for v in hex_vertices)
        max_x = max(v[0] for v in hex_vertices)
        min_y = min(v[1] for v in hex_vertices)
        max_y = max(v[1] for v in hex_vertices)

        # Grid cells that this hexagon touches
        start_col = int(min_x // cell_size)
        end_col = int(max_x // cell_size)
        start_row = int(min_y // cell_size)
        end_row = int(max_y // cell_size)

        for col in range(start_col, end_col + 1):
            for row in range(start_row, end_row + 1):
                if (row, col) not in grid:
                    grid[(row, col)] = []
                grid[(row, col)].append(i)

    return grid

def get_candidate_overlaps(hexagons, spatial_grid, hex_idx, cell_size=3.0):
    """Get candidate hexagon indices that might overlap with hex_idx"""
    # Get the bounding box of the hexagon we're checking
    hex_vertices = hexagons[hex_idx]
    min_x = min(v[0] for v in hex_vertices)
    max_x = max(v[0] for v in hex_vertices)
    min_y = min(v[1] for v in hex_vertices)
    max_y = max(v[1] for v in hex_vertices)

    # Get grid cells this hexagon touches
    start_col = int(min_x // cell_size)
    end_col = int(max_x // cell_size)
    start_row = int(min_y // cell_size)
    end_row = int(max_y // cell_size)

    candidates = set()
    for col in range(start_col, end_col + 1):
        for row in range(start_row, end_row + 1):
            if (row, col) in spatial_grid:
                candidates.update(spatial_grid[(row, col)])

    return candidates

def evaluate_solution(params):
    """Evaluate a solution and return negative combined score (for minimization)"""
    # Parse parameters
    inner_positions = params[:22].reshape(-1, 2)  # x,y pairs
    inner_rotations = params[22:33]  # 11 rotations
    outer_radius = params[33]  # outer hex radius

    # Create outer hexagon
    outer_hex = create_regular_hexagon((0, 0), outer_radius, 0)

    # Check if all inner hexagons fit inside outer hexagon
    num_inner_hexes = len(inner_positions)

    # Create all inner hexagon polygons
    inner_hexes = []
    for i in range(num_inner_hexes):
        pos = tuple(inner_positions[i])
        rot = inner_rotations[i]
        vertices = get_hexagon_vertices(pos, 1, rot)
        inner_hexes.append(vertices)

    # Build spatial grid for faster overlap detection
    spatial_grid = build_spatial_grid(inner_hexes, cell_size=3.0)

    # Check containment and overlaps
    total_penalty = 0
    for i, hex_vertices in enumerate(inner_hexes):
        hex_poly = Polygon(hex_vertices)
        if not check_containment(hex_poly, outer_hex):
            total_penalty += 1000  # Large penalty for containment violations

        # Check overlap with other hexes using spatial grid
        candidates = get_candidate_overlaps(inner_hexes, spatial_grid, i, cell_size=3.0)
        for j in candidates:
            if i < j:  # Only check each pair once
                if check_overlap(hex_vertices, inner_hexes[j]):
                    total_penalty += 10000  # Large penalty for overlaps

    # Calculate combined score (negative since we want to minimize)
    inv_radius = 1.0 / outer_radius if outer_radius > 0 else 0
    return -inv_radius + total_penalty

def generate_initial_guess():
    """Generate a good initial guess using geometric pattern matching"""
    # Generate more sophisticated initial configuration based on geometric principles
    # Create a pattern that mimics efficient packing with central symmetry
    positions = []
    rotations = []
    
    # Central hexagon
    positions.append([0, 0])
    rotations.append(0)
    
    # First ring - 6 hexagons at distance 2
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # Skip last to avoid duplication
    for angle in angles:
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle)
        positions.append([x, y])
        rotations.append(0)  # No rotation for now
        
    # Second ring - 4 hexagons at distance 3, with strategic placement
    angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 7)[:-1]  # Offset by pi/6 for better packing
    for i, angle in enumerate(angles):
        if i >= 4:  # Only place 4 hexagons in second ring to get exactly 11
            break
        x = 3 * np.cos(angle)
        y = 3 * np.sin(angle)
        positions.append([x, y])
        rotations.append(0)
    
    # Ensure exactly 11 positions/rotations
    positions = positions[:11]
    rotations = rotations[:11]
    
    # Add some randomness to positions and rotations to avoid local minima
    for i in range(len(positions)):
        positions[i][0] += np.random.normal(0, 0.1)
        positions[i][1] += np.random.normal(0, 0.1)
        rotations[i] += np.random.normal(0, 10)
    
    # Flatten for parameter vector
    flat_params = []
    for pos in positions:
        flat_params.extend(pos)
    flat_params.extend(rotations)
    flat_params.append(5.0)  # Initial outer radius guess
    
    return np.array(flat_params)

def local_optimization_step(initial_params, max_iter=50):
    """Perform local optimization on a good solution using scipy minimize"""
    def objective(params):
        return evaluate_solution(params)
    
    # Use L-BFGS-B method which handles bounds well
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=[(-10, 10)] * 22 + [(0, 360)] * 11 + [(1, 20)],  # bounds for each param
        options={'maxiter': max_iter, 'ftol': 1e-6},
        jac=False  # Since we're using finite differences for derivatives
    )
    
    return result.x if result.success else initial_params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid evolutionary approach combining global search with local optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate initial guess
    initial_guess = generate_initial_guess()
    
    # Define bounds for optimization
    bounds = []
    # Position bounds (-10, 10) for each of 11 hexagons
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10)])
    # Rotation bounds (0, 360) for each of 11 hexagons
    for _ in range(11):
        bounds.append((0, 360))
    # Outer radius bounds (1, 20)
    bounds.append((1, 20))
    
    # Phase 1: Global evolutionary optimization using differential evolution
    result = differential_evolution(
        evaluate_solution,
        bounds,
        maxiter=80,  # Reduced iterations to allow for local optimization
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=False
    )
    
    # Phase 2: Local optimization on the best solution found
    best_solution = result.x
    refined_solution = local_optimization_step(best_solution, max_iter=50)
    
    # Extract results
    inner_positions = refined_solution[:22].reshape(-1, 2)
    inner_rotations = refined_solution[22:33]
    outer_radius = refined_solution[33]

    # Create final data arrays
    inner_hex_data = np.column_stack([
        inner_positions[:, 0],
        inner_positions[:, 1],
        inner_rotations
    ])

    outer_hex_data = np.array([0, 0, 0])  # Centered at origin

    end_time = time.time()
    eval_time = end_time - start_time

    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END