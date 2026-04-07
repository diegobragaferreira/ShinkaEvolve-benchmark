# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.validation import make_valid
import time
from collections import defaultdict


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
    """Build a spatial grid for fast collision detection"""
    grid = defaultdict(list)
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
                grid[(row, col)].append(i)

    return grid


def get_candidate_overlaps(spatial_grid, hex_idx, hex_vertices, cell_size=3.0):
    """Get candidate hexagon indices that might overlap with hex_idx"""
    # Get the bounding box of the hexagon we're checking
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


def evaluate_solution(params, outer_hex_radius=None):
    """Evaluate a solution and return negative combined score (for minimization)"""
    # Parse parameters
    inner_positions = params[:22].reshape(-1, 2)  # x,y pairs
    inner_rotations = params[22:33]  # 11 rotations
    if outer_hex_radius is None:
        outer_radius = params[33]  # outer hex radius
    else:
        outer_radius = outer_hex_radius

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

    # Build spatial grid for fast overlap detection
    spatial_grid = build_spatial_grid(inner_hexes, cell_size=3.0)

    # Check containment and overlaps
    total_penalty = 0
    for i, hex_vertices in enumerate(inner_hexes):
        hex_poly = Polygon(hex_vertices)
        if not check_containment(hex_poly, outer_hex):
            total_penalty += 1000  # Large penalty for containment violations

        # Check overlap with other hexes using spatial grid
        candidates = get_candidate_overlaps(spatial_grid, i, hex_vertices, cell_size=3.0)
        for j in candidates:
            if i < j:  # Only check each pair once
                if check_overlap(hex_vertices, inner_hexes[j]):
                    total_penalty += 10000  # Large penalty for overlaps

    # Calculate combined score (negative since we want to minimize)
    inv_radius = 1.0 / outer_radius if outer_radius > 0 else 0
    return -inv_radius + total_penalty


def generate_smart_initial_guess():
    """Generate a smart initial guess for hexagon packing"""
    # Use a more sophisticated pattern based on known optimal patterns
    # This follows a pattern similar to what's known to work well for 11 hexagons
    
    # Pattern inspired by dense packings: 3 layers with specific placement
    positions = []
    rotations = []

    # Layer 1: center hexagon
    positions.append([0, 0])
    rotations.append(0)

    # Layer 2: Surrounding hexagons, arranged more carefully
    # Use hexagonal lattice with optimized spacing
    for i in range(6):
        angle = np.radians(i * 60)
        # Place at distance of ~2.17 units (2*sqrt(3)/2) for proper contact
        x = 2.17 * np.cos(angle)
        y = 2.17 * np.sin(angle)
        positions.append([x, y])
        rotations.append(0)

    # Layer 3: Second ring hexagons
    for i in range(6):
        angle = np.radians(i * 60 + 30)  # Offset by 30 degrees
        # Place at distance of ~3.25 units to make room for better packing
        x = 3.25 * np.cos(angle)
        y = 3.25 * np.sin(angle)
        positions.append([x, y])
        rotations.append(0)

    # Adjust to ensure exactly 11 hexagons, and add variation to rotations
    positions = positions[:11]
    rotations = rotations[:11]
    
    # Add small variations to rotations to break symmetry
    for i in range(len(rotations)):
        rotations[i] += np.random.uniform(-10, 10)

    # Flatten for parameter vector
    flat_params = []
    for pos in positions:
        flat_params.extend(pos)
    flat_params.extend(rotations)
    flat_params.append(6.0)  # Initial outer radius guess (reduced from default)

    return np.array(flat_params)


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization with spatial acceleration to find a better solution than the simple grid arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate multiple initial guesses for better chance of finding global optimum
    initial_guesses = []
    
    # Generate several differently initialized solutions
    for i in range(5):
        np.random.seed(42 + i)
        initial_guesses.append(generate_smart_initial_guess())
    
    best_result = None
    best_score = float('inf')
    best_params = None
    
    # Run optimization on each initial guess
    for i, initial_guess in enumerate(initial_guesses):
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

        # Run optimization with reduced iterations and population size to save time
        # But with better parameter tuning for convergence
        try:
            result = differential_evolution(
                evaluate_solution,
                bounds,
                maxiter=80,  # Reduced iterations to stay within time budget
                popsize=12,  # Smaller population size for faster optimization
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42 + i,
                disp=False
            )
            
            # Check if this is a better solution
            current_score = evaluate_solution(result.x)
            if current_score < best_score:
                best_score = current_score
                best_result = result
                best_params = result.x.copy()
                
        except Exception as e:
            continue

    # If we still don't have a good result, use fallback
    if best_result is None:
        # Use the most basic initial guess
        initial_guess = generate_smart_initial_guess()
        bounds = []
        for _ in range(11):
            bounds.extend([(-10, 10), (-10, 10)])
        for _ in range(11):
            bounds.append((0, 360))
        bounds.append((1, 20))
        
        try:
            result = differential_evolution(
                evaluate_solution,
                bounds,
                maxiter=80,
                popsize=12,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            best_params = result.x.copy()
        except:
            # Final fallback
            best_params = generate_smart_initial_guess()
    
    # Extract results
    inner_positions = best_params[:22].reshape(-1, 2)
    inner_rotations = best_params[22:33]
    outer_radius = best_params[33]

    # Create final data arrays
    inner_hex_data = np.column_stack([
        inner_positions[:, 0],
        inner_positions[:, 1],
        inner_rotations
    ])

    outer_hex_data = np.array([0, 0, 0])  # Centered at origin

    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END