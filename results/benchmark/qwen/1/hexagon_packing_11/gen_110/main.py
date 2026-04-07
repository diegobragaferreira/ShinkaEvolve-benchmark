# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.validation import make_valid
import time
from itertools import combinations


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

    # Check containment and overlaps
    total_penalty = 0
    for i, hex_vertices in enumerate(inner_hexes):
        hex_poly = Polygon(hex_vertices)
        if not check_containment(hex_poly, outer_hex):
            total_penalty += 1000  # Large penalty for containment violations

        # Check overlap with other hexes
        for j in range(i+1, len(inner_hexes)):
            if check_overlap(hex_vertices, inner_hexes[j]):
                total_penalty += 10000  # Large penalty for overlaps

    # Calculate combined score (negative since we want to minimize)
    inv_radius = 1.0 / outer_radius if outer_radius > 0 else 0
    return -inv_radius + total_penalty


def generate_diverse_initial_populations():
    """Generate multiple diverse initial population configurations."""
    populations = []
    
    # Configuration 1: Hexagonal lattice arrangement
    base_config = [
        (0, 0, 0),  # center
        (-2.5, 0, 0),  # left
        (2.5, 0, 0),  # right
        (-1.25, 2.17, 0),  # top-left
        (1.25, 2.17, 0),  # top-right
        (-1.25, -2.17, 0),  # bottom-left
        (1.25, -2.17, 0),  # bottom-right
        (-3.75, 2.17, 0),  # far top-left
        (3.75, 2.17, 0),  # far top-right
        (-3.75, -2.17, 0),  # far bottom-left
        (3.75, -2.17, 0),  # far bottom-right
    ]
    
    # Configuration 2: Spiral pattern
    spiral_config = [
        (0, 0, 0),
        (2, 0, 0),
        (1, 1.732, 0),
        (-1, 1.732, 0),
        (-2, 0, 0),
        (-1, -1.732, 0),
        (1, -1.732, 0),
        (3, 0, 0),
        (2, 2.17, 0),
        (-2, 2.17, 0),
        (-3, 0, 0)
    ]
    
    # Configuration 3: Clustered arrangement
    cluster_config = [
        (0, 0, 0),
        (1.5, 0, 0),
        (-1.5, 0, 0),
        (0, 1.5, 0),
        (0, -1.5, 0),
        (1.5, 1.5, 0),
        (-1.5, 1.5, 0),
        (1.5, -1.5, 0),
        (-1.5, -1.5, 0),
        (3, 0, 0),
        (0, 3, 0)
    ]
    
    # Configuration 4: Cross pattern
    cross_config = [
        (0, 0, 0),
        (0, 2.5, 0),
        (0, -2.5, 0),
        (2.5, 0, 0),
        (-2.5, 0, 0),
        (1.25, 2.17, 0),
        (-1.25, 2.17, 0),
        (1.25, -2.17, 0),
        (-1.25, -2.17, 0),
        (3.75, 0, 0),
        (0, 3.75, 0)
    ]
    
    # Configuration 5: Random-like but structured
    random_like_config = [
        (0, 0, 0),
        (2, 1, 30),
        (-1.5, 2, 60),
        (-2.5, -1, 90),
        (1, -2, 120),
        (3, 1.5, 150),
        (-2, 1.5, 180),
        (1.5, -2.5, 210),
        (-1.5, -2.5, 240),
        (3.5, -1, 270),
        (0, 3.5, 300)
    ]
    
    configs = [base_config, spiral_config, cluster_config, cross_config, random_like_config]
    
    for config in configs:
        guess = []
        for x, y, angle in config:
            guess.extend([x, y, angle])
        guess.append(6.0)  # Initial outer radius
        populations.append(np.array(guess))
    
    return populations


def adaptive_mutation_schedule(iteration, maxiter):
    """Adaptive mutation rate with improved scheduling for better exploration-exploitation balance"""
    # Use exponential decay for better exploration in early stages
    # and more precise exploitation in later stages
    if maxiter <= 0:
        return 0.5
    progress = iteration / maxiter

    # Start with high mutation (0.8) and exponentially decay to 0.1
    # This provides better exploration in early iterations
    return 0.1 + 0.7 * np.exp(-3 * progress)


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization to find a better solution than the simple grid arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Set up bounds for optimization
    # Each inner hexagon has (x, y, angle) - 3 parameters each
    # Outer hexagon has radius R - 1 parameter
    # Total parameters: 3*11 + 1 = 34

    # Bounds for positions (x, y): -10 to 10
    pos_bounds = [(-10, 10)] * 22  # 11 hexagons * 2 positions each

    # Bounds for angles: 0 to 360 degrees
    angle_bounds = [(0, 360)] * 11

    # Bounds for outer hexagon radius: 1 to 20 (reasonable range)
    radius_bound = [(1, 20)]

    # Combine all bounds
    bounds = pos_bounds + angle_bounds + radius_bound

    # Generate diverse initial populations
    initial_populations = generate_diverse_initial_populations()
    
    best_result = None
    best_fitness = float('inf')
    
    # Time limit enforcement
    start_time = time.time()
    
    # Try each initial population
    for i, initial_guess in enumerate(initial_populations):
        if time.time() - start_time > 170:  # Leave some buffer
            break
            
        # Run optimization with adaptive mutation
        result = differential_evolution(
            evaluate_solution,
            bounds,
            maxiter=300,
            popsize=20,
            seed=42+i,  # Different seed for each run
            callback=lambda x, convergence: print(f"Population {i+1}: Best fitness: {-1.0/x[-1]:.6f} after {time.time()-start_time:.2f}s") if time.time() - start_time < 175 else None,
            disp=False,
            mutation=(0.5, 1.0)  # Default mutation range
        )
        
        # Update best result if this one is better
        if result.fun < best_fitness:
            best_fitness = result.fun
            best_result = result
    
    # If we still don't have a good result, do a fallback optimization
    if best_result is None:
        # Use default initial guess
        initial_guess = []
        initial_guess.extend([0, 0, 0])  # Center
        hex_coords = [
            (1.732, 0, 0),
            (-1.732, 0, 0),
            (0.866, 1.5, 0),
            (-0.866, 1.5, 0),
            (0.866, -1.5, 0),
            (-0.866, -1.5, 0),
            (2.598, 1.5, 0),
            (-2.598, 1.5, 0),
            (2.598, -1.5, 0),
            (-2.598, -1.5, 0),
            (0, 3, 0)
        ]
        for x, y, angle in hex_coords:
            initial_guess.extend([x, y, angle])
        initial_guess.append(5.0)
        
        result = differential_evolution(
            evaluate_solution,
            bounds,
            maxiter=300,
            popsize=20,
            seed=42,
            callback=lambda x, convergence: print(f"Fallback: Best fitness: {-1.0/x[-1]:.6f} after {time.time()-start_time:.2f}s") if time.time() - start_time < 175 else None,
            disp=False,
            mutation=(0.5, 1.0)
        )
        best_result = result
    
    # Extract results
    best_params = best_result.x
    inner_positions_angles = best_params[:-1].reshape(-1, 3)
    outer_radius = best_params[-1]

    # Convert to desired output format
    inner_hex_data = inner_positions_angles.copy()

    # Create outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin

    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END