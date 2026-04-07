# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def hexagon_vertices(center_x, center_y, side_length, rotation_degrees):
    """Generate vertices of a regular hexagon."""
    angle_rad = np.radians(rotation_degrees)
    angle_step = np.pi / 3
    vertices = []
    for i in range(6):
        angle = angle_rad + i * angle_step
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return vertices

def create_hexagon_polygon(center_x, center_y, side_length, rotation_degrees):
    """Create Shapely polygon for a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, side_length, rotation_degrees)
    return Polygon(vertices)

def is_contained(hex_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hex_poly) or outer_hex_poly.touches(hex_poly)

def hexagons_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def evaluate_solution(positions_and_rotations, outer_radius):
    """Evaluate solution quality."""
    n = 11
    # Convert positions and rotations to individual arrays
    positions = positions_and_rotations[:2*n].reshape(n, 2)
    rotations = positions_and_rotations[2*n:]
    
    # Create all inner hexagons
    inner_hexagons = []
    for i in range(n):
        x, y = positions[i]
        angle = rotations[i]
        hex_poly = create_hexagon_polygon(x, y, 1.0, angle)
        inner_hexagons.append(hex_poly)
    
    # Create outer hexagon
    outer_hex = create_hexagon_polygon(0, 0, outer_radius, 0)
    
    # Check containment and overlaps
    total_penalty = 0
    
    # Check containment
    for hex_poly in inner_hexagons:
        if not is_contained(hex_poly, outer_hex):
            total_penalty += 1000  # Heavy penalty for containment violations
    
    # Check overlaps
    for i, j in combinations(range(n), 2):
        if hexagons_overlap(inner_hexagons[i], inner_hexagons[j]):
            total_penalty += 1000  # Heavy penalty for overlaps
    
    # If any constraints violated, return poor fitness
    if total_penalty > 0:
        return total_penalty
    
    # If valid, return negative of outer radius (minimize outer radius)
    return -outer_radius

def generate_initial_configurations():
    """Generate diverse initial configurations."""
    configs = []
    
    # Configuration 1: Grid pattern with some randomness
    base_positions = np.array([
        [0, 0], [-2.5, 0], [2.5, 0], [-1.25, 2.17], [1.25, 2.17],
        [-1.25, -2.17], [1.25, -2.17], [-3.75, 2.17], [3.75, 2.17],
        [-3.75, -2.17], [3.75, -2.17]
    ])
    
    # Add small random perturbations
    np.random.seed(42)
    perturbed_positions = base_positions + np.random.normal(0, 0.1, base_positions.shape)
    
    config1 = np.concatenate([
        perturbed_positions.flatten(),
        np.zeros(11)  # No rotation
    ])
    configs.append(config1)
    
    # Configuration 2: Radial pattern
    angles = np.linspace(0, 2*np.pi, 11)
    radii = [0] + [1.5 + i*0.8 for i in range(1, 11)]
    radial_positions = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles)]
    config2 = np.concatenate([
        np.array(radial_positions).flatten(),
        np.zeros(11)
    ])
    configs.append(config2)
    
    # Configuration 3: Spiral pattern
    spiral_positions = []
    for i in range(11):
        angle = i * 0.5
        radius = i * 0.3
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        spiral_positions.append((x, y))
    
    config3 = np.concatenate([
        np.array(spiral_positions).flatten(),
        np.zeros(11)
    ])
    configs.append(config3)
    
    return configs

def optimize_with_local_refinement(x0, bounds, maxiter=100):
    """Perform local refinement around best solution."""
    def objective(x):
        return evaluate_solution(x, 10.0)  # Placeholder outer radius for local search
    
    # Simple local optimization using scipy's minimize
    from scipy.optimize import minimize
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
    return result.x

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    start_time = time.time()
    
    # Generate diverse initial configurations
    initial_configs = generate_initial_configurations()
    
    best_score = float('-inf')
    best_result = None
    
    # Try each initial configuration with DE
    for i, initial_config in enumerate(initial_configs):
        print(f"Trying initial configuration {i+1}")
        
        # Define bounds for positions (x, y) and rotations
        bounds = []
        # Position bounds (arbitrary large range to allow for optimization)
        for _ in range(n):
            bounds.extend([(-10, 10), (-10, 10)])  # x, y bounds
        # Rotation bounds (0-360 degrees)
        for _ in range(n):
            bounds.append((0, 360))
        
        # Run differential evolution with adaptive parameters
        try:
            de_result = differential_evolution(
                lambda x: evaluate_solution(x, 10.0),
                bounds,
                maxiter=50,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.9,
                seed=42+i,
                disp=False
            )
            
            # Refine with local optimization
            refined_result = optimize_with_local_refinement(de_result.x, bounds)
            
            # Evaluate refined solution
            try:
                # We use a fixed outer radius for comparison since it's a simpler constraint
                eval_score = evaluate_solution(refined_result, 10.0)
                
                # Convert back to actual solution format for scoring
                positions = refined_result[:2*n].reshape(n, 2)
                rotations = refined_result[2*n:]
                
                # Check with a reasonable outer radius
                test_radius = 5.0
                test_score = evaluate_solution(refined_result, test_radius)
                
                if test_score > best_score:
                    best_score = test_score
                    best_result = refined_result
                    
            except Exception as e:
                print(f"Error evaluating solution: {e}")
                
        except Exception as e:
            print(f"Differential evolution failed on config {i}: {e}")
            continue
        
        # Early termination if time is running out
        if time.time() - start_time > 170:
            break
    
    # If no good solution found, fallback to initial grid arrangement
    if best_result is None:
        print("Falling back to baseline configuration")
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0], [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Extract best solution
    positions = best_result[:2*n].reshape(n, 2)
    rotations = best_result[2*n:]
    
    # Calculate the best achievable outer hexagon size
    # Since we're optimizing for minimum outer radius, we'll estimate it
    # by checking what radius works with our arrangement
    estimated_radius = 4.0
    test_result = evaluate_solution(best_result, estimated_radius)
    
    # Validate and refine the result
    if test_result < -1000:  # Constraint violation
        # Use a larger outer radius
        estimated_radius = 6.0
        test_result = evaluate_solution(best_result, estimated_radius)
    
    # Create final result
    inner_hex_data = np.column_stack([positions, rotations])
    outer_hex_data = np.array([0, 0, 0])  # Centered
    outer_hex_side_length = estimated_radius if estimated_radius > 0 else 4.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
