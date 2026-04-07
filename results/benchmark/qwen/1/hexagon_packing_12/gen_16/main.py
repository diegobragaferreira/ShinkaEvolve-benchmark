# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time

def create_hexagon_vertices(center_x, center_y, side_length, rotation_degrees):
    """Create vertices of a regular hexagon given center, side length, and rotation."""
    angle_rad = np.radians(rotation_degrees)
    # Vertices of a regular hexagon with side length 1, centered at origin
    hex_vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = side_length * np.cos(angle)
        y = side_length * np.sin(angle)
        hex_vertices.append((x, y))
    
    # Translate to center
    translated_vertices = [(x + center_x, y + center_y) for x, y in hex_vertices]
    return translated_vertices

def check_containment(hexagon_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are contained in outer hexagon."""
    inner_poly = Polygon(hexagon_vertices)
    outer_poly = Polygon(outer_hex_vertices)
    return outer_poly.contains(inner_poly)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_penalty_for_violations(inner_hex_data, outer_hex_data, outer_hex_side_length):
    """Calculate penalty for constraint violations."""
    penalty = 0
    
    # Create outer hexagon vertices
    outer_hex_vertices = create_hexagon_vertices(outer_hex_data[0], outer_hex_data[1], outer_hex_side_length, outer_hex_data[2])
    
    # Check containment for all inner hexagons
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        hex_vertices = create_hexagon_vertices(center_x, center_y, 1, rotation)
        
        if not check_containment(hex_vertices, outer_hex_vertices):
            penalty += 1000000  # Large penalty for containment violation
            
    # Check non-overlap between all pairs of inner hexagons
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center_x1, center_y1, rotation1 = inner_hex_data[i]
            center_x2, center_y2, rotation2 = inner_hex_data[j]
            
            hex1_vertices = create_hexagon_vertices(center_x1, center_y1, 1, rotation1)
            hex2_vertices = create_hexagon_vertices(center_x2, center_y2, 1, rotation2)
            
            if check_overlap(hex1_vertices, hex2_vertices):
                penalty += 1000000  # Large penalty for overlap violation
                
    return penalty

def evaluate_packing(inner_hex_data, outer_hex_data, outer_hex_side_length):
    """Evaluate the packing quality."""
    # Calculate penalty for constraint violations  
    penalty = calculate_penalty_for_violations(inner_hex_data, outer_hex_data, outer_hex_side_length)
    
    # If there are any violations, return a very poor score
    if penalty > 0:
        return penalty
    
    # Return negative inverse side length (since we want to maximize 1/outer_hex_side_length)  
    return -1.0 / outer_hex_side_length

def get_initial_population():
    """Generate initial population with symmetric and diverse configurations."""
    population = []
    
    # Configuration 1: Star-like arrangement
    config1 = np.array([
        [0, 0, 0],      # center
        [0, 2.5, 0],    # top
        [0, -2.5, 0],   # bottom
        [2.5, 0, 0],    # right
        [-2.5, 0, 0],   # left
        [1.25, 1.25, 0], # top-right
        [-1.25, 1.25, 0], # top-left
        [1.25, -1.25, 0], # bottom-right
        [-1.25, -1.25, 0], # bottom-left
        [2.17, 2.17, 0], # far top-right
        [-2.17, 2.17, 0], # far top-left
        [2.17, -2.17, 0], # far bottom-right
    ])
    
    # Configuration 2: Ring arrangement
    config2 = np.array([
        [0, 0, 0],      # center
        [0, 2.0, 0],    # top  
        [1.73, 1.0, 0], # top-right
        [1.73, -1.0, 0], # bottom-right
        [0, -2.0, 0],   # bottom
        [-1.73, -1.0, 0], # bottom-left
        [-1.73, 1.0, 0], # top-left
        [0, 2.0, 30],   # top rotated
        [1.73, 1.0, 30], # top-right rotated
        [1.73, -1.0, 30], # bottom-right rotated
        [0, -2.0, 30],  # bottom rotated
        [-1.73, -1.0, 30], # bottom-left rotated
    ])
    
    # Configuration 3: Zig-zag pattern with rotations
    config3 = np.array([
        [0, 0, 0],      # center
        [1.5, 0, 0],    # right
        [0, 1.5, 0],    # top
        [-1.5, 0, 0],   # left
        [0, -1.5, 0],   # bottom
        [1.5, 1.5, 0],  # top-right
        [1.5, -1.5, 0], # bottom-right
        [-1.5, 1.5, 0], # top-left
        [-1.5, -1.5, 0], # bottom-left
        [3.0, 0, 0],    # far right
        [0, 3.0, 0],    # far top
        [0, -3.0, 0],   # far bottom
    ])
    
    population.extend([config1, config2, config3])
    
    # Add some random variations
    for _ in range(5):
        var_config = config1.copy()
        for i in range(12):
            var_config[i][0] += (np.random.rand() - 0.5) * 0.5
            var_config[i][1] += (np.random.rand() - 0.5) * 0.5
            var_config[i][2] += (np.random.rand() - 0.5) * 30
        population.append(var_config)
    
    return population

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Define bounds for optimization
    # Inner hexagons: (x, y, rotation) for 12 hexagons
    bounds = []
    # Bounds for x and y positions (-10 to 10)
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
    
    # Bounds for outer hexagon (center and rotation)
    bounds.extend([(-5, 5), (-5, 5), (-180, 180)])
    # Bounds for outer hexagon side length (minimum reasonable value to maximum reasonable value)  
    bounds.extend([(1, 15)])

    def objective_function(x):
        # Extract inner hexagon parameters
        inner_params = x[:36].reshape(12, 3)
        # Extract outer hexagon parameters  
        outer_params = x[36:39]
        outer_side_length = x[39]
        
        return evaluate_packing(inner_params, outer_params, outer_side_length)

    # Get initial population
    initial_pop = get_initial_population()
    
    # Run optimization with multiple starting points
    best_score = float('inf')
    best_result = None
    
    # Try several different starting configurations
    for i, config in enumerate(initial_pop):
        # Flatten the configuration into the expected format for optimization
        x0 = np.concatenate([
            config.flatten(), 
            np.array([0, 0, 0, 4.5])  # outer hex center at origin, rotation 0, side length 4.5
        ])
        
        try:
            # Use differential evolution with a small population and fewer iterations for faster execution
            result = differential_evolution(
                objective_function, 
                bounds,
                seed=42+i,  # Different seed for each run
                maxiter=50,  # Reduced iterations for faster execution under 180s limit
                popsize=10,  # Smaller population
                mutation=(0.5, 1),  # Standard mutation parameters
                recombination=0.7,  # Standard recombination
                atol=1e-6,
                rtol=1e-6,
                disp=False
            )
            
            # Check if this result is better
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
                
        except Exception as e:
            continue  # Skip failing optimization attempts
    
    # Extract best solution
    if best_result is not None:
        best_x = best_result.x
        inner_hex_data = best_x[:36].reshape(12, 3)
        outer_hex_data = best_x[36:39]
        outer_hex_side_length = best_x[39]
    else:
        # Fallback to a decent initial configuration if optimization fails
        inner_hex_data = np.array([
            [0, 0, 0],
            [0, 2.5, 0],
            [0, -2.5, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 1.25, 0],
            [-1.25, 1.25, 0],
            [1.25, -1.25, 0],
            [-1.25, -1.25, 0],
            [2.17, 2.17, 0],
            [-2.17, 2.17, 0],
            [2.17, -2.17, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 4.5

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
