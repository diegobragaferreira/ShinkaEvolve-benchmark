# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
import time
from itertools import product

def create_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Create vertices of a regular hexagon given center, rotation, and side length"""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def check_hexagon_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon with buffer to avoid floating point issues"""
    outer_polygon = Polygon(outer_hex_vertices)
    # Use a small buffer to prevent floating point precision issues
    buffered_outer = outer_polygon.buffer(1e-6)
    for vertex in hex_vertices:
        if not buffered_outer.contains(Point(vertex[0], vertex[1])):
            return False
    return True

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely with buffer for precision"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    # Use a small buffer to prevent floating point precision issues
    buffered_poly1 = poly1.buffer(1e-6)
    buffered_poly2 = poly2.buffer(1e-6)
    return buffered_poly1.intersects(buffered_poly2)

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum outer hexagon side length that contains all inner hexagons using tight bounding"""
    if len(inner_hex_data) == 0:
        return 1000

    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)

    if not all_vertices:
        return 1000

    # Calculate tight bounding box
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Calculate the diagonal of the bounding box
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    
    # For a hexagon, we need to ensure all points fit within the hexagon's circumscribed circle
    diagonal = math.sqrt(bbox_width**2 + bbox_height**2)
    
    # For a regular hexagon, the relationship between circumradius and side length:
    side_length = diagonal / math.sqrt(3) * 1.1
    
    return side_length

def generate_hexagonal_lattice_config():
    """Generate a hexagonal lattice configuration which often provides good starting points"""
    # Base hexagonal lattice with 11 points
    # We'll place them in a pattern that resembles a hexagonal close-packed arrangement
    config = []
    
    # Center point
    config.append([0, 0, 0])
    
    # First ring of 6 points around center
    for i in range(6):
        angle = i * math.pi / 3
        x = math.cos(angle) * 2.0
        y = math.sin(angle) * 2.0
        config.append([x, y, 0])
    
    # Second ring of 4 points (filling gaps in first ring)
    # This creates a pattern closer to optimal hexagonal packing
    angles = [math.pi/6, 3*math.pi/6, 5*math.pi/6, 7*math.pi/6]  # 4 strategic angles
    for i, angle in enumerate(angles):
        x = math.cos(angle) * 3.0
        y = math.sin(angle) * 3.0
        config.append([x, y, 0])
    
    # Trim to exactly 11 points if needed
    if len(config) > 11:
        config = config[:11]
        
    return np.array(config)

def generate_triangular_pattern_config():
    """Generate a triangular pattern configuration"""
    # Arrange in triangular lattice with 11 points
    config = []
    
    # Place in triangular formation
    # Row 1: 1 point
    config.append([0, 0, 0])
    
    # Row 2: 2 points
    config.append([-1.5, 0, 0])
    config.append([1.5, 0, 0])
    
    # Row 3: 3 points
    config.append([0, 2.17, 0])
    config.append([-1.5, 2.17, 0])
    config.append([1.5, 2.17, 0])
    
    # Row 4: 4 points
    config.append([-3.0, 0, 0])
    config.append([3.0, 0, 0])
    config.append([-1.5, -2.17, 0])
    config.append([1.5, -2.17, 0])
    
    # Trim to exactly 11 points if needed
    if len(config) > 11:
        config = config[:11]
        
    return np.array(config)

def generate_spiral_config():
    """Generate a spiral configuration"""
    config = []
    
    # Start with center
    config.append([0, 0, 0])
    
    # Generate spiral points
    for i in range(1, 11):
        angle = i * 0.6  # Spiral angle
        radius = i * 0.8  # Increasing radius  
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        config.append([x, y, 0])
    
    return np.array(config)

def generate_fibonacci_spiral_config():
    """Generate a Fibonacci spiral configuration for even better distribution"""
    config = []
    
    # Center point
    config.append([0, 0, 0])
    
    # Golden ratio spiral points
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(1, 11):
        angle = i * golden_ratio  # Golden angle increment
        radius = i * 0.7  # Radius grows with index
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        config.append([x, y, 0])
    
    return np.array(config)

def generate_initial_configs():
    """Generate multiple initial configurations to explore different starting points"""
    configs = []
    
    # Add different types of configurations
    configs.append(generate_hexagonal_lattice_config())
    configs.append(generate_triangular_pattern_config()) 
    configs.append(generate_spiral_config())
    configs.append(generate_fibonacci_spiral_config())
    
    # Add slightly perturbed versions to add diversity
    for base_config in configs.copy():
        # Add noise to each configuration
        perturbed = base_config.copy().astype(float)
        for i in range(1, len(perturbed)):  # Skip center point for slight variation
            perturbed[i][0] += np.random.normal(0, 0.3)
            perturbed[i][1] += np.random.normal(0, 0.3)
            perturbed[i][2] += np.random.normal(0, 10)
        configs.append(perturbed)
    
    return configs

def get_constraint_violation(hex_data, outer_side_length, verbose=False):
    """Calculate constraint violation for a given configuration"""
    outer_hex_vertices = create_hexagon_vertices(0, 0, 0, outer_side_length)
    
    # Check containment
    containment_violations = 0
    for i in range(len(hex_data)):
        center_x, center_y, angle = hex_data[i]
        inner_hex_vertices = create_hexagon_vertices(center_x, center_y, angle)
        if not check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
            containment_violations += 1
    
    # Check overlaps
    overlap_violations = 0
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            center_x1, center_y1, angle1 = hex_data[i]
            center_x2, center_y2, angle2 = hex_data[j]

            hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
            hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)

            if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                overlap_violations += 1
    
    if verbose:
        print(f"Containment violations: {containment_violations}, Overlap violations: {overlap_violations}")
    
    return containment_violations + overlap_violations

def evaluate_objective_with_constraints(hex_data):
    """Evaluate objective function with hard constraint penalties"""
    outer_side_length = calculate_outer_hex_side_length(hex_data)
    
    # Check constraints
    try:
        # Check if constraints are violated
        constraint_violation = get_constraint_violation(hex_data, outer_side_length)
        
        if constraint_violation > 0:
            # Apply heavy penalty for constraint violations
            return 1.0 / outer_side_length - 10000 * constraint_violation
        
        # No constraint violations - return objective value
        return 1.0 / outer_side_length
        
    except Exception as e:
        # Return very poor objective for invalid solutions
        return -10000

def optimize_single_config(initial_config):
    """Optimize a single configuration using L-BFGS-B"""
    def objective(params):
        # Reshape flat array back to hex data
        hex_data = params.reshape(-1, 3)
        return -evaluate_objective_with_constraints(hex_data)  # Minimize negative to maximize
    
    # Flatten initial configuration
    x0 = initial_config.flatten()
    
    # Define bounds for optimization
    bounds = []
    for i in range(len(x0)):
        if i % 3 == 0:  # x coordinate
            bounds.append((-6.0, 6.0))
        elif i % 3 == 1:  # y coordinate
            bounds.append((-6.0, 6.0))
        else:  # angle
            bounds.append((0.0, 360.0))
    
    # Optimize using L-BFGS-B
    try:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        # Extract optimized solution
        optimized_solution = result.x.reshape(-1, 3)
        fitness = -result.fun  # Convert back from negative
        
        return optimized_solution, fitness
        
    except Exception as e:
        # Return original if optimization fails
        return initial_config, evaluate_objective_with_constraints(initial_config)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses deterministic geometric optimization with multiple starting configurations.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random_seed = 42
    random.seed(random_seed)
    
    # Generate multiple initial configurations
    initial_configs = generate_initial_configs()
    
    best_fitness = -float('inf')
    best_config = None
    best_side_length = float('inf')
    
    # Try optimization on each configuration
    for i, config in enumerate(initial_configs):
        try:
            # Optimize this configuration
            optimized_config, fitness = optimize_single_config(config)
            
            # Evaluate final result with constraints checked
            final_fitness = evaluate_objective_with_constraints(optimized_config)
            
            # Validate final result
            side_length = calculate_outer_hex_side_length(optimized_config)
            constraint_violation = get_constraint_violation(optimized_config, side_length)
            
            if final_fitness > best_fitness and constraint_violation == 0:
                best_fitness = final_fitness
                best_config = optimized_config
                best_side_length = side_length
                
        except Exception as e:
            continue
    
    # If no good configuration was found, fallback to a simple grid arrangement
    if best_config is None or best_fitness <= 0:
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_side_length = 8
        return inner_hex_data, np.array([0, 0, 0]), outer_hex_side_length
    
    return best_config, np.array([0, 0, 0]), best_side_length

# EVOLVE-BLOCK-END