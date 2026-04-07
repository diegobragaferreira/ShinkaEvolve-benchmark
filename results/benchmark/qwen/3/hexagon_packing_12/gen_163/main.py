# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from scipy.spatial.distance import cdist
import random
from numba import jit, prange
from joblib import Parallel, delayed

# Constants
UNIT_HEX_RADIUS = 1.0
MAX_EVAL_TIME = 180.0
TARGET_RATIO = 0.2537

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius (numba-compiled)"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

def create_regular_hexagon(center=(0,0), side_length=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    x = center[0] + side_length * np.cos(angles)
    y = center[1] + side_length * np.sin(angles)
    return Polygon(list(zip(x, y)))

def hexagon_to_polygon(x, y, angle_deg, radius=1.0):
    """Convert hexagon parameters to shapely polygon"""
    vertices = get_hexagon_vertices(x, y, angle_deg, radius)
    return Polygon(vertices)

def check_overlap_fast(hex1_poly, hex2_poly):
    """Fast overlap check using Shapely with buffer for numerical stability"""
    return hex1_poly.buffer(1e-10).intersects(hex2_poly.buffer(1e-10)) and not hex1_poly.touches(hex2_poly)

def check_containment(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained within outer hexagon"""
    return outer_hex.contains(inner_hex)

def compute_outer_hexagon_radius(inner_hex_data):
    """Compute minimum outer hexagon radius that contains all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 0.0
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return 0.0
    
    # Compute centroid
    centroid_x = np.mean([v[0] for v in all_vertices])
    centroid_y = np.mean([v[1] for v in all_vertices])
    
    # Find maximum distance from centroid to any vertex
    max_distance = 0.0
    for x, y in all_vertices:
        distance = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        max_distance = max(max_distance, distance)
    
    # Add buffer for hexagon radius calculation
    return max_distance + UNIT_HEX_RADIUS

def evaluate_constraint_violations(inner_hex_data, outer_hex_data):
    """Evaluate constraint violations for a given configuration"""
    violations = []
    
    # Create outer hexagon
    outer_x, outer_y, outer_angle = outer_hex_data
    outer_radius = compute_outer_hexagon_radius(inner_hex_data)
    outer_hex = hexagon_to_polygon(outer_x, outer_y, outer_angle, outer_radius)
    
    # Check each inner hexagon for containment
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        inner_hex = hexagon_to_polygon(x, y, angle)
        
        if not check_containment(inner_hex, outer_hex):
            violations.append(f"Inner hexagon {i} not contained")
    
    # Check overlaps between all pairs
    for i in range(len(inner_hex_data)):
        x1, y1, angle1 = inner_hex_data[i]
        hex1_poly = hexagon_to_polygon(x1, y1, angle1)
        
        for j in range(i+1, len(inner_hex_data)):
            x2, y2, angle2 = inner_hex_data[j]
            hex2_poly = hexagon_to_polygon(x2, y2, angle2)
            
            if check_overlap_fast(hex1_poly, hex2_poly):
                violations.append(f"Overlapping hexagons {i} and {j}")
    
    return violations

def compute_objective_function(hex_data):
    """Compute negative of 1/outer_hex_side_length (to minimize instead of maximize)"""
    # Check if hex_data is valid
    if len(hex_data) != 12:
        return 1e10
    
    # Compute outer hexagon radius
    outer_radius = compute_outer_hexagon_radius(hex_data)
    
    # If outer radius is invalid, penalize heavily
    if outer_radius <= 0:
        return 1e10
    
    # Return negative of 1/outer_radius (for minimization)
    return -1.0 / outer_radius

def evaluate_solution(hex_data, outer_hex_data):
    """Comprehensive evaluation of solution validity and quality"""
    # Basic constraint checking
    violations = evaluate_constraint_violations(hex_data, outer_hex_data)
    
    if violations:
        return False, 1e10, violations
    
    # Compute objective value
    obj_value = compute_objective_function(hex_data)
    return True, obj_value, []

def generate_diverse_initial_configurations():
    """Generate multiple diverse initial configurations using different lattice patterns"""
    configs = []
    
    # 1. Hexagonal close-packed arrangement
    hex_data = []
    hex_data.append([0.0, 0.0, 0.0])
    for i in range(6):
        angle = i * 60
        rad = np.radians(angle)
        x = 2.0 * np.cos(rad)
        y = 2.0 * np.sin(rad)
        hex_data.append([x, y, 0.0])
    for i in range(5):
        angle = i * 72 + 30
        rad = np.radians(angle)
        x = 3.464 * np.cos(rad)
        y = 3.464 * np.sin(rad)
        hex_data.append([x, y, 0.0])
    hex_data.append([0.0, -4.0, 0.0])
    hex_data = hex_data[:12]
    configs.append(np.array(hex_data))
    
    # 2. Kagome lattice inspired arrangement
    kagome_config = []
    kagome_config.append([0.0, 0.0, 0.0])
    # Ring 1
    for i in range(6):
        angle = i * 60
        rad = np.radians(angle)
        x = 1.732 * np.cos(rad)
        y = 1.732 * np.sin(rad)
        kagome_config.append([x, y, 0.0])
    # Ring 2
    for i in range(6):
        angle = i * 60 + 30
        rad = np.radians(angle)
        x = 3.464 * np.cos(rad)
        y = 3.464 * np.sin(rad)
        kagome_config.append([x, y, 0.0])
    kagome_config = kagome_config[:12]
    configs.append(np.array(kagome_config))
    
    # 3. Triangular lattice arrangement  
    tri_config = []
    tri_config.append([0.0, 0.0, 0.0])
    # First layer (5 hexagons)
    for i in range(5):
        angle = i * 72
        rad = np.radians(angle)
        x = 2.0 * np.cos(rad)
        y = 2.0 * np.sin(rad)
        tri_config.append([x, y, 0.0])
    # Second layer (6 hexagons)  
    for i in range(6):
        angle = i * 60
        rad = np.radians(angle)
        x = 3.0 * np.cos(rad)
        y = 3.0 * np.sin(rad)
        tri_config.append([x, y, 0.0])
    tri_config = tri_config[:12]
    configs.append(np.array(tri_config))
    
    # 4. Square lattice inspired
    square_config = []
    # Center
    square_config.append([0.0, 0.0, 0.0])
    # Grid pattern
    positions = [(-2, 0), (2, 0), (0, 2), (0, -2), (-1, 1), (1, 1), (-1, -1), (1, -1)]
    for i, (x, y) in enumerate(positions):
        if i < 8:
            square_config.append([x, y, 0.0])
    # Add more positions to reach 12
    extra_positions = [(2, 2), (-2, -2)]
    for i, (x, y) in enumerate(extra_positions):
        if len(square_config) < 12:
            square_config.append([x, y, 0.0])
    square_config = square_config[:12]
    configs.append(np.array(square_config))
    
    # Add small random perturbations to escape symmetric local minima
    for i in range(len(configs)):
        config = configs[i]
        for j in range(12):
            config[j][0] += random.uniform(-0.15, 0.15)
            config[j][1] += random.uniform(-0.15, 0.15)
            config[j][2] += random.uniform(-3, 3)
        configs[i] = config
    
    return configs

def adaptive_mutation(individual, generation, max_generations, base_mut_strength=0.2):
    """Adaptive mutation with decreasing strength over generations"""
    mutated = individual.copy()
    # Dynamic mutation strength
    mut_strength = base_mut_strength * (1.0 - generation / max_generations)
    
    # Mutate central hexagon
    mutated[0, 0] += random.uniform(-mut_strength*0.5, mut_strength*0.5)
    mutated[0, 1] += random.uniform(-mut_strength*0.5, mut_strength*0.5)
    mutated[0, 2] += random.uniform(-mut_strength*0.5, mut_strength*0.5)
    
    # Mutate first ring (6 hexagons)
    offset_x = random.uniform(-mut_strength, mut_strength)
    offset_y = random.uniform(-mut_strength, mut_strength)
    offset_angle = random.uniform(-mut_strength*0.5, mut_strength*0.5)
    for i in range(1, 7):
        mutated[i, 0] += offset_x
        mutated[i, 1] += offset_y
        mutated[i, 2] += offset_angle
    
    # Mutate second ring (6 hexagons)
    offset_x = random.uniform(-mut_strength, mut_strength)
    offset_y = random.uniform(-mut_strength, mut_strength)
    offset_angle = random.uniform(-mut_strength*0.5, mut_strength*0.5)
    for i in range(7, 12):
        mutated[i, 0] += offset_x
        mutated[i, 1] += offset_y
        mutated[i, 2] += offset_angle
    
    return mutated

def multi_stage_optimization(initial_configs):
    """Run multi-stage optimization with progressive refinement"""
    best_score = float('inf')
    best_solution = None
    
    # Stage 1: Global search with high diversity
    stage1_configs = []
    for i, config in enumerate(initial_configs):
        # Add variations to each config
        for j in range(3):
            var_config = config.copy()
            for k in range(12):
                var_config[k, 0] += random.uniform(-0.3, 0.3)
                var_config[k, 1] += random.uniform(-0.3, 0.3)
                var_config[k, 2] += random.uniform(-5, 5)
            stage1_configs.append(var_config)
    
    # Run optimized differential evolution with better parameters
    bounds = []
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    
    # Run DE with multiple restarts
    for restart in range(5):
        try:
            result = differential_evolution(
                lambda params: compute_objective_function(params.reshape(12, 3)),
                bounds,
                seed=42 + restart,
                popsize=50,  # Increased population size
                maxiter=150,  # More iterations
                tol=1e-9,     # Tighter tolerance
                recombination=0.8,  # Higher recombination rate
                mutation=(0.8, 0.3),  # Adaptive mutation
                disp=False,
                workers=1
            )
            
            if result.success:
                optimized_config = result.x.reshape(12, 3)
                valid, obj_value, violations = evaluate_solution(optimized_config, [0, 0, 0])
                
                if valid and obj_value < best_score:
                    best_score = obj_value
                    best_solution = optimized_config
                    
        except Exception:
            continue
    
    # Stage 2: Local refinement with tighter tolerances
    if best_solution is not None:
        try:
            bounds = []
            for i in range(12):
                bounds.extend([(-10, 10), (-10, 10), (0, 360)])
            
            result = minimize(
                lambda params: compute_objective_function(params.reshape(12, 3)),
                best_solution.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14},
                tol=1e-14
            )
            
            if result.success:
                refined_solution = result.x.reshape(12, 3)
                valid, obj_value, violations = evaluate_solution(refined_solution, [0, 0, 0])
                
                if valid and obj_value < best_score:
                    best_score = obj_value
                    best_solution = refined_solution
                    
        except Exception:
            pass
    
    # Stage 3: Return best among all attempts
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to best from initial configurations
        for config in initial_configs:
            valid, obj_value, violations = evaluate_solution(config, [0, 0, 0])
            if valid and obj_value < best_score:
                best_score = obj_value
                best_solution = config
                
    return best_solution if best_solution is not None else initial_configs[0]

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate diverse initial configurations
    initial_configs = generate_diverse_initial_configurations()
    
    # Run multi-stage optimization
    try:
        best_hex_data = multi_stage_optimization(initial_configs)
        
        # Final validation
        valid, obj_value, violations = evaluate_solution(best_hex_data, [0, 0, 0])
        
        if valid:
            final_radius = compute_outer_hexagon_radius(best_hex_data)
            return best_hex_data, np.array([0, 0, 0]), final_radius
            
    except Exception:
        pass
    
    # Fallback to known good configuration
    fallback_config = np.array([
        [0, 0, 0],              # center
        [-2.5, 0, 0],           # left
        [2.5, 0, 0],            # right
        [-1.25, 2.17, 0],       # top-left
        [1.25, 2.17, 0],        # top-right
        [-1.25, -2.17, 0],      # bottom-left
        [1.25, -2.17, 0],       # bottom-right
        [-3.75, 2.17, 0],       # far top-left
        [3.75, 2.17, 0],        # far top-right
        [-3.75, -2.17, 0],      # far bottom-left
        [3.75, -2.17, 0],       # far bottom-right
        [0, -4, 0],             # far bottom-center
    ])
    return fallback_config, np.array([0, 0, 0]), 8.0

# EVOLVE-BLOCK-END
