# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
import math
from joblib import Parallel, delayed
import multiprocessing

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

def check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast bounding box overlap check"""
    # Get bounding boxes
    x1 = [v[0] for v in hex1_vertices]
    y1 = [v[1] for v in hex1_vertices]
    x2 = [v[0] for v in hex2_vertices]
    y2 = [v[1] for v in hex2_vertices]

    min_x1, max_x1 = min(x1), max(x1)
    min_y1, max_y1 = min(y1), max(y1)
    min_x2, max_x2 = min(x2), max(x2)
    min_y2, max_y2 = min(y2), max(y2)

    # Check if bounding boxes overlap
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False
    return True

def check_hexagon_overlap_sat(hex1_vertices, hex2_vertices):
    """SAT-based overlap check for hexagons"""
    # Get all edge normals for both hexagons
    axes = []

    # Get edges and normals for first hexagon
    for i in range(6):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        # Normal vector (perpendicular)
        normal = (-edge[1], edge[0])
        # Normalize
        length = math.sqrt(normal[0]**2 + normal[1]**2)
        if length > 1e-10:
            normal = (normal[0]/length, normal[1]/length)
        axes.append(normal)

    # Get edges and normals for second hexagon
    for i in range(6):
        p1 = hex2_vertices[i]
        p2 = hex2_vertices[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        # Normal vector (perpendicular)
        normal = (-edge[1], edge[0])
        # Normalize
        length = math.sqrt(normal[0]**2 + normal[1]**2)
        if length > 1e-10:
            normal = (normal[0]/length, normal[1]/length)
        axes.append(normal)

    # Check each axis
    for axis in axes:
        # Project both polygons onto axis
        proj1 = [p[0]*axis[0] + p[1]*axis[1] for p in hex1_vertices]
        proj2 = [p[0]*axis[0] + p[1]*axis[1] for p in hex2_vertices]

        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)

        # Check for overlap
        if max1 < min2 or max2 < min1:
            return False  # No overlap on this axis

    return True  # Overlap exists

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely with buffer for precision"""
    # Fast preliminary check with SAT
    if not check_hexagon_overlap_sat(hex1_vertices, hex2_vertices):
        return False

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

    # Calculate centroid of all vertices
    avg_x = sum(v[0] for v in all_vertices) / len(all_vertices)
    avg_y = sum(v[1] for v in all_vertices) / len(all_vertices)

    # Calculate maximum distance from centroid to any vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - avg_x)**2 + (y - avg_y)**2)
        max_dist = max(max_dist, dist)

    # Add safety margin
    side_length = max_dist * 1.05

    return side_length

def generate_geometrically_informed_initialization():
    """Generate high-quality initial configuration using geometric principles"""
    # Start with a carefully designed geometric arrangement
    # This follows principles of efficient hexagonal packing patterns
    # Center hexagon surrounded by 6 others in a triangular arrangement
    # Plus 4 additional hexagons in a second ring
    
    # Base pattern - 11 hexagons in a compact formation
    # Using mathematical relationships for optimal spacing
    
    # Hexagon side length is 1, so distance between centers for touching is 2
    # But we want to maximize packing efficiency
    
    # Pattern: center + 6 surrounding + 4 in outer ring
    config = []
    
    # Center hexagon
    config.append([0.0, 0.0, 0.0])
    
    # First ring - 6 hexagons at distance 2 from center
    angles = [0, 60, 120, 180, 240, 300]
    for angle in angles:
        rad = math.radians(angle)
        x = 2.0 * math.cos(rad)
        y = 2.0 * math.sin(rad)
        config.append([x, y, 0.0])
    
    # Second ring - 4 hexagons placed to maximize packing efficiency
    # These should be placed to fill gaps and reduce outer hexagon footprint
    angles = [30, 90, 150, 210]
    for i, angle in enumerate(angles):
        rad = math.radians(angle)
        # Place them slightly further out
        x = 3.0 * math.cos(rad)
        y = 3.0 * math.sin(rad)
        config.append([x, y, 0.0])
    
    # Ensure we have exactly 11 hexagons
    while len(config) < 11:
        config.append([0.0, 0.0, 0.0])
    while len(config) > 11:
        config.pop()
    
    # Add slight randomness to prevent degenerate cases
    for i in range(len(config)):
        config[i][0] += random.uniform(-0.1, 0.1)
        config[i][1] += random.uniform(-0.1, 0.1)
        config[i][2] += random.uniform(-5, 5)
    
    return np.array(config)

def generate_alternative_initializations():
    """Generate alternative high-quality initial configurations"""
    configs = []
    
    # Configuration 1: Ring pattern with symmetry
    config1 = []
    # Central hexagon
    config1.append([0.0, 0.0, 0.0])
    # Surrounding ring
    for i in range(6):
        angle = i * 60
        rad = math.radians(angle)
        x = 2.0 * math.cos(rad)
        y = 2.0 * math.sin(rad)
        config1.append([x, y, 0.0])
    
    # Outer ring
    for i in range(4):
        angle = 45 + i * 90
        rad = math.radians(angle)
        x = 3.46 * math.cos(rad)
        y = 3.46 * math.sin(rad)
        config1.append([x, y, 0.0])
    
    configs.append(np.array(config1[:11]))
    
    # Configuration 2: Linear arrangement with rotation
    config2 = []
    x_positions = [0.0, -1.5, 1.5, -3.0, 3.0, -4.5, 4.5, -6.0, 6.0, -7.5, 7.5]
    for i, x in enumerate(x_positions):
        config2.append([x, 0.0, 0.0])
    
    configs.append(np.array(config2[:11]))
    
    # Configuration 3: Diamond arrangement
    config3 = []
    # Central
    config3.append([0.0, 0.0, 0.0])
    # Vertical line
    config3.append([0.0, 2.0, 0.0])
    config3.append([0.0, -2.0, 0.0])
    # Horizontal line  
    config3.append([2.0, 0.0, 0.0])
    config3.append([-2.0, 0.0, 0.0])
    # Diagonals
    config3.append([1.73, 1.0, 0.0])
    config3.append([-1.73, 1.0, 0.0])
    config3.append([1.73, -1.0, 0.0])
    config3.append([-1.73, -1.0, 0.0])
    # Outer ring
    config3.append([3.46, 0.0, 0.0])
    config3.append([0.0, 3.46, 0.0])
    
    configs.append(np.array(config3[:11]))
    
    return configs

def evaluate_solution(hex_data):
    """Evaluate fitness of a solution - maximize 1/outer_hex_side_length"""
    # Create outer hexagon vertices (assuming centered at origin)
    outer_side_length = calculate_outer_hex_side_length(hex_data)

    # Check constraints
    try:
        # Check containment for all inner hexagons
        outer_hex_vertices = create_hexagon_vertices(0, 0, 0, outer_side_length)

        # Check if all hexagons are contained and non-overlapping
        total_penalty = 0

        # Check containment - early termination if violated
        for i in range(len(hex_data)):
            center_x, center_y, angle = hex_data[i]
            inner_hex_vertices = create_hexagon_vertices(center_x, center_y, angle)
            if not check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                total_penalty += 10000  # Large penalty for containment violation
                return 1.0 / outer_side_length - total_penalty

        # Check overlaps with early termination
        overlap_count = 0
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                center_x1, center_y1, angle1 = hex_data[i]
                center_x2, center_y2, angle2 = hex_data[j]

                hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
                hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)

                if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                    overlap_count += 1
                    total_penalty += 10000  # Large penalty for overlap
                    # Early termination - don't need to check more overlaps for this individual
                    return 1.0 / outer_side_length - total_penalty

        # Return fitness (inverse of outer hex side length + penalties)
        if total_penalty > 0:
            return 1.0 / outer_side_length - total_penalty

        return 1.0 / outer_side_length

    except Exception as e:
        return -10000  # Very poor fitness for invalid solutions

def run_single_optimization_de(bounds, initial_guess):
    """Run a single optimization attempt using differential evolution"""
    try:
        # Apply differential evolution with refined parameters
        result = differential_evolution(
            evaluate_solution,
            bounds,
            maxiter=150,  # Reduced iterations to save time
            popsize=10,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if result.success:
            final_hex_data = result.x.reshape(11, 3)
            outer_side_length = calculate_outer_hex_side_length(final_hex_data)
            fitness = evaluate_solution(final_hex_data)
            return (final_hex_data, np.array([0, 0, 0]), outer_side_length, fitness)
    except Exception as e:
        pass
    return None

def run_single_optimization_local_refinement(initial_guess):
    """Run local refinement on a given initial guess"""
    try:
        # Local optimization using L-BFGS-B
        def objective(x_flat):
            hex_data = x_flat.reshape(-1, 3)
            return -evaluate_solution(hex_data)  # Negative because we minimize

        # Apply local refinement
        x0 = initial_guess.flatten()
        bounds = []
        for i in range(len(x0)):
            if i % 3 == 0:  # x coordinate
                bounds.append((-6.0, 6.0))
            elif i % 3 == 1:  # y coordinate
                bounds.append((-6.0, 6.0))
            else:  # angle
                bounds.append((0.0, 360.0))

        # Try Nelder-Mead first for robustness
        try:
            result_nm = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 30})
            if result_nm.success:
                solution_nm = result_nm.x.reshape(-1, 3)
                fitness_nm = evaluate_solution(solution_nm)
                
                # Then refine with L-BFGS-B if Nelder-Mead is better
                if fitness_nm > evaluate_solution(initial_guess.reshape(-1, 3)):
                    result_lbfgs = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
                    refined_solution = result_lbfgs.x.reshape(-1, 3)
                    refined_outer_side_length = calculate_outer_hex_side_length(refined_solution)
                    refined_fitness = evaluate_solution(refined_solution)
                    return (refined_solution, np.array([0, 0, 0]), refined_outer_side_length, refined_fitness)
                else:
                    return (solution_nm, np.array([0, 0, 0]), calculate_outer_hex_side_length(solution_nm), fitness_nm)
        except:
            # Fall back to just L-BFGS-B
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
            refined_solution = result.x.reshape(-1, 3)
            refined_outer_side_length = calculate_outer_hex_side_length(refined_solution)
            refined_fitness = evaluate_solution(refined_solution)
            return (refined_solution, np.array([0, 0, 0]), refined_outer_side_length, refined_fitness)
            
    except Exception as e:
        return None

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses hybrid optimization to find the optimal configuration.
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate bounds for optimization
    bounds = []
    for i in range(11):  # 11 hexagons
        # x coordinates
        bounds.append((-6, 6))
        # y coordinates  
        bounds.append((-6, 6))
        # rotations - 0 to 360 degrees
        bounds.append((0, 360))

    # Generate multiple high-quality initial configurations
    initial_configs = []
    
    # 1. Geometrically informed initialization (primary)
    geometric_init = generate_geometrically_informed_initialization()
    initial_configs.append(geometric_init)
    
    # 2. Alternative initializations
    alternative_configs = generate_alternative_initializations()
    for config in alternative_configs:
        initial_configs.append(config)
    
    # 3. Generate some random configurations for diversity
    for _ in range(2):
        random_config = []
        for i in range(11):
            x = random.uniform(-5.0, 5.0)
            y = random.uniform(-5.0, 5.0)
            angle = random.uniform(0, 360)
            random_config.append([x, y, angle])
        initial_configs.append(np.array(random_config))

    # Run multiple optimization attempts in parallel
    results = Parallel(n_jobs=min(4, multiprocessing.cpu_count()), verbose=0)(
        delayed(run_single_optimization_de)(bounds, config) 
        for config in initial_configs
    )

    # Also run local refinement on best candidates
    valid_results = [r for r in results if r is not None]
    
    # Extract top performers and run local refinement
    if valid_results:
        # Sort by fitness and take top 3
        sorted_results = sorted(valid_results, key=lambda x: x[3], reverse=True)
        top_candidates = sorted_results[:3]
        
        local_refinement_results = Parallel(n_jobs=min(2, multiprocessing.cpu_count()), verbose=0)(
            delayed(run_single_optimization_local_refinement)(result[0]) 
            for result in top_candidates
        )
        
        # Combine all results
        all_results = valid_results + [r for r in local_refinement_results if r is not None]
    else:
        all_results = []
    
    if all_results:
        # Find the best result among all attempts
        best_result = max(all_results, key=lambda x: x[3])  # Compare by fitness
        return best_result[0], best_result[1], best_result[2]
    else:
        # Fallback to original configuration if optimization fails
        print("Using fallback configuration")
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


# EVOLVE-BLOCK-END