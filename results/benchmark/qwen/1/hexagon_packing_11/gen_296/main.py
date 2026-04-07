# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
import math
import random
from collections import defaultdict

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = math.radians(angle_deg)
    base_vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(theta)
        y = center_y + side_length * math.sin(theta)
        base_vertices.append((x, y))
    return base_vertices

def check_containment(hex_vertices, outer_vertices):
    """Check if hexagon vertices are contained within outer hexagon"""
    inner_poly = Polygon(hex_vertices)
    outer_poly = Polygon(outer_vertices)
    return outer_poly.contains(inner_poly)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_radius(inner_positions, inner_angles):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_distance = 0
    outer_center = (0, 0)

    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        vertices = hexagon_vertices(pos[0], pos[1], angle)
        
        for vertex in vertices:
            distance = math.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_distance = max(max_distance, distance)

    return max_distance * 1.1  # Safety factor

def evaluate_solution(params):
    """Evaluate solution fitness - returns negative of 1/outer_radius for maximization"""
    # Reshape parameters into positions and angles
    positions = params[:22].reshape(-1, 2)
    angles = params[22:]
    
    # Create inner hexagons
    inner_hexagons = []
    for i in range(11):
        pos = positions[i]
        angle = angles[i]
        vertices = hexagon_vertices(pos[0], pos[1], angle)
        inner_hexagons.append(vertices)

    # Check containment
    outer_radius = calculate_outer_radius(positions, angles)
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    
    # Check containment for all inner hexagons
    for vertices in inner_hexagons:
        if not check_containment(vertices, outer_vertices):
            return 1e10  # Large penalty for non-containment

    # Check overlaps between all pairs of hexagons
    for i in range(11):
        for j in range(i+1, 11):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return 1e10  # Large penalty for overlap

    # Return negative of 1/outer_radius for maximization (minimize the negative)
    return -1.0 / outer_radius

def generate_symmetric_configurations():
    """Generate multiple symmetric configurations for multi-start approach"""
    configs = []
    
    # Configuration 1: Hexagonal lattice with center
    config1 = [
        [0, 0, 0],  # center
        [-2.2, 0, 0],  # left
        [2.2, 0, 0],  # right
        [0, 2.2, 0],  # top
        [0, -2.2, 0],  # bottom
        [-1.1, 1.9, 0],  # top-left
        [1.1, 1.9, 0],  # top-right
        [-1.1, -1.9, 0],  # bottom-left
        [1.1, -1.9, 0],  # bottom-right
        [-2.2, 1.65, 0],  # further top-left
        [2.2, 1.65, 0],  # further top-right
    ]
    configs.append(config1)
    
    # Configuration 2: Spiral-like arrangement
    config2 = [
        [0, 0, 0],  # center
        [0, 2.5, 0],  # top
        [2.17, 1.25, 0],  # top-right
        [2.17, -1.25, 0],  # bottom-right
        [0, -2.5, 0],  # bottom
        [-2.17, -1.25, 0],  # bottom-left
        [-2.17, 1.25, 0],  # top-left
        [0, 3.5, 0],  # far top
        [3.03, 1.75, 0],  # far top-right
        [3.03, -1.75, 0],  # far bottom-right
        [0, -3.5, 0],  # far bottom
    ]
    configs.append(config2)
    
    # Configuration 3: Cluster arrangement
    config3 = [
        [0, 0, 0],  # center
        [-2.0, 0, 0],  # left
        [2.0, 0, 0],  # right
        [0, 2.0, 0],  # top
        [0, -2.0, 0],  # bottom
        [-1.5, 1.5, 0],  # top-left
        [1.5, 1.5, 0],  # top-right
        [-1.5, -1.5, 0],  # bottom-left
        [1.5, -1.5, 0],  # bottom-right
        [-2.5, 2.5, 0],  # far top-left
        [2.5, 2.5, 0],  # far top-right
    ]
    configs.append(config3)
    
    return configs

def adaptive_local_search(positions, angles, max_iter=100):
    """Enhanced local search with adaptive step sizes"""
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_score = evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))
    
    # Adaptive step sizes
    step_sizes = [0.2, 0.1, 0.05, 0.02]
    improvement_threshold = 1e-8
    
    for iteration in range(max_iter):
        improved = False
        current_step = step_sizes[min(iteration // 25, len(step_sizes)-1)]
        
        # Try perturbing each hexagon
        for i in range(11):
            # Store original values
            orig_pos = best_positions[i].copy()
            orig_angle = best_angles[i]
            
            # Try position perturbations
            for dim in range(2):
                for delta in [-current_step, current_step]:
                    test_positions = best_positions.copy()
                    test_positions[i][dim] = orig_pos[dim] + delta
                    test_angles = best_angles.copy()
                    
                    new_score = evaluate_solution(np.concatenate([test_positions.flatten(), test_angles]))
                    if new_score < best_score:
                        best_score = new_score
                        best_positions = test_positions
                        best_angles = test_angles
                        improved = True
                        
            # Try angle perturbation
            for delta in [-5.0, 5.0]:
                test_positions = best_positions.copy()
                test_angles = best_angles.copy()
                test_angles[i] = (orig_angle + delta) % 360
                
                new_score = evaluate_solution(np.concatenate([test_positions.flatten(), test_angles]))
                if new_score < best_score:
                    best_score = new_score
                    best_positions = test_positions
                    best_angles = test_angles
                    improved = True
                    
        if not improved:
            break
            
    return best_positions, best_angles

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    max_time_seconds = 175
    
    best_fitness = 0.0
    best_individual = None
    best_positions = None
    best_angles = None
    
    # Try multiple symmetric configurations as starting points
    configs = generate_symmetric_configurations()
    
    for i, config in enumerate(configs):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Convert config to parameter array
        positions = np.array([[c[0], c[1]] for c in config])
        angles = np.array([c[2] for c in config])
        
        # Evaluate this configuration
        fitness = evaluate_solution(np.concatenate([positions.flatten(), angles]))
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_positions = positions.copy()
            best_angles = angles.copy()
    
    # If no good configurations were found, start with a fallback
    if best_positions is None:
        fallback_config = [
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
        ]
        best_positions = np.array([[c[0], c[1]] for c in fallback_config])
        best_angles = np.array([c[2] for c in fallback_config])
        best_fitness = evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))
    
    # Multi-start optimization from different configurations
    for start_run in range(5):
        if time.time() - start_time > max_time_seconds - 10:
            break
            
        # Randomize the starting configuration slightly
        np.random.seed(start_run)
        positions = best_positions.copy()
        angles = best_angles.copy()
        
        # Add some random perturbation
        for i in range(11):
            positions[i][0] += np.random.uniform(-0.5, 0.5)
            positions[i][1] += np.random.uniform(-0.5, 0.5)
            angles[i] += np.random.uniform(-15, 15)
            angles[i] %= 360
            
        # Optimization with differential evolution for coarse tuning
        try:
            bounds = []
            for _ in range(22):  # Positions
                bounds.append((-10, 10))
            for _ in range(11):  # Angles
                bounds.append((0, 360))
                
            result = differential_evolution(
                lambda x: evaluate_solution(x),
                bounds,
                maxiter=30,
                popsize=10,
                seed=start_run,
                disp=False,
                tol=1e-6,
                strategy='best1bin'
            )
            
            if result.success:
                refined_positions = result.x[:22].reshape(-1, 2)
                refined_angles = result.x[22:]
                
                # Local refinement
                refined_positions, refined_angles = adaptive_local_search(refined_positions, refined_angles, max_iter=50)
                
                current_fitness = evaluate_solution(np.concatenate([refined_positions.flatten(), refined_angles]))
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_positions = refined_positions
                    best_angles = refined_angles
                    
        except:
            continue
    
    # Final local refinement
    if time.time() - start_time < max_time_seconds - 5:
        best_positions, best_angles = adaptive_local_search(best_positions, best_angles, max_iter=100)
    
    # Construct final result
    inner_hex_data = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    # Calculate actual outer hex side length
    outer_radius = calculate_outer_radius(best_positions, best_angles)
    # For a regular hexagon with circumradius R, side length = R
    outer_hex_side_length = outer_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END