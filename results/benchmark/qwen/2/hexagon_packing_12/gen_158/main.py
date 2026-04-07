# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
import math
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def outer_hexagon_vertices(side_length):
    """Generate vertices of outer hexagon centered at origin."""
    return hexagon_vertices(0, 0, 0, side_length)

def check_containment(hexagon_vertices_list, outer_side_length):
    """Check if all hexagon vertices are within the outer hexagon."""
    outer_polygon = Polygon(outer_hexagon_vertices(outer_side_length))

    for vertices in hexagon_vertices_list:
        hex_polygon = Polygon(vertices)
        if not outer_polygon.contains(hex_polygon):
            return False
    return True

def check_overlap(hexagon_vertices_list):
    """Check if any hexagons overlap."""
    polygons = [Polygon(vertices) for vertices in hexagon_vertices_list]
    try:
        union = unary_union(polygons)
        total_area = sum(polygon.area for polygon in polygons)
        union_area = union.area
        # If areas match, no overlap
        return abs(total_area - union_area) < 1e-10
    except:
        # Fallback for complex cases
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if polygons[i].intersects(polygons[j]):
                    return False
        return True

def generate_voronoi_based_configurations():
    """Generate configurations using Voronoi-based geometric construction."""
    
    # Start with a well-known symmetric pattern that's close to optimal
    base_config = np.array([
        # Center hexagon
        [0.0, 0.0, 0.0],
        # First ring around center
        [-1.732, 0.0, 0.0],  # Left
        [1.732, 0.0, 0.0],   # Right
        [0.0, 1.732, 0.0],   # Top
        [0.0, -1.732, 0.0],  # Bottom
        [-0.866, 0.866, 0.0],  # Top-left
        [0.866, 0.866, 0.0],   # Top-right
        [-0.866, -0.866, 0.0], # Bottom-left
        [0.866, -0.866, 0.0],  # Bottom-right
        # Outer ring
        [-2.598, 0.0, 0.0],   # Far left
        [2.598, 0.0, 0.0],    # Far right
        [0.0, 2.598, 0.0],    # Far top
    ])
    
    # Generate multiple variants using Voronoi-inspired placement
    configurations = []
    
    # Variant 1: Original pattern with small variations
    config1 = base_config.copy()
    config1 += np.random.normal(0, 0.05, config1.shape)
    configurations.append(config1.flatten())
    
    # Variant 2: Rotated and scaled version
    config2 = base_config.copy()
    # Apply rotation matrix and scaling to create another valid configuration
    rotation_angle = math.pi / 6  # 30 degrees
    cos_a, sin_a = math.cos(rotation_angle), math.sin(rotation_angle)
    for i in range(12):
        x, y = config2[i, 0], config2[i, 1]
        config2[i, 0] = x * cos_a - y * sin_a
        config2[i, 1] = x * sin_a + y * cos_a
    configurations.append(config2.flatten())
    
    # Variant 3: Symmetrically modified pattern
    config3 = base_config.copy()
    # Perturb only outer hexagons for maximum stability
    for i in range(8, 12):  # Only outer ring hexagons
        config3[i, 0] += np.random.normal(0, 0.1)
        config3[i, 1] += np.random.normal(0, 0.1)
    configurations.append(config3.flatten())
    
    # Variant 4: Grid-based Voronoi approximation
    # Create a Voronoi-like grid for hexagon placement
    config4 = np.zeros((12, 3))
    
    # Place hexagons in a modified grid pattern
    positions = [
        (0, 0), (-1.732, 0), (1.732, 0), (0, 1.732), (0, -1.732),
        (-0.866, 0.866), (0.866, 0.866), (-0.866, -0.866), (0.866, -0.866),
        (-2.598, 0), (2.598, 0), (0, 2.598)
    ]
    
    for i in range(12):
        x, y = positions[i]
        config4[i, 0] = x + np.random.normal(0, 0.02)
        config4[i, 1] = y + np.random.normal(0, 0.02)
        config4[i, 2] = np.random.uniform(0, 360)
    
    configurations.append(config4.flatten())
    
    return configurations

def voronoi_optimize_single_config(config, outer_side_length):
    """Optimize a single configuration using Voronoi-based refinement."""
    # Parse configuration
    hexagons = config.reshape(12, 3)
    hexagon_vertices_list = []
    
    # Generate vertices for all hexagons
    for i in range(12):
        x, y, angle = hexagons[i]
        vertices = hexagon_vertices(x, y, angle)
        hexagon_vertices_list.append(vertices)
    
    # Check current validity
    if not check_containment(hexagon_vertices_list, outer_side_length):
        return config, False
    
    if not check_overlap(hexagon_vertices_list):
        return config, False
    
    # Try Voronoi-based refinement
    best_config = config.copy()
    best_valid = True
    
    # For each hexagon, try to move it to Voronoi-corrected positions
    # This is a simplified approach - in practice, we'd compute true Voronoi cells
    for iteration in range(20):
        # Compute centroids and distances
        centroids = []
        for vertices in hexagon_vertices_list:
            cx = sum(v[0] for v in vertices) / 6
            cy = sum(v[1] for v in vertices) / 6
            centroids.append([cx, cy])
        
        centroids_array = np.array(centroids)
        
        # Try to improve by moving hexagons closer to optimal positions
        improved = False
        for i in range(12):
            current_pos = hexagons[i, :2]
            # If we're in a Voronoi-like pattern, move towards ideal lattice positions
            ideal_x = round(current_pos[0] / 1.732) * 1.732
            ideal_y = round(current_pos[1] / 1.732) * 1.732
            
            # Apply small correction towards ideal position
            new_x = current_pos[0] * 0.9 + ideal_x * 0.1
            new_y = current_pos[1] * 0.9 + ideal_y * 0.1
            
            # Store new configuration temporarily
            temp_config = config.copy()
            temp_config[i*3:i*3+2] = [new_x, new_y]
            
            # Test if this change improves validity
            temp_hexagons = temp_config.reshape(12, 3)
            temp_vertices_list = []
            for j in range(12):
                x, y, angle = temp_hexagons[j]
                temp_vertices_list.append(hexagon_vertices(x, y, angle))
            
            if check_containment(temp_vertices_list, outer_side_length) and \
               check_overlap(temp_vertices_list):
                config = temp_config
                hexagons = temp_hexagons
                hexagon_vertices_list = temp_vertices_list
                improved = True
                
        if not improved:
            break
    
    return config, True

def evaluate_voronoi_config(config, outer_side_length):
    """Fast evaluation using Voronoi constraints."""
    # Parse configuration into 12 hexagons (x, y, angle)
    hexagons = config.reshape(12, 3)

    # Get vertices for all hexagons
    hexagon_vertices_list = []
    for i in range(12):
        x, y, angle = hexagons[i]
        vertices = hexagon_vertices(x, y, angle)
        hexagon_vertices_list.append(vertices)

    # Check containment and overlap
    if not check_containment(hexagon_vertices_list, outer_side_length):
        return float('inf')  # Invalid configuration

    if not check_overlap(hexagon_vertices_list):
        return float('inf')  # Overlapping hexagons

    return 0  # Valid configuration

def optimize_hexagon_positions():
    """Main optimization routine using Voronoi-based approach."""
    best_outer_side_length = 3.9419123  # Target the SOTA
    best_config = None
    best_valid = False
    
    # Generate Voronoi-based configurations
    initial_configs = generate_voronoi_based_configurations()
    
    # Evaluate all configurations
    for i, config in enumerate(initial_configs):
        # Try to fit with current boundary
        penalty = evaluate_voronoi_config(config, best_outer_side_length)
        if penalty != float('inf'):  # Valid configuration
            # Try to improve with smaller outer hexagon
            for test_side in np.linspace(3.8, best_outer_side_length, 30)[::-1]:
                penalty_test = evaluate_voronoi_config(config, test_side)
                if penalty_test != float('inf'):
                    if test_side < best_outer_side_length:
                        best_outer_side_length = test_side
                        best_config = config.copy()
                        best_valid = True
                        break
    
    # If no good solution found, fallback to standard approach
    if not best_valid:
        # Use a simple structured approach
        # Try specific geometric arrangements
        test_configs = []
        
        # Simple hexagonal packing arrangement
        simple_config = np.array([
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
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ]).flatten()
        
        test_configs.append(simple_config)
        
        # Try a more symmetric version
        symmetric_config = np.array([
            [0, 0, 0],
            [-1.732, 0, 0],  # Left
            [1.732, 0, 0],   # Right
            [0, 1.732, 0],   # Top
            [0, -1.732, 0],  # Bottom
            [-0.866, 0.866, 0],  # Top-left
            [0.866, 0.866, 0],   # Top-right
            [-0.866, -0.866, 0], # Bottom-left
            [0.866, -0.866, 0],  # Bottom-right
            [-2.598, 0, 0],   # Far left
            [2.598, 0, 0],    # Far right
            [0, 2.598, 0],    # Far top
        ]).flatten()
        
        test_configs.append(symmetric_config)
        
        for config in test_configs:
            penalty = evaluate_voronoi_config(config, best_outer_side_length)
            if penalty != float('inf'):
                for test_side in np.linspace(3.8, best_outer_side_length, 20)[::-1]:
                    penalty_test = evaluate_voronoi_config(config, test_side)
                    if penalty_test != float('inf'):
                        if test_side < best_outer_side_length:
                            best_outer_side_length = test_side
                            best_config = config.copy()
                            best_valid = True
                            break
    
    # If still no good solution, use fallback
    if not best_valid:
        # Use the standard grid approach as final fallback
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
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Final validation
    final_valid = evaluate_voronoi_config(best_config, best_outer_side_length) != float('inf')
    if not final_valid:
        # Fallback to grid pattern
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
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    return best_config.reshape(12, 3), np.array([0, 0, 0]), best_outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Run optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_positions()

    # Calculate actual score
    inv_side_length = 1.0 / outer_hex_side_length
    eval_time = time.time() - start_time

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END