# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import random
import time
from scipy.spatial.distance import cdist

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    base_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = base_vertices @ rotation_matrix.T
    translated_vertices = rotated_vertices + np.array([center_x, center_y])

    return translated_vertices

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon"""
    return outer_hex_poly.contains(hexagon_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap"""
    return hex1_poly.intersects(hex2_poly)

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum side length of outer hexagon needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)

    all_vertices = np.array(all_vertices)

    # Find bounding box
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])

    # Calculate approximate side length (simplified approach)
    # A hexagon with side length s has width 2*s and height sqrt(3)*s
    width = max_x - min_x
    height = max_y - min_y

    # Estimate side length from dimensions
    side_len_width = width / 2.0
    side_len_height = height / (np.sqrt(3))

    # Take maximum to ensure containment
    estimated_side_length = max(side_len_width, side_len_height) * 1.1  # Add small buffer

    return estimated_side_length

def evaluate_solution(inner_hex_data):
    """Evaluate fitness of solution - maximize 1/outer_hex_side_length"""
    try:
        # Create polygons for all inner hexagons
        hex_polygons = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = generate_hexagon_vertices(center_x, center_y, angle)
            hex_polygons.append(Polygon(vertices))

        # Check containment and overlap
        outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
        outer_vertices = generate_hexagon_vertices(0, 0, 0, outer_side_length)
        outer_polygon = Polygon(outer_vertices)

        # Check containment
        for poly in hex_polygons:
            if not check_containment(poly, outer_polygon):
                return 0.0  # Invalid - not fully contained

        # Check overlaps
        for i in range(len(hex_polygons)):
            for j in range(i+1, len(hex_polygons)):
                if check_overlap(hex_polygons[i], hex_polygons[j]):
                    return 0.0  # Invalid - overlaps

        # Return 1/outer_side_length as fitness
        return 1.0 / outer_side_length if outer_side_length > 0 else 0.0

    except Exception:
        return 0.0

def generate_symmetric_configurations():
    """Generate known symmetric configurations that are likely to be good"""
    configs = []
    
    # Configuration 1: Central hexagon surrounded by 6 others in hexagonal pattern, plus 4 more
    config1 = [
        [0, 0, 0],  # center
        [-2, 0, 0],  # left
        [2, 0, 0],  # right  
        [0, 2, 0],  # top
        [0, -2, 0],  # bottom
        [-1, 1.732, 0],  # top-left
        [1, 1.732, 0],  # top-right
        [-1, -1.732, 0],  # bottom-left
        [1, -1.732, 0],  # bottom-right
        [-2.5, 0, 0],  # far left
        [2.5, 0, 0],  # far right
    ]
    configs.append(config1)
    
    # Configuration 2: Hexagonal pattern with more spacing
    config2 = [
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [0, 2.5, 0],
        [0, -2.5, 0],
        [-1.25, 2.17, 0],
        [1.25, 2.17, 0],
        [-1.25, -2.17, 0],
        [1.25, -2.17, 0],
        [-3.75, 2.17, 0],
        [3.75, 2.17, 0],
    ]
    configs.append(config2)
    
    # Configuration 3: Spiral-like arrangement
    config3 = [
        [0, 0, 0],
        [0, 2, 0],
        [1.732, 1, 0],
        [1.732, -1, 0],
        [0, -2, 0],
        [-1.732, -1, 0],
        [-1.732, 1, 0],
        [0, 3.5, 0],
        [3.031, 1.75, 0],
        [3.031, -1.75, 0],
        [0, -3.5, 0],
    ]
    configs.append(config3)
    
    # Configuration 4: Optimized honeycomb with rotations
    config4 = [
        [0, 0, 0],  # center
        [-2.3, 0, 30],  # left
        [2.3, 0, 60],  # right  
        [0, 2.3, 90],  # top
        [0, -2.3, 120],  # bottom
        [-1.15, 1.99, 150],  # top-left
        [1.15, 1.99, 180],  # top-right
        [-1.15, -1.99, 210],  # bottom-left
        [1.15, -1.99, 240],  # bottom-right
        [-3.45, 0, 270],  # far left
        [3.45, 0, 300],  # far right
    ]
    configs.append(config4)
    
    return configs

def adaptive_local_search(initial_individual, max_iterations=100):
    """Advanced local search with adaptive step sizes and simulated annealing"""
    current_individual = initial_individual.copy()
    current_fitness = evaluate_solution(current_individual)
    
    # Parameters for adaptive search
    temperature = 1.0
    cooling_rate = 0.95
    min_temperature = 0.01
    step_sizes = [0.1, 0.05, 0.02]  # Adaptive step sizes
    step_idx = 0
    
    for iteration in range(max_iterations):
        # Adaptively adjust step size based on convergence
        if iteration > 0 and iteration % 10 == 0:
            step_idx = min(step_idx + 1, len(step_sizes) - 1)
            
        # Temperature cooling for simulated annealing
        if temperature > min_temperature:
            temperature *= cooling_rate
            
        improved = False
        
        # Test all possible moves
        for i in range(len(current_individual)):
            for j in range(3):  # x, y, angle
                # Try different step sizes
                for step_size in [step_sizes[step_idx], step_sizes[step_idx]*0.5]:
                    # Test both positive and negative steps
                    for direction in [-1, 1]:
                        test_individual = current_individual.copy()
                        old_value = current_individual[i][j]
                        
                        if j < 2:  # x or y coordinate
                            test_individual[i][j] = old_value + direction * step_size
                        else:  # angle
                            test_individual[i][j] = (old_value + direction * step_size) % 360
                        
                        # Evaluate the move
                        new_fitness = evaluate_solution(test_individual)
                        
                        # Accept if better
                        if new_fitness > current_fitness:
                            current_individual = test_individual
                            current_fitness = new_fitness
                            improved = True
                            
                        # Accept with probability if worse (simulated annealing)
                        elif random.random() < np.exp((new_fitness - current_fitness) / max(temperature, 1e-10)):
                            current_individual = test_individual
                            current_fitness = new_fitness
                            improved = True
        
        # Early stopping if no improvement for many iterations
        if not improved:
            if iteration > 10:
                break
                
    return current_individual, current_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    start_time = time.time()
    max_time = 175  # Leave some margin for cleanup

    best_fitness = 0.0
    best_individual = None
    
    # Try multiple symmetric configurations first
    initial_configs = generate_symmetric_configurations()
    
    for i, config in enumerate(initial_configs):
        if time.time() - start_time > max_time:
            break
            
        individual = np.array(config)
        fitness = evaluate_solution(individual)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual.copy()

    # Enhanced optimization using adaptive local search
    if best_fitness < 0.25:  # If still below target, run more aggressive optimization
        # Use the best symmetric configuration as starting point
        if best_individual is not None:
            initial_individual = best_individual.copy()
        else:
            initial_individual = np.array([
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
        
        # Apply advanced local search to refine the solution
        refined_individual, refined_fitness = adaptive_local_search(
            initial_individual, 
            max_iterations=150
        )
        
        if refined_fitness > best_fitness:
            best_fitness = refined_fitness
            best_individual = refined_individual

    # Final aggressive refinement
    if best_individual is not None and best_fitness > 0:
        # Run additional local search with fine-tuning
        final_individual, final_fitness = adaptive_local_search(
            best_individual,
            max_iterations=50
        )
        
        if final_fitness > best_fitness:
            best_fitness = final_fitness
            best_individual = final_individual

    # Final evaluation of best solution
    if best_individual is None:
        # Fallback to initial solution if optimization failed
        best_individual = np.array([
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

    outer_side_length = 1.0 / best_fitness if best_fitness > 0 else 8.0

    # Ensure valid outer hexagon side length
    if outer_side_length > 100:
        outer_side_length = 10.0

    # Center the outer hexagon at origin
    outer_hex_data = np.array([0, 0, 0])

    return best_individual, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END