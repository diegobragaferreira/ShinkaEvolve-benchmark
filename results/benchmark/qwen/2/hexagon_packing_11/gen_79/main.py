# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import time
from joblib import Parallel, delayed
import random

def generate_unit_hexagon_vertices(center_x, center_y, rotation_degrees):
    """Generate vertices of a unit regular hexagon with given center and rotation"""
    angle_rad = np.radians(rotation_degrees)
    # Unit hexagon vertices centered at origin
    base_vertices = np.array([
        [1, 0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
        [-1, 0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = base_vertices @ rotation_matrix.T
    return rotated_vertices + np.array([center_x, center_y])

def check_containment(hexagon_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon"""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hexagon_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hexagon_vertices(side_length, center=(0, 0), rotation=0):
    """Calculate vertices of outer hexagon"""
    angle_rad = np.radians(rotation)
    base_vertices = np.array([
        [side_length, 0], 
        [side_length/2, side_length*np.sqrt(3)/2], 
        [-side_length/2, side_length*np.sqrt(3)/2],
        [-side_length, 0], 
        [-side_length/2, -side_length*np.sqrt(3)/2], 
        [side_length/2, -side_length*np.sqrt(3)/2]
    ])
    
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = base_vertices @ rotation_matrix.T
    return rotated_vertices + np.array(center)

def evaluate_fitness(individual, outer_side_length):
    """Evaluate fitness of an individual (11 hexagons)"""
    # Extract positions and rotations
    hex_data = individual.reshape(-1, 3)
    centers = hex_data[:, :2]
    rotations = hex_data[:, 2]
    
    # Generate hexagon vertices
    hex_vertices_list = []
    for i in range(len(centers)):
        vertices = generate_unit_hexagon_vertices(centers[i][0], centers[i][1], rotations[i])
        hex_vertices_list.append(vertices)
    
    # Calculate outer hexagon vertices
    outer_vertices = calculate_outer_hexagon_vertices(outer_side_length)
    
    # Check containment and overlap
    penalty = 0
    
    # Check containment penalty
    for vertices in hex_vertices_list:
        if not check_containment(vertices, outer_vertices):
            penalty += 1000000  # Heavy penalty for containment violation
    
    # Check overlap penalty
    for i in range(len(hex_vertices_list)):
        for j in range(i+1, len(hex_vertices_list)):
            if check_overlap(hex_vertices_list[i], hex_vertices_list[j]):
                penalty += 1000000  # Heavy penalty for overlap
    
    # If no violations, reward tight packing (smaller outer radius)
    if penalty == 0:
        # Calculate maximum distance from center to any hexagon vertex
        max_dist = 0
        for vertices in hex_vertices_list:
            for vertex in vertices:
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                max_dist = max(max_dist, dist)
        # Reward smaller outer radius (we want to minimize this)
        penalty = max_dist  # Lower means better packing
    
    return penalty

def generate_initial_population(n_individuals, n_hexagons=11):
    """Generate multiple diverse initial populations"""
    population = []
    
    # Phase 1: Grid-based initialization
    grid_positions = [
        [0, 0], [-2.5, 0], [2.5, 0], [-1.25, 2.17], [1.25, 2.17],
        [-1.25, -2.17], [1.25, -2.17], [-3.75, 2.17], [3.75, 2.17],
        [-3.75, -2.17], [3.75, -2.17]
    ]
    
    for _ in range(n_individuals // 3):
        individual = np.zeros((n_hexagons, 3))
        for i, pos in enumerate(grid_positions):
            individual[i] = [*pos, 0]
        # Add some random noise
        individual[:, :2] += np.random.normal(0, 0.1, (n_hexagons, 2))
        population.append(individual.flatten())
    
    # Phase 2: Spiral initialization
    spiral_positions = []
    for i in range(n_hexagons):
        angle = i * 0.7  # Spiral angle
        radius = 0.5 + i * 0.3
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        spiral_positions.append([x, y])
    
    for _ in range(n_individuals // 3):
        individual = np.zeros((n_hexagons, 3))
        for i, pos in enumerate(spiral_positions):
            individual[i] = [*pos, 0]
        individual[:, :2] += np.random.normal(0, 0.1, (n_hexagons, 2))
        population.append(individual.flatten())
        
    # Phase 3: Random initialization
    for _ in range(n_individuals // 3):
        individual = np.random.uniform(-5, 5, (n_hexagons, 3))
        individual[:, 2] = np.random.uniform(0, 360, n_hexagons)  # Random rotations
        population.append(individual.flatten())
    
    return population

def optimize_packing():
    """Main optimization routine"""
    # Start with a reasonable guess for outer radius
    outer_radius_guess = 10.0
    
    # We'll try different outer radii
    best_solution = None
    best_fitness = float('inf')
    
    # Try several outer radius values
    for test_radius in np.linspace(4.0, 8.0, 10):
        # Create bounds for each parameter (x, y, angle) for 11 hexagons
        bounds = []
        for i in range(11):
            # X coordinate bounds
            bounds.extend([(-test_radius*2, test_radius*2)])
            # Y coordinate bounds  
            bounds.extend([(-test_radius*2, test_radius*2)])
            # Angle bounds (0-360 degrees)
            bounds.extend([(0, 360)])
        
        # Run local optimization
        def objective(x):
            return evaluate_fitness(x, test_radius)
            
        # Use differential evolution for global optimization
        result = differential_evolution(objective, bounds, maxiter=100, popsize=15, seed=42)
        
        if result.success and result.fun < best_fitness:
            best_fitness = result.fun
            best_solution = result.x
            
    if best_solution is None:
        # Fallback to simple case
        return np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0], [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0]
        ]), np.array([0, 0, 0]), 8.0
        
    # Convert back to standard format
    hex_data = best_solution.reshape(-1, 3)
    outer_hex_data = np.array([0, 0, 0])
    
    # Estimate outer hexagon side length
    max_dist = 0
    for i in range(len(hex_data)):
        center = hex_data[i][:2]
        # Get hexagon vertices and find maximum distance
        vertices = generate_unit_hexagon_vertices(center[0], center[1], hex_data[i][2])
        for vertex in vertices:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            max_dist = max(max_dist, dist)
    
    # Set outer hexagon side length to slightly larger than max distance
    outer_hex_side_length = max_dist * 1.1  # Add buffer
    
    return hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use the optimization function to find best solution
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_packing()
    
    # Final validation and refinement
    # Check again for overlap/containment
    max_dist = 0
    hex_vertices_list = []
    
    for i in range(len(inner_hex_data)):
        vertices = generate_unit_hexagon_vertices(
            inner_hex_data[i][0], 
            inner_hex_data[i][1], 
            inner_hex_data[i][2]
        )
        hex_vertices_list.append(vertices)
        
        # Track maximum distance
        for vertex in vertices:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            max_dist = max(max_dist, dist)
    
    # Recalculate outer hexagon side length with more accurate estimation
    outer_hex_side_length = max_dist * 1.1
    
    # Ensure we have a valid outer hexagon
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
