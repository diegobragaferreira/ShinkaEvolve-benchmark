# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from itertools import combinations
import random
from deap import base, creator, tools, algorithms
import math

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad
    vertices = np.array([
        [center_x + side_length * np.cos(a), center_y + side_length * np.sin(a)]
        for a in angles
    ])
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if hexagon vertices are contained within outer hexagon using Shapely"""
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def evaluate_solution(individual, num_hexes=11):
    """Evaluate a solution - returns negative of inverse radius for minimization"""
    # Extract parameters
    positions = np.array(individual[:2*num_hexes]).reshape(-1, 2)
    angles = np.array(individual[2*num_hexes:3*num_hexes])
    outer_radius = individual[-1]
    
    # Create outer hexagon vertices
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    
    # Check all constraints
    total_penalty = 0
    
    # Check containment for all inner hexagons
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        if not check_containment(hex_vertices, outer_vertices):
            total_penalty += 10000  # Large penalty for containment violation
    
    # Check overlaps between all pairs of inner hexagons
    for i, j in combinations(range(len(positions)), 2):
        hex1_vertices = hexagon_vertices(positions[i][0], positions[i][1], angles[i])
        hex2_vertices = hexagon_vertices(positions[j][0], positions[j][1], angles[j])
        if check_overlap(hex1_vertices, hex2_vertices):
            total_penalty += 10000  # Large penalty for overlap violation
    
    # Return negative inverse radius plus penalties
    return -(1.0/outer_radius) + total_penalty

def create_individual(num_hexes=11):
    """Create a random individual with positions, angles, and outer radius"""
    # Random positions (within reasonable bounds)
    positions = np.random.uniform(-5, 5, size=(num_hexes, 2))
    # Random angles (0-360 degrees)
    angles = np.random.uniform(0, 360, size=num_hexes)
    # Outer radius (reasonable starting point)
    outer_radius = np.random.uniform(3, 15)
    
    return list(positions.flatten()) + list(angles) + [outer_radius]

def generate_initial_layout():
    """Generate initial hexagon layout using a structured approach"""
    # Place 11 hexagons in a 3x4 grid pattern but optimize layout
    positions = []
    angles = []
    
    # Center hexagon
    positions.append([0, 0])
    angles.append(0)
    
    # Row 1: Left and right
    positions.extend([[-2.5, 0], [2.5, 0]])
    angles.extend([0, 0])
    
    # Row 2: Top row
    positions.extend([[-1.25, 2.17], [1.25, 2.17]])
    angles.extend([0, 0])
    
    # Row 3: Bottom row
    positions.extend([[-1.25, -2.17], [1.25, -2.17]])
    angles.extend([0, 0])
    
    # Row 4: Far sides
    positions.extend([[-3.75, 2.17], [3.75, 2.17]])
    angles.extend([0, 0])
    
    # Row 5: Far bottom
    positions.extend([[-3.75, -2.17], [3.75, -2.17]])
    angles.extend([0, 0])
    
    return np.array(positions), np.array(angles)

def optimize_packing(initial_positions, initial_angles):
    """Optimize the packing arrangement using evolutionary algorithm"""
    num_hexes = len(initial_positions)
    
    # Create DEAP individual and fitness classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual, num_hexes=num_hexes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Define bounds for optimization
    bds = []
    # Position bounds
    for i in range(num_hexes):
        bds.extend([(-10, 10), (-10, 10)])  # x, y for each hexagon
    # Angle bounds (0-360)
    for i in range(num_hexes):
        bds.extend([(0, 360)])  # angle for each hexagon
    # Outer radius bounds
    bds.append((1, 20))
    
    # Register genetic operators
    toolbox.register("evaluate", evaluate_solution, num_hexes=num_hexes)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initial population
    pop = toolbox.population(n=50)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run evolution
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                         ngen=30, stats=stats, verbose=False)
    except:
        # Fallback to basic optimization if evolution fails
        pass
    
    # Get best individual
    best_ind = tools.selBest(pop, k=1)[0]
    
    # Extract results
    best_positions = np.array(best_ind[:2*num_hexes]).reshape(-1, 2)
    best_angles = np.array(best_ind[2*num_hexes:3*num_hexes])
    best_radius = best_ind[-1]
    
    # Fine-tune with local optimization
    try:
        # Prepare flattened parameters for scipy optimizer
        initial_params = np.concatenate([
            best_positions.flatten(),
            best_angles,
            [best_radius]
        ])
        
        # Define bounds for scipy optimization
        bounds = [(b[0], b[1]) for b in bds]
        
        # Use scipy optimization for fine-tuning
        def obj_func(params):
            return -evaluate_solution(params, num_hexes)  # Negative because we want to maximize
        
        result = minimize(obj_func, initial_params, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 100}, tol=1e-6)
        
        if result.success:
            final_positions = result.x[:2*num_hexes].reshape(-1, 2)
            final_angles = result.x[2*num_hexes:3*num_hexes]
            final_radius = result.x[-1]
        else:
            final_positions = best_positions
            final_angles = best_angles
            final_radius = best_radius
            
    except Exception:
        final_positions = best_positions
        final_angles = best_angles
        final_radius = best_radius
    
    return final_positions, final_angles, final_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate initial layout
    initial_positions, initial_angles = generate_initial_layout()
    
    # Optimize the packing
    optimized_positions, optimized_angles, outer_radius = optimize_packing(initial_positions, initial_angles)
    
    # Create final data structures
    inner_hex_data = np.column_stack([
        optimized_positions,
        optimized_angles
    ])
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = outer_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
