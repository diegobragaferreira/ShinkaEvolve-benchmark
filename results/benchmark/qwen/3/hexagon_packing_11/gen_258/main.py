# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
from joblib import Parallel, delayed
import math

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
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

def compute_outer_hexagon_radius(inner_positions, inner_angles, initial_radius_estimate=5.0):
    """Compute minimum outer hexagon radius using binary search with adaptive precision"""
    # Binary search with adaptive precision based on convergence
    left = initial_radius_estimate
    right = 20.0
    best_radius = right
    
    # Track convergence to adjust precision dynamically
    prev_diff = float('inf')
    max_iterations = 50
    iterations = 0
    
    while iterations < max_iterations:
        current_diff = right - left
        # Adaptive precision: more precise as we converge
        if abs(current_diff - prev_diff) < 1e-3 and current_diff > 1e-4:
            precision_threshold = 1e-6
        else:
            precision_threshold = 1e-4
            
        if current_diff <= precision_threshold:
            break
            
        mid = (left + right) / 2.0
        outer_vertices = hexagon_vertices(0, 0, 0, mid)
        valid = True
        
        # Check all inner hexagons
        for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
            hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
            if not check_containment(hex_vertices, outer_vertices):
                valid = False
                break
                
        if valid:
            best_radius = mid
            right = mid
        else:
            left = mid
            
        prev_diff = current_diff
        iterations += 1
        
    return best_radius

def evaluate_fitness_hexagon_config(positions, angles):
    """Evaluate fitness for a hexagon configuration"""
    # Check overlap constraint first (early rejection)
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            hex1_vertices = hexagon_vertices(positions[i][0], positions[i][1], angles[i])
            hex2_vertices = hexagon_vertices(positions[j][0], positions[j][1], angles[j])
            if check_overlap(hex1_vertices, hex2_vertices):
                return -1e10, 1e10  # Invalid configuration penalty
                
    # Compute outer radius
    outer_radius = compute_outer_hexagon_radius(positions, angles)
    
    # Check containment constraint
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        if not check_containment(hex_vertices, outer_vertices):
            return -1e10, 1e10  # Invalid containment penalty
            
    # Return 1/radius as fitness (higher is better)
    return 1.0 / outer_radius, outer_radius

def generate_geometric_initial_config():
    """Generate initial configuration using geometric principles for better packing"""
    # More carefully arranged hexagon positions
    # Based on hexagonal close packing principles with optimized spacing
    
    # Central hexagon
    positions = [[0.0, 0.0]]
    
    # First ring: 6 hexagons at unit distance from center
    for i in range(6):
        angle = i * math.pi/3
        positions.append([math.cos(angle) * 2.0, math.sin(angle) * 2.0])
    
    # Second ring: 4 additional hexagons in triangular formation
    positions.append([-3.0, 0.0])  # Left
    positions.append([3.0, 0.0])   # Right
    positions.append([0.0, -3.0])  # Bottom
    positions.append([0.0, 3.0])   # Top
    
    # Take only first 11 positions
    positions = positions[:11]
    
    # Convert to numpy array and add rotation (all initially 0)
    positions = np.array(positions)
    angles = np.zeros(11)
    
    # Add small random perturbations to escape symmetry and improve diversity
    np.random.seed(42)
    positions += np.random.normal(0, 0.1, positions.shape)
    
    return positions, angles

def generate_diverse_initial_population(population_size):
    """Generate diverse initial population with geometric seeds"""
    population = []
    for i in range(population_size):
        positions, angles = generate_geometric_initial_config()
        population.append((positions, angles))
    return population

def local_refinement_step(positions, angles, max_iter=50):
    """Apply local refinement to improve solution quality"""
    # Flatten for optimization
    initial_vars = np.concatenate([positions.flatten(), angles])
    
    def objective_function(vars):
        # Reshape variables back to positions and angles
        pos_flat = vars[:-11]
        angle_vars = vars[-11:]
        positions = pos_flat.reshape(-1, 2)
        angles = angle_vars
        
        # Evaluate fitness
        fitness, _ = evaluate_fitness_hexagon_config(positions, angles)
        return -fitness  # Minimize negative fitness = maximize fitness
    
    # Bounds for optimization
    bounds = []
    for i in range(len(positions)):
        bounds.extend([(-10, 10), (-10, 10)])  # Position bounds
    for i in range(len(angles)):
        bounds.extend([(0, 360)])  # Angle bounds
    
    try:
        # Use L-BFGS-B for local refinement
        result = minimize(
            objective_function,
            initial_vars,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result.success:
            final_vars = result.x
            pos_flat = final_vars[:-11]
            angles = final_vars[-11:]
            positions = pos_flat.reshape(-1, 2)
            return positions, angles
    except:
        pass
    
    # Return original if optimization fails
    return positions, angles

def parallel_evaluate_population(population):
    """Evaluate multiple configurations in parallel"""
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(lambda x: evaluate_fitness_hexagon_config(x[0], x[1]))(indiv) 
        for indiv in population
    )
    return results

def evolutionary_hexagon_optimization():
    """Main evolutionary optimization algorithm"""
    # Parameters
    population_size = 30
    generations = 80
    mutation_rate = 0.2
    crossover_rate = 0.7
    elitism_rate = 0.1
    max_time_seconds = 170
    
    start_time = time.time()
    
    # Initialize population
    population = generate_diverse_initial_population(population_size)
    
    # Best solution tracking
    best_fitness = float('-inf')
    best_positions = None
    best_angles = None
    
    for gen in range(generations):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Evaluate population in parallel
        fitness_results = parallel_evaluate_population(population)
        
        # Extract fitness values and find the best
        fitness_values = [result[0] for result in fitness_results]
        current_best_idx = np.argmax(fitness_values)
        current_best_fitness = fitness_values[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_positions = population[current_best_idx][0].copy()
            best_angles = population[current_best_idx][1].copy()
        
        # Local refinement on best individual periodically
        if gen % 5 == 0:
            refined_positions, refined_angles = local_refinement_step(
                best_positions, best_angles, max_iter=20
            )
            refined_fitness, _ = evaluate_fitness_hexagon_config(refined_positions, refined_angles)
            if refined_fitness > best_fitness:
                best_fitness = refined_fitness
                best_positions = refined_positions
                best_angles = refined_angles
        
        # Selection based on fitness
        sorted_indices = np.argsort(fitness_values)[::-1]
        selected_population = [population[i] for i in sorted_indices[:population_size//2]]
        
        # Elitism - keep best individuals
        elite_count = int(elitism_rate * population_size)
        elites = [population[i] for i in sorted_indices[:elite_count]]
        
        # Create new population through crossover and mutation
        new_population = elites.copy()
        
        # Crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = np.random.choice(len(selected_population))
            parent2_idx = np.random.choice(len(selected_population))
            
            parent1_positions, parent1_angles = selected_population[parent1_idx]
            parent2_positions, parent2_angles = selected_population[parent2_idx]
            
            # Crossover
            if np.random.rand() < crossover_rate:
                # Single point crossover for positions
                crossover_point = np.random.randint(1, len(parent1_positions))
                child1_positions = np.vstack([parent1_positions[:crossover_point], parent2_positions[crossover_point:]])
                child1_angles = np.hstack([parent1_angles[:crossover_point], parent2_angles[crossover_point:]])
                
                child2_positions = np.vstack([parent2_positions[:crossover_point], parent1_positions[crossover_point:]])
                child2_angles = np.hstack([parent2_angles[:crossover_point], parent1_angles[crossover_point:]])
            else:
                child1_positions, child1_angles = parent1_positions.copy(), parent1_angles.copy()
                child2_positions, child2_angles = parent2_positions.copy(), parent2_angles.copy()
            
            # Mutation
            for i in range(len(child1_positions)):
                if np.random.rand() < mutation_rate:
                    child1_positions[i][0] += np.random.normal(0, 0.2)
                    child1_positions[i][1] += np.random.normal(0, 0.2)
                    child1_angles[i] += np.random.normal(0, 10)
                    child1_angles[i] %= 360
                    
            for i in range(len(child2_positions)):
                if np.random.rand() < mutation_rate:
                    child2_positions[i][0] += np.random.normal(0, 0.2)
                    child2_positions[i][1] += np.random.normal(0, 0.2)
                    child2_angles[i] += np.random.normal(0, 10)
                    child2_angles[i] %= 360
            
            new_population.extend([(child1_positions, child1_angles), (child2_positions, child2_angles)])
        
        # Trim to exact population size
        population = new_population[:population_size]
    
    # Final local refinement on the best solution
    if best_positions is not None:
        final_positions, final_angles = local_refinement_step(best_positions, best_angles, max_iter=100)
        final_fitness, outer_radius = evaluate_fitness_hexagon_config(final_positions, final_angles)
        return final_positions, final_angles, outer_radius
    else:
        # Fallback to initial configuration
        positions, angles = generate_geometric_initial_config()
        return positions, angles, 20.0

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use evolutionary optimization approach
    positions, angles, outer_radius = evolutionary_hexagon_optimization()
    
    # Format result properly
    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    # Validate solution
    fitness, _ = evaluate_fitness_hexagon_config(positions, angles)
    if fitness < 0.1:  # If fitness is too low, use fallback
        # Fallback to known good configuration
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END