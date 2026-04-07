# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist
import random
import time
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Calculate vertices of a hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vx = x + side_length * np.cos(theta)
        vy = y + side_length * np.sin(theta)
        vertices.append((vx, vy))
    return np.array(vertices)

@jit(nopython=True)
def point_in_hexagon(px, py, hx, hy, angle_deg, side_length=1):
    """Fast point-in-hexagon test using dot products"""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    dx = px - hx
    dy = py - hy
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    r = np.sqrt(lx*lx + ly*ly)
    if r > 1.0: return False
    return True

@jit(nopython=True)
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Distance from point to line segment"""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx*dx + dy*dy
    
    if len_sq == 0.0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def hexagon_distance(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Fast approximation of minimum distance between two hexagons"""
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle, 1.0)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle, 1.0)
    
    min_dist = 1000000.0
    
    for i in range(6):
        j = (i + 1) % 6
        x1, y1 = v1[i]
        x2, y2 = v1[j]
        
        for k in range(6):
            x3, y3 = v2[k]
            dist = distance_point_to_line(x3, y3, x1, y1, x2, y2)
            min_dist = min(min_dist, dist)
    
    return min_dist

@jit(nopython=True)
def hexagon_overlap_fast(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Fast approximation of whether two hexagons overlap"""
    dist = np.sqrt((h1_x - h2_x)**2 + (h1_y - h2_y)**2)
    if dist > 2.0:
        return False
    
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle, 1.0)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle, 1.0)
    
    for i in range(6):
        x1, y1 = v1[i]
        if point_in_hexagon(x1, y1, h2_x, h2_y, h2_angle):
            return True
    
    for i in range(6):
        x1, y1 = v2[i]
        if point_in_hexagon(x1, y1, h1_x, h1_y, h1_angle):
            return True
    
    return False

def get_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Get shapely polygon representation of hexagon"""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment(hex_poly, outer_poly):
    """Check if hexagon is completely contained within outer hexagon"""
    return outer_poly.contains(hex_poly) or (outer_poly.intersects(hex_poly) and 
                                           outer_poly.intersection(hex_poly).area == hex_poly.area)

def calculate_outer_hexagon_radius(inner_positions, inner_angles):
    """Calculate minimum radius needed to contain all inner hexagons"""
    max_dist = 0
    outer_center = (0, 0)
    
    all_vertices = []
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        all_vertices.extend(hex_vertices)
    
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
        max_dist = max(max_dist, dist)
    
    return max_dist * 1.1

def build_adaptive_spatial_hash(positions, angles, min_cell_size=1.0):
    """Build adaptive spatial hash based on hexagon density and distribution"""
    if len(positions) == 0:
        return {}
    
    # Calculate statistics about the distribution
    positions_array = np.array(positions)
    mean_position = np.mean(positions_array, axis=0)
    std_dev = np.std(positions_array, axis=0)
    
    # Estimate cell size based on density - smaller cells in dense regions
    avg_distance = np.mean([np.linalg.norm(np.array(positions[i]) - np.array(positions[j])) 
                           for i in range(len(positions)) for j in range(i+1, len(positions)) if i != j])
    
    # Adaptive cell size: smaller for dense areas, bigger for sparse
    cell_size = max(min_cell_size, avg_distance * 0.7)
    
    # Create hash grid
    grid = {}
    
    for i in range(len(positions)):
        pos = positions[i]
        angle = angles[i]
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        
        # Get bounding box
        min_x = min(vx for vx, vy in hex_vertices)
        max_x = max(vx for vx, vy in hex_vertices)
        min_y = min(vy for vx, vy in hex_vertices)
        max_y = max(vy for vx, vy in hex_vertices)
        
        # Determine grid cells
        min_cell_x = int(min_x // cell_size)
        max_cell_x = int(max_x // cell_size)
        min_cell_y = int(min_y // cell_size)
        max_cell_y = int(max_y // cell_size)
        
        # Add to all affected cells
        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                cell_key = (cx, cy)
                if cell_key not in grid:
                    grid[cell_key] = []
                grid[cell_key].append(i)
    
    return grid

def get_collision_candidates(grid, positions, angles, hex_idx, cell_size=2.0):
    """Get collision candidates using adaptive spatial hashing"""
    pos = positions[hex_idx]
    angle = angles[hex_idx]
    hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
    
    min_x = min(vx for vx, vy in hex_vertices)
    max_x = max(vx for vx, vy in hex_vertices)
    min_y = min(vy for vx, vy in hex_vertices)
    max_y = max(vy for vx, vy in hex_vertices)
    
    min_cell_x = int(min_x // cell_size) - 1
    max_cell_x = int(max_x // cell_size) + 1
    min_cell_y = int(min_y // cell_size) - 1
    max_cell_y = int(max_y // cell_size) + 1
    
    candidates = set()
    for cx in range(min_cell_x, max_cell_x + 1):
        for cy in range(min_cell_y, max_cell_y + 1):
            cell_key = (cx, cy)
            if cell_key in grid:
                candidates.update(grid[cell_key])
    
    return list(candidates)

def evaluate_fitness(solution, use_adaptive_hash=True):
    """Evaluate solution fitness with adaptive collision detection"""
    positions = solution[:22].reshape(-1, 2)
    angles = solution[22:]
    
    # Create inner hexagons
    inner_hexagons = []
    for i in range(11):
        pos = positions[i]
        angle = angles[i]
        hex_poly = get_hexagon_polygon(pos[0], pos[1], angle)
        inner_hexagons.append(hex_poly)
    
    # Check containment
    outer_radius = calculate_outer_hexagon_radius(positions, angles)
    outer_hexagon = get_hexagon_polygon(0, 0, 0, outer_radius)
    
    for hex_poly in inner_hexagons:
        if not check_containment(hex_poly, outer_hexagon):
            return 1e10  # Penalty for non-containment
    
    # Adaptive collision detection with dynamic cell sizing
    grid = None
    if use_adaptive_hash:
        grid = build_adaptive_spatial_hash(positions, angles)
    
    # Check for overlaps
    penalty = 0
    
    if grid is not None and len(grid) > 0:
        # Use adaptive spatial hashing
        for i in range(11):
            candidates = get_collision_candidates(grid, positions, angles, i)
            for j in candidates:
                if i >= j: continue
                if not hexagon_overlap_fast(positions[i][0], positions[i][1], angles[i],
                                          positions[j][0], positions[j][1], angles[j]):
                    continue
                penalty += 1000000
    else:
        # Fallback to direct checking
        for i in range(11):
            for j in range(i+1, 11):
                if not hexagon_overlap_fast(positions[i][0], positions[i][1], angles[i],
                                          positions[j][0], positions[j][1], angles[j]):
                    continue
                penalty += 1000000
    
    # Return fitness (negative because we want to maximize 1/outer_radius)
    if penalty > 0:
        return penalty + 1.0 / outer_radius
    
    return -1.0 / outer_radius

def create_hexagonal_pattern():
    """Create initial solution using hexagonal packing principles"""
    # Create a hexagonal pattern with center and surrounding ring
    positions = []
    angles = []
    
    # Center hexagon
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Surrounding ring of 6 hexagons
    for i in range(6):
        angle = i * 60
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y])
        angles.append(0.0)
    
    # Additional hexagons in a pattern that maximizes packing density
    additional_positions = [
        (-3.0, 1.0), (3.0, 1.0), (-3.0, -1.0), (3.0, -1.0),
        (0.0, 3.0), (0.0, -3.0), (1.5, 2.6), (-1.5, -2.6),
        (-1.5, 2.6), (1.5, -2.6)
    ]
    
    for pos in additional_positions:
        if len(positions) < 11:
            positions.append(list(pos))
            angles.append(0.0)
    
    # Ensure exactly 11 positions
    while len(positions) < 11:
        positions.append([0.0, 0.0])
        angles.append(0.0)
    
    # Add small random perturbations to avoid degenerate cases
    for i in range(len(positions)):
        positions[i][0] += random.uniform(-0.1, 0.1)
        positions[i][1] += random.uniform(-0.1, 0.1)
        angles[i] += random.uniform(-3, 3)
    
    # Flatten for solution vector
    solution = []
    for pos in positions:
        solution.extend(pos)
    solution.extend(angles)
    
    return np.array(solution)

def generate_initial_population(pop_size=10):
    """Generate diverse initial population using multiple strategies"""
    population = []
    
    # Strategy 1: Hexagonal pattern 
    for _ in range(pop_size // 3):
        population.append(create_hexagonal_pattern())
    
    # Strategy 2: Random configurations with density considerations
    for _ in range(pop_size // 3):
        solution = []
        for i in range(11):
            # Place hexagons with some minimal separation
            if i == 0:
                # Center hexagon
                solution.extend([0.0, 0.0])
            else:
                # Place randomly with some constraints
                x = random.uniform(-3.0, 3.0)
                y = random.uniform(-3.0, 3.0)
                solution.extend([x, y])
            
            # Random angle
            solution.append(random.uniform(0, 360))
        
        population.append(np.array(solution))
    
    # Strategy 3: Clustered configurations
    for _ in range(pop_size - len(population)):
        solution = []
        for i in range(11):
            if i == 0:
                # Center hexagon
                solution.extend([0.0, 0.0])
            else:
                # Cluster around center with some variation
                angle = random.uniform(0, 2*np.pi)
                radius = random.uniform(1.5, 2.5)  
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                solution.extend([x, y])
            
            # Random angle
            solution.append(random.uniform(0, 360))
            
        population.append(np.array(solution))
    
    return population

def adaptive_crossover(parent1, parent2):
    """Specialized crossover operator for hexagon positions and angles"""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Crossover for positions and angles with different strategies
    for i in range(11):
        # For first 2 coordinates (position), use uniform crossover
        if random.random() < 0.5:
            child1[2*i:2*i+2] = parent2[2*i:2*i+2]
            child2[2*i:2*i+2] = parent1[2*i:2*i+2]
        
        # For angle, use a blend of both parents with some randomness
        if random.random() < 0.5:
            alpha = random.random()
            child1[22+i] = alpha * parent1[22+i] + (1-alpha) * parent2[22+i]
            child2[22+i] = alpha * parent2[22+i] + (1-alpha) * parent1[22+i]
        else:
            # Use one parent's angle with some variation
            if random.random() < 0.5:
                child1[22+i] = parent1[22+i] + random.uniform(-10, 10)
                child2[22+i] = parent2[22+i] + random.uniform(-10, 10)
            else:
                child1[22+i] = parent2[22+i] + random.uniform(-10, 10)
                child2[22+i] = parent1[22+i] + random.uniform(-10, 10)
        
        # Normalize angles to [0, 360)
        child1[22+i] = child1[22+i] % 360
        child2[22+i] = child2[22+i] % 360
    
    return child1, child2

def adaptive_mutation(individual, generation, max_generations):
    """Adaptive mutation operator that decreases over time"""
    mutated = individual.copy()
    
    # Adaptive mutation rate based on generation
    mutation_rate = 0.3 * (1.0 - generation/max_generations)
    mutation_rate = max(mutation_rate, 0.05)  # Minimum mutation rate
    
    for i in range(11):
        # Position mutation with adaptive magnitude
        if random.random() < mutation_rate:
            mutated[2*i] += random.uniform(-0.2, 0.2)
            mutated[2*i+1] += random.uniform(-0.2, 0.2)
        
        # Angle mutation
        if random.random() < mutation_rate:
            mutated[22+i] += random.uniform(-15, 15)
            mutated[22+i] = mutated[22+i] % 360
    
    return mutated

def packing_density_optimization(solution, max_iterations=100):
    """Optimize solution by adjusting positions to increase packing density"""
    positions = solution[:22].reshape(-1, 2)
    angles = solution[22:]
    
    # Simple iterative optimization: push hexagons away from each other
    for iter in range(max_iterations):
        # Calculate pairwise distances
        pos_array = np.array(positions)
        
        # Calculate forces between hexagons based on distance
        for i in range(11):
            for j in range(i+1, 11):
                # Skip if overlapping or very close
                dist = np.linalg.norm(pos_array[i] - pos_array[j])
                if dist < 1.8:  # Too close
                    # Push them apart
                    direction = pos_array[i] - pos_array[j]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        pos_array[i] += direction * 0.05
                        pos_array[j] -= direction * 0.05
        
        # Reconstruct solution
        new_solution = []
        for i in range(11):
            new_solution.extend(pos_array[i])
        new_solution.extend(angles)
        
        # Evaluate new solution
        new_fitness = evaluate_fitness(np.array(new_solution), use_adaptive_hash=False)
        old_fitness = evaluate_fitness(solution, use_adaptive_hash=False)
        
        if new_fitness < old_fitness:
            solution = np.array(new_solution)
            positions = solution[:22].reshape(-1, 2)
        else:
            # Slight reduction in step size
            break
    
    return solution

def evolutionary_hexagon_packing():
    """Main evolutionary algorithm with adaptive strategies"""
    population_size = 15
    generations = 100
    elite_count = 3
    
    # Generate initial population
    population = generate_initial_population(population_size)
    
    best_fitness = float('inf')
    best_individual = None
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for indiv in population:
            fitness = evaluate_fitness(indiv)
            fitness_scores.append(fitness)
        
        # Track best individual
        min_fitness_idx = np.argmin(fitness_scores)
        if fitness_scores[min_fitness_idx] < best_fitness:
            best_fitness = fitness_scores[min_fitness_idx]
            best_individual = population[min_fitness_idx].copy()
        
        # Sort by fitness (lower is better)
        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(elite_count):
            new_population.append(population[i].copy())
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = random.randint(0, elite_count*2 - 1)
            parent2_idx = random.randint(0, elite_count*2 - 1)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child1, child2 = adaptive_crossover(parent1, parent2)
            
            # Mutation
            child1 = adaptive_mutation(child1, gen, generations)
            child2 = adaptive_mutation(child2, gen, generations)
            
            # Packing density optimization
            child1 = packing_density_optimization(child1)
            child2 = packing_density_optimization(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
    
    return best_individual

def local_geometric_refinement(solution):
    """Refine solution using geometric optimization approaches"""
    # Phase 1: Local optimization with gradient-like moves
    positions = solution[:22].reshape(-1, 2)
    angles = solution[22:]
    
    # Try small adjustments to find better local optima
    best_solution = solution.copy()
    best_fitness = evaluate_fitness(solution)
    
    for _ in range(100):
        # Make several small changes and see if any improve the solution
        test_solution = best_solution.copy()
        
        # Choose random hexagon to adjust
        hex_idx = random.randint(0, 10)
        
        # Small position adjustment
        test_solution[2*hex_idx] += random.uniform(-0.05, 0.05)
        test_solution[2*hex_idx+1] += random.uniform(-0.05, 0.05)
        
        # Small angle adjustment
        test_solution[22+hex_idx] += random.uniform(-5, 5)
        test_solution[22+hex_idx] = test_solution[22+hex_idx] % 360
        
        # Evaluate
        new_fitness = evaluate_fitness(test_solution)
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            best_solution = test_solution.copy()
    
    return best_solution

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Run evolutionary algorithm
        initial_solution = evolutionary_hexagon_packing()
        
        # Fine-tune with local refinement
        refined_solution = local_geometric_refinement(initial_solution)
        
        # Extract final data
        final_positions = refined_solution[:22].reshape(-1, 2)
        final_angles = refined_solution[22:]
        
        # Create inner hex data
        inner_hex_data = np.column_stack([final_positions, final_angles])
        
        # Create outer hex data (centered)
        outer_hex_data = np.array([0, 0, 0])
        
        # Calculate outer hex side length
        outer_radius = calculate_outer_hexagon_radius(final_positions, final_angles)
        # Convert to side length for regular hexagon
        outer_hex_side_length = outer_radius / (np.sqrt(3) / 2)
        
        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial configuration that works well
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
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END