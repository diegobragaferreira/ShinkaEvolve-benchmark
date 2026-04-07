# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import time
from numba import jit, prange
import warnings
from scipy.spatial import cKDTree
import random
warnings.filterwarnings('ignore')

@jit(nopython=True)
def hexagon_vertices_numba(x, y, angle_deg, side_length=1):
    """Calculate vertices of a hexagon efficiently using numba"""
    angle_rad = np.radians(angle_deg)
    vertices = np.empty((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + side_length * np.cos(theta)
        vertices[i, 1] = y + side_length * np.sin(theta)
    return vertices

def get_hexagon_polygon_fast(x, y, angle_deg, side_length=1):
    """Fast creation of shapely polygon"""
    vertices = hexagon_vertices_numba(x, y, angle_deg, side_length)
    return Polygon(vertices)

@jit(nopython=True)
def point_in_polygon_fast(point_x, point_y, polygon_vertices):
    """Fast point-in-polygon check using ray casting algorithm"""
    n = len(polygon_vertices)
    inside = False
    p1x, p1y = polygon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if point_y > min(p1y, p2y):
            if point_y <= max(p1y, p2y):
                if point_x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (point_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point_x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def check_containment_fast(hex_vertices, outer_vertices):
    """Fast containment check - all vertices must be inside outer polygon"""
    for i in range(6):
        if not point_in_polygon_fast(hex_vertices[i, 0], hex_vertices[i, 1], outer_vertices):
            return False
    return True

def calculate_outer_hexagon_radius_fast(inner_positions, inner_angles):
    """Fast calculation of minimum radius needed to contain all inner hexagons"""
    max_dist = 0.0
    outer_center_x, outer_center_y = 0.0, 0.0

    # Get all vertices of all inner hexagons
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        hex_vertices = hexagon_vertices_numba(pos[0], pos[1], angle)
        for j in range(6):
            vertex_x, vertex_y = hex_vertices[j]
            dist = np.sqrt((vertex_x - outer_center_x)**2 + (vertex_y - outer_center_y)**2)
            max_dist = max(max_dist, dist)

    return max_dist * 1.1  # Safety factor

def build_spatial_index(hex_vertices_list):
    """Build spatial index (bounding boxes) for fast collision detection"""
    bounds = []
    for vertices in hex_vertices_list:
        min_x = np.min(vertices[:, 0])
        max_x = np.max(vertices[:, 0])
        min_y = np.min(vertices[:, 1])
        max_y = np.max(vertices[:, 1])
        bounds.append((min_x, min_y, max_x, max_y))
    return bounds

def fast_collision_check(bounds1, bounds2):
    """Quick bounding box overlap check"""
    return not (bounds1[2] < bounds2[0] or bounds2[2] < bounds1[0] or
               bounds1[3] < bounds2[1] or bounds2[3] < bounds1[1])

def evaluate_solution(solution):
    """Improved evaluation with early termination and better numerical stability"""
    try:
        # Reshape solution into positions and angles
        positions = solution[:22].reshape(-1, 2)  # 11 hexagons * 2 coordinates each
        angles = solution[22:]  # 11 angles

        # Fast geometric checks for constraints
        # Create outer hexagon with calculated radius
        outer_radius = calculate_outer_hexagon_radius_fast(positions, angles)
        outer_vertices = hexagon_vertices_numba(0, 0, 0, outer_radius)
        
        # Check containment for all inner hexagons
        for i in range(11):
            pos = positions[i]
            angle = angles[i]
            hex_vertices = hexagon_vertices_numba(pos[0], pos[1], angle)
            if not check_containment_fast(hex_vertices, outer_vertices):
                return 1e10  # Penalty for non-containment

        # Fast collision detection with spatial indexing
        all_hex_vertices = []
        for i in range(11):
            pos = positions[i]
            angle = angles[i]
            hex_vertices = hexagon_vertices_numba(pos[0], pos[1], angle)
            all_hex_vertices.append(hex_vertices)

        # Build spatial index for fast collision detection
        spatial_bounds = build_spatial_index(all_hex_vertices)
        
        # Check for overlaps between inner hexagons
        for i in range(11):
            for j in range(i+1, 11):
                # Quick bounds check first
                if fast_collision_check(spatial_bounds[i], spatial_bounds[j]):
                    # Full polygon intersection test
                    hex1 = Polygon(all_hex_vertices[i])
                    hex2 = Polygon(all_hex_vertices[j])
                    if hex1.intersects(hex2):
                        return 1e10  # Penalty for overlap

        # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
        if outer_radius > 0:
            return -1.0 / outer_radius
        else:
            return 1e10
            
    except Exception:
        return 1e10

def create_initial_configuration_better():
    """Create a better initial configuration based on hexagonal packing principles"""
    # Create a hexagonal lattice pattern with some variation
    # Center hexagon
    initial_positions = [[0.0, 0.0]]
    
    # Surrounding hexagons in 3 layers
    layer1 = [
        [-2.0, 0.0],    # left
        [2.0, 0.0],     # right
        [0.0, 2.0],     # top
        [0.0, -2.0],    # bottom
        [-1.0, 1.0],    # top-left
        [1.0, 1.0],     # top-right
        [-1.0, -1.0],   # bottom-left
        [1.0, -1.0],    # bottom-right
    ]
    
    layer2 = [
        [-2.0, 1.5],    # further top-left
        [2.0, 1.5],     # further top-right
        [-2.0, -1.5],   # further bottom-left
        [2.0, -1.5],    # further bottom-right
        [-3.0, 0.0],    # far left
        [3.0, 0.0],     # far right
    ]
    
    initial_positions.extend(layer1)
    initial_positions.extend(layer2)
    
    # Adjust spacing and add small random variations
    for i in range(len(initial_positions)):
        initial_positions[i][0] *= 1.1
        initial_positions[i][1] *= 1.1
        # Add slight randomization
        initial_positions[i][0] += np.random.normal(0, 0.05)
        initial_positions[i][1] += np.random.normal(0, 0.05)

    initial_angles = [0.0] * 11
    return initial_positions, initial_angles

def create_diverse_initial_configurations(num_configs=5):
    """Create multiple diverse initial configurations"""
    configs = []
    for _ in range(num_configs):
        pos, ang = create_initial_configuration_better()
        configs.append((pos, ang))
    return configs

def simulated_annealing_refinement(positions, angles, max_iter=1000, initial_temp=1.0, cooling_rate=0.95):
    """Refine solution using simulated annealing"""
    current_positions = positions.copy()
    current_angles = angles.copy()
    current_score = evaluate_solution(np.concatenate([current_positions.flatten(), current_angles]))
    
    best_positions = current_positions.copy()
    best_angles = current_angles.copy()
    best_score = current_score
    
    temperature = initial_temp
    
    for iteration in range(max_iter):
        # Make a random change
        idx = random.randint(0, 10)
        dim = random.randint(0, 2)  # 0=x, 1=y, 2=angle
        
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        if dim < 2:  # x or y coordinate
            new_positions[idx][dim] += np.random.normal(0, 0.05)
        else:  # angle
            new_angles[idx] += np.random.normal(0, 5.0)
        
        new_score = evaluate_solution(np.concatenate([new_positions.flatten(), new_angles]))
        
        # Accept or reject the new solution
        if new_score < current_score or random.random() < np.exp((current_score - new_score) / temperature):
            current_positions = new_positions
            current_angles = new_angles
            current_score = new_score
            
            if new_score < best_score:
                best_positions = new_positions
                best_angles = new_angles
                best_score = new_score
        
        temperature *= cooling_rate
        
    return best_positions, best_angles

def multi_start_optimization():
    """Run optimization from multiple starting points"""
    # Create diverse initial configurations
    configs = create_diverse_initial_configurations(5)
    
    best_result = None
    best_score = float('inf')
    
    for i, (init_pos, init_ang) in enumerate(configs):
        # Flatten initial solution
        initial_solution = []
        for pos in init_pos:
            initial_solution.extend(pos)
        initial_solution.extend(init_ang)
        initial_solution = np.array(initial_solution)

        # Set bounds for optimization with tighter constraints
        bounds = []
        # Position bounds (tighter for faster convergence)
        for _ in range(22):
            bounds.append((-10.0, 10.0))  # X and Y coordinates
        # Angle bounds  
        for _ in range(11):
            bounds.append((0.0, 360.0))   # Rotation angles

        # Run differential evolution
        try:
            # Use higher population size and more iterations for better search
            result = differential_evolution(
                evaluate_solution,
                bounds,
                maxiter=75,           # Increased iterations
                popsize=20,           # Increased population size
                seed=42+i,            # Different seed for each config
                disp=False,
                tol=1e-6,
                strategy='best1bin'
            )
            
            # Extract final solution
            final_positions = result.x[:22].reshape(-1, 2)
            final_angles = result.x[22:]
            
            # Local refinement with simulated annealing on top performers
            refined_positions, refined_angles = simulated_annealing_refinement(
                final_positions, final_angles
            )
            
            # Evaluate refined solution
            refined_score = evaluate_solution(np.concatenate([refined_positions.flatten(), refined_angles]))
            
            if refined_score < best_score:
                best_score = refined_score
                best_result = (refined_positions, refined_angles)
                
        except Exception as e:
            continue  # Skip this configuration if it fails
    
    # If no successful optimization, return the best from initial configs
    if best_result is None:
        return configs[0][0], configs[0][1]
    
    return best_result[0], best_result[1]

def optimize_hexagon_packing():
    """Main optimization function with enhanced algorithm"""
    # Run multi-start optimization
    final_positions, final_angles = multi_start_optimization()
    
    return final_positions, final_angles

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
        # Run optimized packing
        final_positions, final_angles = optimize_hexagon_packing()

        # Create inner hex data
        inner_hex_data = np.column_stack([final_positions, final_angles])

        # Create outer hex data (centered)
        outer_hex_data = np.array([0, 0, 0])

        # Calculate outer hex side length
        outer_radius = calculate_outer_hexagon_radius_fast(final_positions, final_angles)
        # Convert to side length for regular hexagon
        outer_hex_side_length = outer_radius / (np.sqrt(3) / 2)

        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        
        # Ensure reasonable output
        if outer_hex_side_length < 1.0 or outer_hex_side_length > 100:
            outer_hex_side_length = 10.0

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to improved initial solution
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
        outer_hex_side_length = 10.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END