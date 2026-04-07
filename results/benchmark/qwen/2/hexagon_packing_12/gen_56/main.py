# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import affinity
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# Global constants
HEX_RADIUS = 1.0
HEX_APO = HEX_RADIUS * np.sqrt(3) / 2  # Apothem of unit hexagon

class SpatialHash:
    """Spatial hash for efficient collision detection"""
    def __init__(self, cell_size=3.0):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
    
    def _hash(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))
    
    def add_point(self, x, y, obj_id):
        cell = self._hash(x, y)
        self.grid[cell].append((x, y, obj_id))
    
    def get_candidates(self, x, y):
        candidates = []
        cell = self._hash(x, y)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                candidates.extend(self.grid[neighbor_cell])
        return candidates

def generate_hexagon_vertices(center_x, center_y, angle_degrees):
    """Generate vertices of a unit regular hexagon given center and rotation."""
    angle_rad = np.radians(angle_degrees)
    # Hexagon vertices in local coordinates (unit radius)
    local_verts = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = HEX_RADIUS * np.cos(theta)
        y = HEX_RADIUS * np.sin(theta)
        local_verts.append((x, y))
    
    # Transform to global coordinates
    global_verts = []
    for x, y in local_verts:
        # Rotate
        rot_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        rot_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        # Translate
        global_verts.append((rot_x + center_x, rot_y + center_y))
    
    return np.array(global_verts)

def create_hexagon_polygon(center_x, center_y, angle_degrees):
    """Create Shapely polygon representation of a hexagon."""
    vertices = generate_hexagon_vertices(center_x, center_y, angle_degrees)
    return Polygon(vertices)

def check_containment(hex_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hex_poly) or outer_hex_poly.intersects(hex_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def compute_outer_hexagon_side_length(inner_hex_data):
    """Compute the minimal side length needed for the outer hexagon to contain all inner hexagons."""
    # Find the bounding box of all hexagon vertices
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle_degrees = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle_degrees)
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return 1e6
    
    # Calculate bounding rectangle
    min_x = min(v[0] for v in all_vertices)
    max_x = max(v[0] for v in all_vertices)
    min_y = min(v[1] for v in all_vertices)
    max_y = max(v[1] for v in all_vertices)
    
    # Compute required side length for outer hexagon
    # Outer hexagon's circumradius = distance from center to farthest vertex
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    max_dist = 0
    for x, y in all_vertices:
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        max_dist = max(max_dist, dist_sq)
    
    # Side length = sqrt(max_dist) * 2 / sqrt(3)
    return np.sqrt(max_dist) * 2 / np.sqrt(3)

def evaluate_fitness_adaptive(solution, outer_hex_center=(0, 0), outer_hex_angle=0):
    """Evaluate fitness with adaptive penalty system."""
    # Reshape solution back to 12 hexagons with (x, y, angle)
    positions = solution.reshape(-1, 3)
    
    # Create polygons for all inner hexagons
    inner_polygons = []
    for i in range(len(positions)):
        center_x, center_y, angle_degrees = positions[i]
        poly = create_hexagon_polygon(center_x, center_y, angle_degrees)
        inner_polygons.append(poly)
    
    # Find outer hexagon side length
    side_length = compute_outer_hexagon_side_length(positions)
    
    # Check containment and overlaps using spatial hash for efficiency
    total_penalty = 0
    num_inner = len(inner_polygons)
    
    # Outer hexagon polygon (centered at origin with given angle)
    outer_poly = create_hexagon_polygon(outer_hex_center[0], outer_hex_center[1], outer_hex_angle)
    
    # Spatial hash for efficient overlap checking
    spatial_hash = SpatialHash(cell_size=3.0)
    
    # Add all hexagon centers to spatial hash
    for i, pos in enumerate(positions):
        spatial_hash.add_point(pos[0], pos[1], i)
    
    # Check containment (larger penalty for containment violations)
    containment_violations = 0
    for i in range(num_inner):
        if not check_containment(inner_polygons[i], outer_poly):
            total_penalty += 1e6
            containment_violations += 1
    
    # Check overlaps with spatial hashing (reduced complexity)
    overlap_count = 0
    for i in range(num_inner):
        # Get nearby candidates using spatial hash
        candidates = spatial_hash.get_candidates(positions[i][0], positions[i][1])
        for _, _, j in candidates:
            if i < j:  # Only check each pair once
                if check_overlap(inner_polygons[i], inner_polygons[j]):
                    # Adjust penalty based on overlap severity
                    overlap_penalty = 1e5 + overlap_count * 1e3
                    total_penalty += overlap_penalty
                    overlap_count += 1
    
    # Adaptive penalty scaling based on constraint violations
    if containment_violations > 0:
        # Scale penalty inversely with number of violations for better convergence
        total_penalty *= (1.0 + containment_violations * 0.1)
    
    # Objective: maximize 1/side_length
    # So minimize negative log of side_length plus penalties
    if side_length < 1e-6:
        fitness = -1e10
    else:
        fitness = -np.log(side_length) - total_penalty
    
    return fitness, side_length

def get_multiscale_symmetric_config():
    """Generate initial configuration with multi-scale symmetry properties."""
    # Start with a well-known symmetric pattern
    positions = []
    
    # Central hexagon
    positions.append([0, 0, 0])
    
    # First ring (6 hexagons)
    angles = np.linspace(0, 360, 7)[:-1]  # 6 angles, excluding last to avoid duplication
    radius1 = 2.0
    for angle in angles:
        rad = np.radians(angle)
        x = radius1 * np.cos(rad)
        y = radius1 * np.sin(rad)
        positions.append([x, y, 0])
    
    # Second ring (6 hexagons)
    radius2 = 3.5  # Slightly larger
    offset_angles = [angle + 30 for angle in angles]  # Offset by 30 degrees
    for angle in offset_angles:
        rad = np.radians(angle)
        x = radius2 * np.cos(rad)
        y = radius2 * np.sin(rad)
        positions.append([x, y, 0])
    
    # Third hexagon at specific location
    positions.append([0, -5.0, 0])
    
    return np.array(positions).flatten()

def refine_with_local_search(initial_solution, max_iterations=100):
    """Use local optimization to refine the solution."""
    # Flatten the solution
    x0 = initial_solution.copy()
    
    def objective(x):
        # Reshape for evaluation
        positions = x.reshape(-1, 3)
        # Add small penalty to encourage staying close to initial
        initial_penalty = 0.01 * np.sum((x - initial_solution)**2)
        _, side_length = evaluate_fitness_adaptive(x)
        return -np.log(side_length) - initial_penalty
    
    bounds = [(-15, 15)] * len(x0)  # Reasonable bounds
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': max_iterations, 'ftol': 1e-8})
        return result.x
    except:
        return x0

def optimize_hexagon_packing():
    """Main optimization function with enhanced multi-strategy approach."""
    # Multiple initial configurations to sample different regions of search space
    initial_configs = []
    
    # Configuration 1: Multi-scale symmetric pattern
    config1 = get_multiscale_symmetric_config()
    initial_configs.append(config1)
    
    # Configuration 2: Random perturbed version of first
    config2 = config1 + np.random.normal(0, 0.5, len(config1))
    initial_configs.append(config2)
    
    # Configuration 3: Another symmetric variant with different spacing
    positions = []
    positions.append([0, 0, 0])
    angles = np.linspace(0, 360, 7)[:-1] 
    radius1 = 1.8
    for angle in angles:
        rad = np.radians(angle)
        x = radius1 * np.cos(rad)
        y = radius1 * np.sin(rad)
        positions.append([x, y, 0])
    
    # Add additional positions
    positions.append([0, -4.0, 0])
    positions.append([3.0, 0, 0])
    positions.append([-3.0, 0, 0])
    positions.append([0, 4.0, 0])
    positions.append([1.5, 2.6, 0])
    positions.append([-1.5, -2.6, 0])
    
    config3 = np.array(positions).flatten()
    initial_configs.append(config3)
    
    best_side_length = float('inf')
    best_positions = None
    
    # Multi-run optimization with differential evolution using different seeds
    for i, initial_config in enumerate(initial_configs):
        try:
            # Run differential evolution with better parameters
            bounds = [(-15, 15)] * len(initial_config)
            
            def de_objective(x):
                _, side_length = evaluate_fitness_adaptive(x)
                return -np.log(side_length) if side_length > 1e-6 else 1e10
            
            # Run with multiple different seeds for robustness
            for seed in [i*123 + 456, i*789 + 101]:
                result = differential_evolution(
                    de_objective, bounds,
                    maxiter=150, popsize=20, 
                    seed=seed, disp=False,
                    polish=True
                )
                
                if result.success:
                    _, side_length = evaluate_fitness_adaptive(result.x)
                    if side_length < best_side_length and side_length > 0:
                        best_side_length = side_length
                        best_positions = result.x.reshape(-1, 3)
                        
        except Exception as e:
            continue
    
    # If no improvement found, use the best initial configuration
    if best_positions is None:
        best_positions = initial_configs[0].reshape(-1, 3)
        best_side_length = compute_outer_hexagon_side_length(best_positions)
    
    # Final local refinement with adaptive penalty system
    flattened_best = best_positions.flatten()
    final_refined = refine_with_local_search(flattened_best, 50)
    final_positions = final_refined.reshape(-1, 3)
    
    # Final validation
    _, final_side_length = evaluate_fitness_adaptive(final_refined)
    
    return final_positions, final_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Optimize
    inner_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    
    # Set outer hexagon parameters (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END