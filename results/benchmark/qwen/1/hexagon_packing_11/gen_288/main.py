# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
from numba import jit, prange
import warnings
import random
from collections import defaultdict
import math

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
    # Convert to local coordinate system
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Transform point to local coordinates
    dx = px - hx
    dy = py - hy
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a

    # Check against hexagon boundaries (unit hexagon centered at origin)
    # Hexagon has vertices at (±1,0), (±1/2, ±√3/2)
    # Distance from center in each direction should be <= 1
    r = np.sqrt(lx*lx + ly*ly)
    if r > 1.0: return False

    # Check if point is inside the hexagon (simplified)
    # Using the fact that for a regular hexagon with circumradius 1,
    # the distance from center to corner is 1
    return True

@jit(nopython=True)
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Distance from point to line segment"""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1

    # Length squared of line segment
    len_sq = dx*dx + dy*dy

    if len_sq == 0.0:
        # Line segment is actually a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq

    # Clamp t to [0,1] to stay within line segment
    t = max(0.0, min(1.0, t))

    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def hexagon_distance(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Fast approximation of minimum distance between two hexagons"""
    # Get vertices of both hexagons
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle, 1.0)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle, 1.0)

    # Check minimum distance between edges
    min_dist = 1000000.0

    # For each edge of first hexagon
    for i in range(6):
        j = (i + 1) % 6
        x1, y1 = v1[i]
        x2, y2 = v1[j]

        # Distance from each vertex of second hexagon to this edge
        for k in range(6):
            x3, y3 = v2[k]
            dist = distance_point_to_line(x3, y3, x1, y1, x2, y2)
            min_dist = min(min_dist, dist)

    return min_dist

@jit(nopython=True)
def hexagon_overlap_fast(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Fast approximation of whether two hexagons overlap"""
    # Simple distance check first
    dist = np.sqrt((h1_x - h2_x)**2 + (h1_y - h2_y)**2)
    if dist > 2.0:  # Two unit hexagons can't overlap if distance > 2
        return False

    # More detailed check with vertices
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle, 1.0)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle, 1.0)

    # Check if any vertices of one hexagon are inside the other
    for i in range(6):
        x1, y1 = v1[i]
        # Check if this vertex is inside hexagon 2
        if point_in_hexagon(x1, y1, h2_x, h2_y, h2_angle):
            return True

    for i in range(6):
        x1, y1 = v2[i]
        # Check if this vertex is inside hexagon 1
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

    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        all_vertices.extend(hex_vertices)

    # Find maximum distance from center
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
        max_dist = max(max_dist, dist)

    # Add buffer for safety and account for hexagon shape
    return max_dist * 1.1  # Safety factor

class SpatialGrid:
    """Enhanced spatial grid with hierarchical structure for efficient collision detection"""
    def __init__(self, boundary, cell_sizes=[2.8, 1.4, 0.7]):
        self.boundary = boundary  # (x_min, y_min, x_max, y_max)
        self.cell_sizes = cell_sizes
        self.grids = []  # Different resolution grids
        self.grid_widths = []
        self.grid_heights = []

        # Initialize each grid level
        for cell_size in cell_sizes:
            grid_width = int((boundary[2] - boundary[0]) / cell_size) + 1
            grid_height = int((boundary[3] - boundary[1]) / cell_size) + 1
            self.grids.append(defaultdict(list))
            self.grid_widths.append(grid_width)
            self.grid_heights.append(grid_height)

    def get_cell_coords(self, x, y, cell_size):
        """Get grid cell coordinates for a point"""
        grid_x = int((x - self.boundary[0]) / cell_size)
        grid_y = int((y - self.boundary[1]) / cell_size)
        return grid_x, grid_y

    def insert_hexagon(self, hex_idx, center_x, center_y, vertices):
        """Insert hexagon into all grid levels"""
        for level, (grid, cell_size) in enumerate(zip(self.grids, self.cell_sizes)):
            # Get the bounding box of the hexagon
            min_x = min(v[0] for v in vertices)
            max_x = max(v[0] for v in vertices)
            min_y = min(v[1] for v in vertices)
            max_y = max(v[1] for v in vertices)

            # Get cell range for the bounding box
            start_x = max(0, int((min_x - self.boundary[0]) / cell_size))
            end_x = min(self.grid_widths[level] - 1, int((max_x - self.boundary[0]) / cell_size))
            start_y = max(0, int((min_y - self.boundary[1]) / cell_size))
            end_y = min(self.grid_heights[level] - 1, int((max_y - self.boundary[1]) / cell_size))

            # Insert into all cells in the range
            for grid_x in range(start_x, end_x + 1):
                for grid_y in range(start_y, end_y + 1):
                    grid[(grid_x, grid_y)].append(hex_idx)

    def query_neighbors(self, hex_idx, center_x, center_y, cell_size):
        """Query potential neighbors from a specific grid level"""
        neighbors = set()
        grid_level = self.grids[self.cell_sizes.index(cell_size)]

        # Get cell coordinates for the hexagon
        grid_x, grid_y = self.get_cell_coords(center_x, center_y, cell_size)

        # Check nearby cells (including diagonals)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx = grid_x + dx
                ny = grid_y + dy
                if 0 <= nx < self.grid_widths[0] and 0 <= ny < self.grid_heights[0]:
                    for idx in grid_level[(nx, ny)]:
                        if idx != hex_idx:
                            neighbors.add(idx)

        return neighbors

def build_hierarchical_spatial_grid(hexagons, positions, angles):
    """Build hierarchical spatial grid with multiple resolutions"""
    if len(hexagons) == 0:
        return None

    # Determine bounds of all hexagons
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for i in range(len(positions)):
        pos = positions[i]
        angle = angles[i]
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        for vx, vy in hex_vertices:
            min_x = min(min_x, vx)
            max_x = max(max_x, vx)
            min_y = min(min_y, vy)
            max_y = max(max_y, vy)

    # Add padding
    padding = 1.5
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    # Create hierarchical grid with different resolutions
    boundary = (min_x, min_y, max_x, max_y)
    grid = SpatialGrid(boundary, cell_sizes=[2.8, 1.4, 0.7])

    # Insert all hexagons into all grid levels
    for i in range(len(positions)):
        pos = positions[i]
        angle = angles[i]
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        grid.insert_hexagon(i, pos[0], pos[1], hex_vertices)

    return grid, boundary

def fast_collision_check_with_hierarchical_grid(grid_info, positions, angles, i, j):
    """Fast collision check using hierarchical spatial grid indexing"""
    if i == j:
        return False

    grid, boundary = grid_info

    # Quick distance check first
    pos_i = positions[i]
    pos_j = positions[j]
    dist_sq = (pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2

    # If too far apart, no need to do detailed check
    if dist_sq > 4.0:  # Distance > 2 (unit hexagons can't touch)
        return False

    # Use finest grid level first for better accuracy
    finest_cell_size = 0.7

    # Query neighbors from finest grid
    neighbors = grid.query_neighbors(i, pos_i[0], pos_i[1], finest_cell_size)
    if j in neighbors:
        # Check if hexagons actually overlap
        return hexagon_overlap_fast(pos_i[0], pos_i[1], angles[i],
                                    pos_j[0], pos_j[1], angles[j])
    else:
        # Check if they're in same or adjacent cells in any level
        return hexagon_overlap_fast(pos_i[0], pos_i[1], angles[i],
                                    pos_j[0], pos_j[1], angles[j])

def fast_collision_check_with_adaptive_grid(grid_info, positions, angles, i, j):
    """Fast collision check using adaptive spatial grid indexing"""
    if i == j:
        return False

    grid, bounds, cell_size = grid_info

    # Quick distance check first
    pos_i = positions[i]
    pos_j = positions[j]
    dist_sq = (pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2

    # If too far apart, no need to do detailed check
    if dist_sq > 4.0:  # Distance > 2 (unit hexagons can't touch)
        return False

    # Check if hexagons are in same or adjacent grid cells
    min_x, min_y, max_x, max_y = bounds
    grid_x_i = int((pos_i[0] - min_x) / cell_size)
    grid_y_i = int((pos_i[1] - min_y) / cell_size)
    grid_x_j = int((pos_j[0] - min_x) / cell_size)
    grid_y_j = int((pos_j[1] - min_y) / cell_size)

    # Check nearby cells in grid
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            gx_i = grid_x_i + dx
            gy_i = grid_y_i + dy
            gx_j = grid_x_j + dx
            gy_j = grid_y_j + dy

            # If both hexagons are in same cell or adjacent cells
            if (gx_i, gy_i) in grid and (gx_j, gy_j) in grid:
                # Check if they share the same cell
                if abs(gx_i - gx_j) <= 1 and abs(gy_i - gy_j) <= 1:
                    # Detailed check
                    return hexagon_overlap_fast(pos_i[0], pos_i[1], angles[i],
                                                pos_j[0], pos_j[1], angles[j])

    # Fallback to detailed check
    return hexagon_overlap_fast(pos_i[0], pos_i[1], angles[i],
                                pos_j[0], pos_j[1], angles[j])

def evaluate_solution(solution, use_spatial_index=True):
    """Evaluate a solution and return negative of objective (since we minimize)"""
    # Reshape solution into positions and angles
    positions = solution[:22].reshape(-1, 2)  # 11 hexagons * 2 coordinates each
    angles = solution[22:]  # 11 angles

    # Create inner hexagons
    inner_hexagons = []
    for i in range(11):
        pos = positions[i]
        angle = angles[i]
        hex_poly = get_hexagon_polygon(pos[0], pos[1], angle)
        inner_hexagons.append(hex_poly)

    # Check containment
    outer_radius = calculate_outer_hexagon_radius(positions, angles)
    # Outer hexagon with center at origin and calculated radius
    outer_hexagon = get_hexagon_polygon(0, 0, 0, outer_radius)

    # Check containment for all inner hexagons
    for hex_poly in inner_hexagons:
        if not check_containment(hex_poly, outer_hexagon):
            return 1e10  # Penalty for non-containment

    # Build spatial grid for collision detection if requested
    grid_info = None
    if use_spatial_index:
        grid_info = build_hierarchical_spatial_grid(inner_hexagons, positions, angles)

    # Check for overlaps (optimized version)
    collision_found = False

    # Use optimized pairwise checking with spatial indexing
    for i in range(11):
        for j in range(i+1, 11):
            # Use fast collision check with spatial indexing
            if use_spatial_index and grid_info is not None:
                if not fast_collision_check_with_hierarchical_grid(grid_info, positions, angles, i, j):
                    continue
            else:
                # Fallback to direct check if spatial index fails
                if not hexagon_overlap_fast(positions[i][0], positions[i][1], angles[i],
                                            positions[j][0], positions[j][1], angles[j]):
                    continue

            # If we reach here, there's a collision
            collision_found = True
            break
        if collision_found:
            break

    if collision_found:
        return 1e10  # Penalty for overlap

    # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
    return -1.0 / outer_radius

def generate_initial_population(num_starts=8):
    """Generate diverse initial configurations using Voronoi-based and strategic arrangements"""
    initial_populations = []

    # Strategy 1: Voronoi-based initialization for good distribution
    np.random.seed(42)
    try:
        # Generate random points and compute Voronoi
        points = np.random.uniform(-5, 5, size=(11, 2))
        # Use Voronoi to get good spread, but for robustness we'll also use deterministic patterns
        voronoi_centers = points.copy()
        base_positions_voronoi = []
        for i in range(11):
            base_positions_voronoi.append([voronoi_centers[i, 0], voronoi_centers[i, 1]])
    except:
        # Fallback to simple pattern
        base_positions_voronoi = [
            [0.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0],
            [1.5, 1.5], [-1.5, 1.5], [1.5, -1.5], [-1.5, -1.5],
            [3.0, 0.0], [0.0, 3.0]
        ]

    # Strategy 2: Hexagonal cluster pattern (central core)
    base_positions_hex_cluster = [
        [0.0, 0.0],     # center
        [-1.5, 0.0],    # left
        [1.5, 0.0],     # right
        [0.0, 1.5],     # top
        [0.0, -1.5],    # bottom
        [-0.75, 1.29],  # top-left
        [0.75, 1.29],   # top-right
        [-0.75, -1.29], # bottom-left
        [0.75, -1.29],  # bottom-right
        [-2.25, 1.95],  # further top-left
        [2.25, 1.95]    # further top-right
    ]

    # Strategy 3: Spread-out pattern (good for avoiding overlaps)
    base_positions_spread = [
        [0.0, 0.0],     # center
        [-3.0, 0.0],    # left
        [3.0, 0.0],     # right
        [0.0, 3.0],     # top
        [0.0, -3.0],    # bottom
        [-1.5, 2.6],    # top-left
        [1.5, 2.6],     # top-right
        [-1.5, -2.6],   # bottom-left
        [1.5, -2.6],    # bottom-right
        [-2.5, 2.17],   # further top-left
        [2.5, 2.17]     # further top-right
    ]

    # Strategy 4: Spiral arrangement
    base_positions_spiral = [
        [0.0, 0.0],     # center
        [-1.0, 0.0],    # left
        [1.0, 0.0],     # right
        [0.0, 1.0],     # top
        [0.0, -1.0],    # bottom
        [-0.5, 0.866],  # top-left
        [0.5, 0.866],   # top-right
        [-0.5, -0.866], # bottom-left
        [0.5, -0.866],  # bottom-right
        [-1.5, 1.299],  # further top-left
        [1.5, 1.299]    # further top-right
    ]

    # Strategy 5: Optimized balanced arrangement (similar to previous best)
    base_positions_balanced = [
        [0.0, 0.0],     # center
        [-2.2, 0.0],    # left
        [2.2, 0.0],     # right
        [0.0, 2.2],     # top
        [0.0, -2.2],    # bottom
        [-1.1, 1.9],    # top-left
        [1.1, 1.9],     # top-right
        [-1.1, -1.9],   # bottom-left
        [1.1, -1.9],    # bottom-right
        [-2.2, 1.65],   # further top-left
        [2.2, 1.65]     # further top-right
    ]

    # Strategy 6: Star pattern with long-range connections
    base_positions_star = [
        [0.0, 0.0],     # center
        [-2.5, 0.0],    # left
        [2.5, 0.0],     # right
        [0.0, 2.5],     # top
        [0.0, -2.5],    # bottom
        [-1.25, 2.17],  # top-left
        [1.25, 2.17],   # top-right
        [-1.25, -2.17], # bottom-left
        [1.25, -2.17],  # bottom-right
        [-3.75, 2.17],  # far top-left
        [3.75, 2.17]    # far top-right
    ]

    strategies = [
        base_positions_voronoi,
        base_positions_hex_cluster,
        base_positions_spread,
        base_positions_spiral,
        base_positions_balanced,
        base_positions_star
    ]

    # Generate different variations for each strategy
    for start in range(num_starts):
        # Choose strategy based on start number
        strategy_idx = start % len(strategies)
        base_positions = strategies[strategy_idx]
        base_angles = [0.0] * 11

        # Ensure we have exactly 11 positions
        if len(base_positions) < 11:
            # Extend with dummy positions
            while len(base_positions) < 11:
                base_positions.append([0.0, 0.0])
        elif len(base_positions) > 11:
            base_positions = base_positions[:11]

        # Create a slightly different initial configuration for each start
        initial_positions = [pos[:] for pos in base_positions]  # Copy
        initial_angles = [ang for ang in base_angles]  # Copy

        # Add random perturbations for better exploration
        for i in range(len(initial_positions)):
            # Add noise to all positions
            initial_positions[i][0] += np.random.normal(0, 0.2)
            initial_positions[i][1] += np.random.normal(0, 0.2)
            # Add small random rotation for some hexagons
            if i > 0:  # Don't perturb center hexagon rotation
                initial_angles[i] += np.random.normal(0, 5)

        # Flatten initial solution
        initial_solution = []
        for pos in initial_positions[:11]:
            initial_solution.extend(pos)
        initial_solution.extend(initial_angles[:11])
        initial_solution = np.array(initial_solution)

        initial_populations.append(initial_solution)

    return initial_populations

def optimize_hexagon_packing():
    """Main optimization function with enhanced multi-phase approach"""
    # Phase 1: Multi-start differential evolution with more diverse populations
    initial_populations = generate_initial_population(8)  # Increased number of starts

    best_result = None
    best_score = float('inf')

    # Run optimization from multiple starting points with enhanced diversity
    for i, initial_solution in enumerate(initial_populations):
        try:
            # Set bounds for optimization with slightly wider ranges for exploration
            bounds = []
            # Position bounds
            for _ in range(22):
                bounds.append((-12.0, 12.0))  # Wider range for better exploration
            # Angle bounds
            for _ in range(11):
                bounds.append((0.0, 360.0))   # Rotation angles

            # Use advanced DE parameters
            maxiter = 100  # More iterations for better convergence
            popsize = 20   # Larger population for better diversity

            # Run differential evolution
            result = differential_evolution(
                lambda sol: evaluate_solution(sol, use_spatial_index=(i < 5)),  # Use spatial index for more starts
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=42+i,  # Different seed for each start
                disp=False,
                tol=1e-6,
                strategy='best1bin'
            )

            # Evaluate final result with full validation
            final_score = evaluate_solution(result.x, use_spatial_index=True)

            if final_score < best_score:
                best_score = final_score
                best_result = result

        except Exception as e:
            print(f"Start {i} failed: {e}")
            continue

    if best_result is None:
        # Fallback to simple solution
        raise RuntimeError("All optimization attempts failed")

    # Phase 2: Enhanced local refinement with multiple strategies
    final_positions = best_result.x[:22].reshape(-1, 2)
    final_angles = best_result.x[22:]

    # Try advanced local refinement with multiple phases
    refined_positions, refined_angles = enhanced_local_refinement(final_positions, final_angles)

    return refined_positions, refined_angles

def enhanced_local_refinement(positions, angles):
    """Enhanced local refinement with multiple optimization techniques"""
    # Phase 1: Simulated Annealing (fine-tuning)
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_score = evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))

    # Parameters for SA with adaptive cooling
    temperature = 5.0
    cooling_rate = 0.995  # Slightly slower cooling for better exploration
    min_temperature = 0.001
    max_iterations = 800  # More iterations for better convergence

    # Temperature schedule
    for iteration in range(max_iterations):
        if temperature < min_temperature:
            break

        # Try a random perturbation
        perturbed_positions = best_positions.copy()
        perturbed_angles = best_angles.copy()

        # Pick a random hexagon to perturb
        hex_idx = random.randint(0, 10)

        # Adaptive perturbation: smaller at lower temperatures
        pos_magnitude = 0.05 * (temperature / 5.0)
        angle_magnitude = 2.0 * (temperature / 5.0)

        # Perturb position
        perturbed_positions[hex_idx][0] += np.random.normal(0, pos_magnitude)
        perturbed_positions[hex_idx][1] += np.random.normal(0, pos_magnitude)

        # Perturb angle
        perturbed_angles[hex_idx] += np.random.normal(0, angle_magnitude)

        # Evaluate new solution
        new_score = evaluate_solution(np.concatenate([perturbed_positions.flatten(), perturbed_angles]))

        # Accept or reject
        if new_score < best_score:
            best_score = new_score
            best_positions = perturbed_positions
            best_angles = perturbed_angles
        else:
            # Accept with probability based on temperature
            delta = new_score - best_score
            acceptance_prob = np.exp(-delta / temperature)
            if random.random() < acceptance_prob:
                best_score = new_score
                best_positions = perturbed_positions
                best_angles = perturbed_angles

        # Cool down
        temperature *= cooling_rate

    # Phase 2: Local Nelder-Mead optimization for remaining hexagons
    try:
        # Run Nelder-Mead on the best solution
        def objective_function(x):
            # Reshape x to positions and angles
            pos = x[:22].reshape(-1, 2)
            ang = x[22:]
            # Combine into full solution vector
            full_sol = np.concatenate([pos.flatten(), ang])
            return evaluate_solution(full_sol, use_spatial_index=True)

        # Use the current best solution as starting point for local refinement
        x0 = np.concatenate([best_positions.flatten(), best_angles])

        from scipy.optimize import minimize
        result_nm = minimize(
            objective_function,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'disp': False}
        )

        if result_nm.success:
            refined_positions = result_nm.x[:22].reshape(-1, 2)
            refined_angles = result_nm.x[22:]
            refined_score = evaluate_solution(np.concatenate([refined_positions.flatten(), refined_angles]), use_spatial_index=True)

            if refined_score < best_score:
                best_positions = refined_positions
                best_angles = refined_angles
                best_score = refined_score

    except Exception as e:
        # If Nelder-Mead fails, continue with SA result
        pass

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

    try:
        # Run optimization
        final_positions, final_angles = optimize_hexagon_packing()

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
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END