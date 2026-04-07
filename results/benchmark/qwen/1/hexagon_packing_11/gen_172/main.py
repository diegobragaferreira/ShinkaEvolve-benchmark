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

class QuadTree:
    """Simple quadtree for spatial indexing"""
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary  # (x_min, y_min, x_max, y_max)
        self.capacity = capacity
        self.points = []
        self.divided = False

    def intersects(self, rect):
        """Check if rectangle intersects with boundary"""
        x_min, y_min, x_max, y_max = self.boundary
        rx_min, ry_min, rx_max, ry_max = rect
        return not (rx_max < x_min or rx_min > x_max or ry_max < y_min or ry_min > y_max)

    def contains(self, point):
        """Check if point is in boundary"""
        x, y = point
        x_min, y_min, x_max, y_max = self.boundary
        return x_min <= x <= x_max and y_min <= y <= y_max

    def subdivide(self):
        """Divide node into four quadrants"""
        x_min, y_min, x_max, y_max = self.boundary
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        # Create four quadrants
        ne = (x_mid, y_min, x_max, y_mid)
        nw = (x_min, y_min, x_mid, y_mid)
        se = (x_mid, y_mid, x_max, y_max)
        sw = (x_min, y_mid, x_mid, y_max)

        self.northeast = QuadTree(ne, self.capacity)
        self.northwest = QuadTree(nw, self.capacity)
        self.southeast = QuadTree(se, self.capacity)
        self.southwest = QuadTree(sw, self.capacity)

        self.divided = True

    def insert(self, point, index):
        """Insert point into quadtree"""
        if not self.contains(point):
            return False

        if len(self.points) < self.capacity and not self.divided:
            self.points.append((point, index))
            return True

        if not self.divided:
            self.subdivide()

        # Try to insert into children
        for qt in [self.northeast, self.northwest, self.southeast, self.southwest]:
            if qt.insert(point, index):
                return True

        return False

    def query_range(self, range_rect):
        """Find all points in the range"""
        found_points = []

        if not self.intersects(range_rect):
            return found_points

        # Check points in current node
        for point, index in self.points:
            x, y = point
            rx_min, ry_min, rx_max, ry_max = range_rect
            if rx_min <= x <= rx_max and ry_min <= y <= ry_max:
                found_points.append(index)

        if self.divided:
            for qt in [self.northeast, self.northwest, self.southeast, self.southwest]:
                found_points.extend(qt.query_range(range_rect))

        return found_points

def build_spatial_hash_grid(positions, angles, cell_size=2.5):
    """Build spatial hash grid for collision detection"""
    if len(positions) == 0:
        return {}

    # Create hash grid where each cell is a list of hexagon indices
    grid = {}

    # For each hexagon, determine which cells it occupies
    for i in range(len(positions)):
        pos = positions[i]
        angle = angles[i]
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)

        # Get bounding box of hexagon
        min_x = min(vx for vx, vy in hex_vertices)
        max_x = max(vx for vx, vy in hex_vertices)
        min_y = min(vy for vx, vy in hex_vertices)
        max_y = max(vy for vx, vy in hex_vertices)

        # Determine which grid cells this hexagon spans
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

def get_candidates_from_hash_grid(grid, positions, angles, hex_idx, cell_size=2.5):
    """Get candidate hexagons for collision checking using spatial hash"""
    pos = positions[hex_idx]
    angle = angles[hex_idx]
    hex_vertices = hexagon_vertices(pos[0], pos[1], angle)

    # Get bounding box of target hexagon
    min_x = min(vx for vx, vy in hex_vertices)
    max_x = max(vx for vx, vy in hex_vertices)
    min_y = min(vy for vx, vy in hex_vertices)
    max_y = max(vy for vx, vy in hex_vertices)

    # Determine which grid cells to check
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

def fast_collision_check_with_hash(grid, positions, angles, i, j, cell_size=2.5):
    """Fast collision check using spatial hash"""
    if i == j:
        return False

    # Quick distance check first
    pos_i = positions[i]
    pos_j = positions[j]
    dist_sq = (pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2

    # If too far apart, no need to do detailed check
    if dist_sq > 4.0:  # Distance > 2 (unit hexagons can't touch)
        return False

    # Check if they're in the same or neighboring cells
    # If they're in different cells, we can skip detailed check
    if grid is not None:
        # Get bounding boxes for quick overlap check
        v1 = hexagon_vertices(pos_i[0], pos_i[1], angles[i])
        v2 = hexagon_vertices(pos_j[0], pos_j[1], angles[j])

        # Get bounding rectangles
        min_x1 = min(vx for vx, vy in v1)
        max_x1 = max(vx for vx, vy in v1)
        min_y1 = min(vy for vx, vy in v1)
        max_y1 = max(vy for vx, vy in v1)

        min_x2 = min(vx for vx, vy in v2)
        max_x2 = max(vx for vx, vy in v2)
        min_y2 = min(vy for vx, vy in v2)
        max_y2 = max(vy for vx, vy in v2)

        # Quick overlap check
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            return False

    # Detailed check
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

    # Build spatial hash grid for collision detection if requested
    grid = None
    if use_spatial_index:
        grid = build_spatial_hash_grid(positions, angles)

    # Check for overlaps
    # Early termination optimization: only check neighbors in spatial hash
    if use_spatial_index and grid is not None:
        # For each hexagon, get candidates from spatial hash and check collisions
        for i in range(11):
            candidates = get_candidates_from_hash_grid(grid, positions, angles, i)
            for j in candidates:
                if i >= j:  # Avoid double checking and self-checking
                    continue
                if not fast_collision_check_with_hash(grid, positions, angles, i, j):
                    continue
                # If we reach here, there's a collision
                return 1e10  # Penalty for overlap
    else:
        # Fallback to direct checking
        for i in range(11):
            for j in range(i+1, 11):
                if not hexagon_overlap_fast(positions[i][0], positions[i][1], angles[i],
                                            positions[j][0], positions[j][1], angles[j]):
                    continue
                # If we reach here, there's a collision
                return 1e10  # Penalty for overlap

    # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
    return -1.0 / outer_radius

def generate_initial_population(num_starts=3):
    """Generate multiple initial configurations using hexagonal packing approach"""
    initial_populations = []

    # Base pattern: hexagonal pattern with center and surrounding ring
    base_positions = []
    base_angles = []

    # Center hexagon
    base_positions.append([0.0, 0.0])
    base_angles.append(0.0)

    # Surrounding hexagons in ring
    for i in range(6):
        angle = i * 60
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        base_positions.append([x, y])
        base_angles.append(0.0)

    # Additional positions for remaining hexagons - arrange in hexagonal pattern
    additional_positions = [
        (-3.0, 1.0), (3.0, 1.0),
        (-3.0, -1.0), (3.0, -1.0),
        (0.0, 3.0), (0.0, -3.0),
        (1.5, 2.6), (-1.5, -2.6),
        (-1.5, 2.6), (1.5, -2.6)
    ]

    for pos in additional_positions:
        if len(base_positions) < 11:
            base_positions.append(list(pos))
            base_angles.append(0.0)

    # Ensure we have exactly 11 positions
    while len(base_positions) < 11:
        base_positions.append([0.0, 0.0])
        base_angles.append(0.0)

    # Generate different variations
    for start in range(num_starts):
        # Create a slightly different initial configuration for each start
        initial_positions = [pos[:] for pos in base_positions]  # Copy
        initial_angles = [ang for ang in base_angles]  # Copy

        # Add small random perturbations
        for i in range(len(initial_positions)):
            if i > 0:  # Don't perturb center hexagon significantly
                initial_positions[i][0] += random.uniform(-0.2, 0.2)
                initial_positions[i][1] += random.uniform(-0.2, 0.2)
                initial_angles[i] += random.uniform(-5, 5)

        # Flatten initial solution
        initial_solution = []
        for pos in initial_positions[:11]:
            initial_solution.extend(pos)
        initial_solution.extend(initial_angles[:11])
        initial_solution = np.array(initial_solution)

        initial_populations.append(initial_solution)

    return initial_populations

def optimize_hexagon_packing():
    """Main optimization function with multi-phase approach"""
    # Phase 1: Multi-start differential evolution with better bounds
    initial_populations = generate_initial_population(5)

    best_result = None
    best_score = float('inf')

    # Run optimization from multiple starting points
    for i, initial_solution in enumerate(initial_populations):
        try:
            # Set bounds for optimization with better ranges
            bounds = []
            # Position bounds - wider range for exploration
            for _ in range(22):
                bounds.append((-12.0, 12.0))  # X and Y coordinates
            # Angle bounds
            for _ in range(11):
                bounds.append((0.0, 360.0))   # Rotation angles

            # Use adaptive DE parameters - faster convergence
            maxiter = 100
            popsize = 15

            # Run differential evolution with better strategy
            result = differential_evolution(
                lambda sol: evaluate_solution(sol, use_spatial_index=(i < 3)),  # Use spatial index for first 3 starts
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=42+i,  # Different seed for each start
                disp=False,
                tol=1e-6,
                strategy='best1bin'
            )

            # Evaluate final result with full check
            final_score = evaluate_solution(result.x, use_spatial_index=False)

            if final_score < best_score:
                best_score = final_score
                best_result = result

        except Exception as e:
            print(f"Start {i} failed: {e}")
            continue

    if best_result is None:
        # Fallback to simple solution
        raise RuntimeError("All optimization attempts failed")

    # Phase 2: Local refinement with simulated annealing for fine-tuning
    final_positions = best_result.x[:22].reshape(-1, 2)
    final_angles = best_result.x[22:]

    # Apply enhanced refinement with better cooling schedule
    refined_positions, refined_angles = enhanced_simulated_annealing_refinement(final_positions, final_angles)

    return refined_positions, refined_angles

def enhanced_simulated_annealing_refinement(positions, angles):
    """Apply enhanced simulated annealing for fine-tuning with adaptive cooling"""
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_score = evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))

    # Parameters with better tuning for hexagon packing
    temperature = 3.0
    cooling_rate = 0.9998  # Very slow cooling for thorough exploration
    min_temperature = 0.0001
    max_iterations = 1500  # More iterations for better refinement

    # Track convergence
    last_improvement = 0
    patience = 50  # Stop if no improvement for 50 iterations

    # Temperature schedule with more careful cooling
    for iteration in range(max_iterations):
        if temperature < min_temperature:
            break

        # Try multiple random perturbations per iteration for better exploration
        for _ in range(5):  # Increased to 5 perturbations per iteration
            # Try a random perturbation
            perturbed_positions = best_positions.copy()
            perturbed_angles = best_angles.copy()

            # Pick a random hexagon to perturb
            hex_idx = random.randint(0, 10)

            # Perturb position with adaptive magnitude
            pos_magnitude = 0.03 * (temperature / 3.0)  # Even smaller steps at lower temp
            perturbed_positions[hex_idx][0] += np.random.normal(0, pos_magnitude)
            perturbed_positions[hex_idx][1] += np.random.normal(0, pos_magnitude)

            # Perturb angle with adaptive magnitude
            angle_magnitude = 1.0 * (temperature / 3.0)  # Smaller steps at lower temp
            perturbed_angles[hex_idx] += np.random.normal(0, angle_magnitude)

            # Evaluate new solution
            new_score = evaluate_solution(np.concatenate([perturbed_positions.flatten(), perturbed_angles]))

            # Accept or reject
            if new_score < best_score:
                best_score = new_score
                best_positions = perturbed_positions
                best_angles = perturbed_angles
                last_improvement = iteration
            else:
                # Accept with probability based on temperature
                delta = new_score - best_score
                acceptance_prob = np.exp(-delta / temperature)
                if random.random() < acceptance_prob:
                    best_score = new_score
                    best_positions = perturbed_positions
                    best_angles = perturbed_angles
                    last_improvement = iteration

        # Check for early stopping if no improvement
        if iteration - last_improvement > patience:
            break

        # Cool down
        temperature *= cooling_rate

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