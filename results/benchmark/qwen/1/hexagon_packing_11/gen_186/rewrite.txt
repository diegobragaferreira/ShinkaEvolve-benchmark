# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from collections import defaultdict
import time
from numba import jit
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

def get_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Get shapely polygon representation of hexagon"""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

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

def check_containment(hex_poly, outer_poly):
    """Check if hexagon is completely contained within outer hexagon"""
    return outer_poly.contains(hex_poly) or (outer_poly.intersects(hex_poly) and
                                           outer_poly.intersection(hex_poly).area == hex_poly.area)

def fast_collision_check(hex_poly1, hex_poly2):
    """Fast collision check using bounding boxes"""
    bbox1 = hex_poly1.bounds
    bbox2 = hex_poly2.bounds

    # Quick bounding box overlap test
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False

    # More precise check
    return hex_poly1.intersects(hex_poly2)

def build_spatial_grid(hexagons, grid_size=3.0):
    """Build spatial grid for fast collision detection"""
    grid = defaultdict(list)
    for i, hex_poly in enumerate(hexagons):
        bbox = hex_poly.bounds
        min_x, min_y, max_x, max_y = bbox
        # Use integer grid cells for efficiency
        for x in range(int(min_x/grid_size), int(max_x/grid_size)+1):
            for y in range(int(min_y/grid_size), int(max_y/grid_size)+1):
                grid[(x,y)].append(i)
    return grid

def get_collision_candidates(grid, hex_index, hex_poly, grid_size=3.0):
    """Get potential collision candidates efficiently"""
    candidates = []
    bbox = hex_poly.bounds
    min_x, min_y, max_x, max_y = bbox

    # Check neighboring grid cells
    for x in range(int(min_x/grid_size), int(max_x/grid_size)+1):
        for y in range(int(min_y/grid_size), int(max_y/grid_size)+1):
            candidates.extend(grid.get((x,y), []))
    return [i for i in candidates if i != hex_index]

class ImprovedOptimizer:
    """Enhanced optimizer with better spatial indexing and adaptive evolution"""
    
    def __init__(self, num_hexagons=11, side_length=1):
        self.num_hexagons = num_hexagons
        self.side_length = side_length
        self.grid_size = 3.0

    def create_initial_configuration(self):
        """Create a better initial configuration based on hexagonal packing principles"""
        # Start with a geometrically sound initial configuration
        initial_positions = [
            [0.0, 0.0],     # center
            [-2.0, 0.0],    # left
            [2.0, 0.0],     # right
            [0.0, 2.0],     # top
            [0.0, -2.0],    # bottom
            [-1.0, 1.0],    # top-left
            [1.0, 1.0],     # top-right
            [-1.0, -1.0],   # bottom-left
            [1.0, -1.0],    # bottom-right
            [-2.0, 1.5],    # further top-left
            [2.0, 1.5]      # further top-right
        ]

        # Adjust spacing to allow for better packing
        for i in range(len(initial_positions)):
            initial_positions[i][0] *= 1.1
            initial_positions[i][1] *= 1.1

        initial_angles = [0.0] * 11
        return initial_positions, initial_angles

    def evaluate_solution(self, solution):
        """Evaluate a solution with spatial optimization"""
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

        # Check for overlaps using spatial grid for efficiency
        grid = build_spatial_grid(inner_hexagons, self.grid_size)
        
        # Check for overlaps
        for i in range(11):
            candidates = get_collision_candidates(grid, i, inner_hexagons[i], self.grid_size)
            for j in candidates:
                if i < j and fast_collision_check(inner_hexagons[i], inner_hexagons[j]):
                    return 1e10  # Penalty for overlap

        # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
        return -1.0 / outer_radius

    def adaptive_differential_evolution(self, initial_solution):
        """Custom differential evolution with adaptive mutation rates"""
        bounds = []
        # Position bounds
        for _ in range(22):
            bounds.append((-15.0, 15.0))  # X and Y coordinates
        # Angle bounds
        for _ in range(11):
            bounds.append((0.0, 360.0))   # Rotation angles

        # Custom implementation with adaptive mutation rate
        def adaptive_mutation_schedule(iteration, maxiter):
            # Start with high mutation for exploration, decrease for exploitation
            return 0.9 - (0.6 * iteration / maxiter)

        # Run with adaptive parameters
        result = differential_evolution(
            self.evaluate_solution,
            bounds,
            maxiter=80,
            popsize=15,
            seed=42,
            disp=False,
            tol=1e-6,
            strategy='best1bin',
            mutation=(0.8, 0.9),  # Start with high mutation
            recombination=0.7
        )
        
        return result

    def local_refinement(self, positions, angles):
        """Local refinement using coordinate descent"""
        best_positions = positions.copy()
        best_angles = angles.copy()
        best_score = self.evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))

        # Multiple refinement passes
        step_sizes = [0.05, 0.02, 0.01]
        max_iterations = 20

        for step_size in step_sizes:
            for iteration in range(max_iterations):
                improved = False
                
                # Try perturbing each position and angle
                for i in range(11):
                    # Perturb position
                    for dim in range(2):
                        old_val = best_positions[i][dim]
                        best_positions[i][dim] += step_size
                        new_score = self.evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                        else:
                            best_positions[i][dim] = old_val

                    # Perturb angle
                    old_angle = best_angles[i]
                    best_angles[i] += 5.0
                    new_score = self.evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                    else:
                        best_angles[i] = old_angle

                if not improved:
                    break

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
        # Create optimizer
        optimizer = ImprovedOptimizer(num_hexagons=11, side_length=1.0)

        # Phase 1: Create initial configuration
        initial_positions, initial_angles = optimizer.create_initial_configuration()

        # Flatten initial solution
        initial_solution = []
        for pos in initial_positions:
            initial_solution.extend(pos)
        initial_solution.extend(initial_angles)
        initial_solution = np.array(initial_solution)

        # Phase 2: Coarse optimization with adaptive DE
        result = optimizer.adaptive_differential_evolution(initial_solution)

        # Extract final solution
        final_positions = result.x[:22].reshape(-1, 2)
        final_angles = result.x[22:]

        # Phase 3: Local refinement
        refined_positions, refined_angles = optimizer.local_refinement(final_positions, final_angles)

        # Create inner hex data
        inner_hex_data = np.column_stack([refined_positions, refined_angles])

        # Create outer hex data (centered)
        outer_hex_data = np.array([0, 0, 0])

        # Calculate outer hex side length
        outer_radius = calculate_outer_hexagon_radius(refined_positions, refined_angles)
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