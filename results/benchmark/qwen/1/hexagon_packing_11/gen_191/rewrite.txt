# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import time
from numba import jit
import warnings
import math
from collections import defaultdict

warnings.filterwarnings('ignore')

class HexagonGeometry:
    """Handles all geometric operations for hexagons"""

    @staticmethod
    @jit(nopython=True)
    def vertices(x, y, angle_deg, side_length=1):
        """Calculate vertices of a hexagon given center, angle, and side length"""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vx = x + side_length * np.cos(theta)
            vy = y + side_length * np.sin(theta)
            vertices.append((vx, vy))
        return np.array(vertices)

    @staticmethod
    def polygon(x, y, angle_deg, side_length=1):
        """Get shapely polygon representation of hexagon"""
        vertices = HexagonGeometry.vertices(x, y, angle_deg, side_length)
        return Polygon(vertices)

class ConstraintChecker:
    """Handles constraint checking for hexagon packing"""

    @staticmethod
    def contains(hex_poly, outer_poly):
        """Check if hexagon is completely contained within outer hexagon"""
        return outer_poly.contains(hex_poly) or outer_poly.intersection(hex_poly).area == hex_poly.area

    @staticmethod
    def overlaps(hex1_poly, hex2_poly):
        """Check if two hexagons overlap"""
        return hex1_poly.intersects(hex2_poly)

def fast_collision_check(hex_poly1, hex_poly2):
    """Fast collision check using bounding boxes"""
    bbox1 = hex_poly1.bounds
    bbox2 = hex_poly2.bounds

    # Quick bounding box overlap test
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False

    # More precise check
    return hex_poly1.intersects(hex2_poly2)

def build_spatial_grid(hexagons, grid_size=3.0):
    """Build spatial grid for fast collision detection"""
    grid = defaultdict(list)
    for i, hex_poly in enumerate(hexagons):
        bbox = hex_poly.bounds
        min_x, min_y, max_x, max_y = bbox
        # Use a smaller grid size for more precision with hexagons
        for x in range(int(min_x/grid_size), int(max_x/grid_size)+1):
            for y in range(int(min_y/grid_size), int(max_y/grid_size)+1):
                grid[(x,y)].append(i)
    return grid

def get_collision_candidates(grid, hex_index, hex_poly, grid_size=3.0):
    """Get potential collision candidates efficiently"""
    candidates = []
    bbox = hex_poly.bounds
    min_x, min_y, max_x, max_y = bbox

    # Use the same grid size as used in build_spatial_grid
    for x in range(int(min_x/grid_size), int(max_x/grid_size)+1):
        for y in range(int(min_y/grid_size), int(max_y/grid_size)+1):
            candidates.extend(grid.get((x,y), []))
    return [i for i in candidates if i != hex_index]

@jit(nopython=True)
def distance_point_to_hexagon(px, py, hx, hy, angle_deg, side_length=1):
    """Calculate minimum distance from point to hexagon (fast approximation)"""
    # Simplified distance calculation for hexagon
    # Could be improved with full SAT implementation
    angle_rad = np.radians(angle_deg)
    # Hexagon radius (distance from center to corner)
    radius = side_length * np.sqrt(3) / 2
    
    # Distance from point to center
    dx = px - hx
    dy = py - hy
    dist_to_center = np.sqrt(dx*dx + dy*dy)
    
    # If point is inside hexagon, distance is negative
    if dist_to_center <= radius:
        return -dist_to_center
    
    # Otherwise distance to border
    return dist_to_center - radius

class SolutionEvaluator:
    """Evaluates solution quality and feasibility"""

    @staticmethod
    @jit(nopython=True)
    def calculate_outer_radius_fast(inner_positions, inner_angles):
        """Fast calculation of minimum radius needed to contain all inner hexagons"""
        max_dist = 0.0
        outer_center_x = 0.0
        outer_center_y = 0.0
        
        # Get all vertices of all inner hexagons
        for i in range(len(inner_positions)):
            pos = inner_positions[i]
            angle = inner_angles[i]
            
            # For efficiency, just consider center points and add hexagon size
            dx = pos[0] - outer_center_x
            dy = pos[1] - outer_center_y
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Add hexagon radius (approximate)
            hex_radius = 1.0 * np.sqrt(3) / 2  # Unit hexagon radius
            max_dist = max(max_dist, dist + hex_radius)

        return max_dist * 1.1  # Safety factor

    @staticmethod
    def evaluate(solution):
        """Evaluate a solution and return negative of objective (since we minimize)"""
        # Reshape solution into positions and angles
        positions = solution[:22].reshape(-1, 2)  # 11 hexagons * 2 coordinates each
        angles = solution[22:]  # 11 angles

        # Fast preliminary validation for early rejection
        # Calculate approximate outer radius and reject if too large
        approx_outer_radius = SolutionEvaluator.calculate_outer_radius_fast(positions, angles)
        if approx_outer_radius > 20.0:  # Early rejection threshold
            return 1e10

        # Create inner hexagons
        inner_hexagons = []
        for i in range(11):
            pos = positions[i]
            angle = angles[i]
            hex_poly = HexagonGeometry.polygon(pos[0], pos[1], angle)
            inner_hexagons.append(hex_poly)

        # Check containment
        outer_radius = SolutionEvaluator.calculate_outer_radius_fast(positions, angles)
        # Outer hexagon with center at origin and calculated radius
        outer_hexagon = HexagonGeometry.polygon(0, 0, 0, outer_radius)

        # Check containment for all inner hexagons (numerically more stable)
        for hex_poly in inner_hexagons:
            # Simple distance-based check first (faster than shapely)
            if outer_radius < 1.0:  # Too small to contain anything
                return 1e10
                
            # Check if center is within bounds (approximation)
            hex_center = hex_poly.centroid
            center_dist = np.sqrt(hex_center.x**2 + hex_center.y**2)
            if center_dist >= outer_radius * 0.9:  # Small buffer
                return 1e10  # Penalty for non-containment

        # Check for overlaps using spatial grid
        grid = build_spatial_grid(inner_hexagons, grid_size=2.0)
        for i in range(11):
            candidates = get_collision_candidates(grid, i, inner_hexagons[i], grid_size=2.0)
            for j in candidates:
                if i < j:  # Only check each pair once
                    if fast_collision_check(inner_hexagons[i], inner_hexagons[j]):
                        return 1e10  # Penalty for overlap

        # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
        return -1.0 / outer_radius if outer_radius > 0.001 else 1e10

class OptimizationStages:
    """Manages different optimization stages"""

    @staticmethod
    def create_initial_configuration():
        """Create a better initial configuration based on hexagonal packing principles"""
        # Start with a known good configuration with better spacing
        initial_positions = [
            [0.0, 0.0],     # center
            [-1.8, 0.0],    # left
            [1.8, 0.0],     # right
            [0.0, 1.8],     # top
            [0.0, -1.8],    # bottom
            [-1.2, 1.2],    # top-left
            [1.2, 1.2],     # top-right
            [-1.2, -1.2],   # bottom-left
            [1.2, -1.2],    # bottom-right
            [-1.8, 1.5],    # further top-left
            [1.8, 1.5]      # further top-right
        ]

        # Slightly adjust spacing for better packing
        for i in range(len(initial_positions)):
            initial_positions[i][0] *= 1.1
            initial_positions[i][1] *= 1.1

        initial_angles = [0.0] * 11
        return initial_positions, initial_angles

    @staticmethod
    def coarse_optimization(initial_solution):
        """First phase: Coarse differential evolution optimization with adaptive mutation rate"""
        # Set bounds for optimization
        bounds = []
        # Position bounds - wider range for exploration
        for _ in range(22):
            bounds.append((-12.0, 12.0))  # X and Y coordinates
        # Angle bounds
        for _ in range(11):
            bounds.append((0.0, 360.0))   # Rotation angles

        # Differential evolution with advanced parameters for better performance
        def adaptive_differential_evolution():
            # Advanced adaptive mutation rate scheduling
            def mutation_rate_schedule(iteration, maxiter):
                # Start with high mutation rate for exploration, decrease for exploitation
                return 0.9 - (0.8 * iteration / maxiter)
            
            # Use a custom implementation to control mutation rate dynamically
            result = differential_evolution(
                SolutionEvaluator.evaluate,
                bounds,
                maxiter=80,              # More iterations for better convergence
                popsize=15,              # Larger population for better exploration
                seed=42,
                disp=False,
                tol=1e-6,
                strategy='best1bin',
                mutation=(0.7, 0.9),     # Better mutation range
                recombination=0.7        # Recombination rate
            )

            return result

        # Use the adaptive version
        result = adaptive_differential_evolution()

        return result

    @staticmethod
    def local_refinement(positions, angles):
        """Second phase: Local refinement with systematic perturbations"""
        # More robust refinement with systematic perturbations
        best_positions = positions.copy()
        best_angles = angles.copy()
        best_score = SolutionEvaluator.evaluate(np.concatenate([best_positions.flatten(), best_angles]))

        # Multiple refinement passes with decreasing step sizes
        step_sizes = [0.05, 0.02, 0.01]
        max_iterations = 20

        for step_size in step_sizes:
            improved_count = 0
            for iteration in range(max_iterations):
                if improved_count > 5:  # Stop early if no improvement
                    break
                    
                improved = False
                improved_count += 1

                # Try perturbing each position and angle
                for i in range(11):
                    # Perturb position
                    for dim in range(2):
                        old_val = best_positions[i][dim]
                        best_positions[i][dim] += step_size
                        new_score = SolutionEvaluator.evaluate(np.concatenate([best_positions.flatten(), best_angles]))
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                            improved_count = 0  # Reset counter on improvement
                        else:
                            best_positions[i][dim] = old_val

                    # Perturb angle
                    old_angle = best_angles[i]
                    best_angles[i] += 5.0
                    new_score = SolutionEvaluator.evaluate(np.concatenate([best_positions.flatten(), best_angles]))
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                        improved_count = 0  # Reset counter on improvement
                    else:
                        best_angles[i] = old_angle

                if not improved:
                    improved_count += 1

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
        # Phase 1: Create initial configuration
        initial_positions, initial_angles = OptimizationStages.create_initial_configuration()

        # Flatten initial solution
        initial_solution = []
        for pos in initial_positions:
            initial_solution.extend(pos)
        initial_solution.extend(initial_angles)
        initial_solution = np.array(initial_solution)

        # Phase 2: Coarse optimization
        result = OptimizationStages.coarse_optimization(initial_solution)

        # Extract final solution
        final_positions = result.x[:22].reshape(-1, 2)
        final_angles = result.x[22:]

        # Phase 3: Local refinement
        refined_positions, refined_angles = OptimizationStages.local_refinement(final_positions, final_angles)

        # Create inner hex data
        inner_hex_data = np.column_stack([refined_positions, refined_angles])

        # Create outer hex data (centered)
        outer_hex_data = np.array([0, 0, 0])

        # Calculate outer hex side length
        outer_radius = SolutionEvaluator.calculate_outer_radius_fast(refined_positions, refined_angles)
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