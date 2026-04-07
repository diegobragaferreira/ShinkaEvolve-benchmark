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

    @staticmethod
    def get_bounding_box(hex_poly):
        """Get bounding box of hexagon for spatial indexing"""
        bounds = hex_poly.bounds
        return (bounds[0], bounds[1], bounds[2], bounds[3])

    @staticmethod
    def build_spatial_grid(hexagons, grid_size=5.0):
        """Build a simple spatial grid for fast collision detection"""
        grid = defaultdict(list)
        for i, hex_poly in enumerate(hexagons):
            bbox = ConstraintChecker.get_bounding_box(hex_poly)
            min_x, min_y, max_x, max_y = bbox
            for x in range(int(min_x/grid_size), int(max_x/grid_size)+1):
                for y in range(int(min_y/grid_size), int(max_y/grid_size)+1):
                    grid[(x,y)].append(i)
        return grid

    @staticmethod
    def fast_collision_check(hexagons, grid, i, j):
        """Fast collision check using spatial grid"""
        if i == j:
            return False

        # Check if hexagons are in same or adjacent grid cells
        bbox_i = ConstraintChecker.get_bounding_box(hexagons[i])
        bbox_j = ConstraintChecker.get_bounding_box(hexagons[j])

        # Simple overlap test first
        if not hexagons[i].intersects(hexagons[j]):
            return False

        # More precise check
        return hexagons[i].intersects(hexagons[j])

class SolutionEvaluator:
    """Evaluates solution quality and feasibility"""

    @staticmethod
    def calculate_outer_radius(inner_positions, inner_angles):
        """Calculate minimum radius needed to contain all inner hexagons"""
        max_dist = 0
        outer_center = (0, 0)

        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(inner_positions)):
            pos = inner_positions[i]
            angle = inner_angles[i]
            hex_vertices = HexagonGeometry.vertices(pos[0], pos[1], angle)
            all_vertices.extend(hex_vertices)

        # Find maximum distance from center
        for vertex in all_vertices:
            dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)

        # Add buffer for safety and account for hexagon shape
        return max_dist * 1.1  # Safety factor

    @staticmethod
    def evaluate(solution, use_spatial_index=True):
        """Evaluate a solution and return negative of objective (since we minimize)"""
        # Reshape solution into positions and angles
        positions = solution[:22].reshape(-1, 2)  # 11 hexagons * 2 coordinates each
        angles = solution[22:]  # 11 angles

        # Create inner hexagons
        inner_hexagons = []
        for i in range(11):
            pos = positions[i]
            angle = angles[i]
            hex_poly = HexagonGeometry.polygon(pos[0], pos[1], angle)
            inner_hexagons.append(hex_poly)

        # Check containment
        outer_radius = SolutionEvaluator.calculate_outer_radius(positions, angles)
        # Outer hexagon with center at origin and calculated radius
        outer_hexagon = HexagonGeometry.polygon(0, 0, 0, outer_radius)

        # Check containment for all inner hexagons
        for hex_poly in inner_hexagons:
            if not ConstraintChecker.contains(hex_poly, outer_hexagon):
                return 1e10  # Penalty for non-containment

        # Check for overlaps using spatial indexing if requested
        if use_spatial_index:
            grid = ConstraintChecker.build_spatial_grid(inner_hexagons)
            # Check overlaps using spatial indexing
            for i in range(11):
                for j in range(i+1, 11):
                    if ConstraintChecker.fast_collision_check(inner_hexagons, grid, i, j):
                        return 1e10  # Penalty for overlap
        else:
            # Check for overlaps without spatial indexing (slower but simpler)
            for i in range(11):
                for j in range(i+1, 11):
                    if ConstraintChecker.overlaps(inner_hexagons[i], inner_hexagons[j]):
                        return 1e10  # Penalty for overlap

        # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
        return -1.0 / outer_radius

class OptimizationStages:
    """Manages different optimization stages"""

    @staticmethod
    def create_initial_configuration():
        """Create a better initial configuration based on hexagonal packing principles"""
        # Start with a known good configuration with better spacing
        initial_positions = [
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

        initial_angles = [0.0] * 11
        return initial_positions, initial_angles

    @staticmethod
    def adaptive_coarse_optimization(initial_solution, max_generations=30):
        """First phase: Coarse differential evolution optimization with adaptive parameters"""
        # Set bounds for optimization
        bounds = []
        # Position bounds - wider range for exploration
        for _ in range(22):
            bounds.append((-15.0, 15.0))  # X and Y coordinates
        # Angle bounds
        for _ in range(11):
            bounds.append((0.0, 360.0))   # Rotation angles

        # Differential evolution with adaptive parameters
        result = differential_evolution(
            lambda sol: SolutionEvaluator.evaluate(sol, use_spatial_index=(max_generations < 20)),  # Use spatial index for fewer generations
            bounds,
            maxiter=max_generations,
            popsize=10,            # Larger population for better exploration
            seed=42,
            disp=False,
            tol=1e-6,
            strategy='best1bin'
        )

        return result

    @staticmethod
    def adaptive_local_refinement(positions, angles, max_iterations=20):
        """Second phase: Local refinement with adaptive perturbations and early stopping"""
        # More robust refinement with adaptive strategies
        best_positions = positions.copy()
        best_angles = angles.copy()
        best_score = SolutionEvaluator.evaluate(np.concatenate([best_positions.flatten(), best_angles]), use_spatial_index=False)

        # Multiple refinement passes with varying step sizes and early stopping
        step_sizes = [0.1, 0.05, 0.02, 0.01]
        improvement_threshold = 1e-8
        patience = 5
        patience_counter = 0

        for step_size in step_sizes:
            for iteration in range(max_iterations):
                improved = False
                # Try perturbing each position and angle
                for i in range(11):
                    # Perturb position
                    for dim in range(2):
                        old_val = best_positions[i][dim]
                        for delta in [-step_size, step_size]:
                            best_positions[i][dim] = old_val + delta
                            new_score = SolutionEvaluator.evaluate(np.concatenate([best_positions.flatten(), best_angles]), use_spatial_index=False)
                            if new_score < best_score:
                                best_score = new_score
                                improved = True
                            else:
                                best_positions[i][dim] = old_val

                    # Perturb angle
                    old_angle = best_angles[i]
                    for delta in [-5.0, 5.0]:
                        best_angles[i] = old_angle + delta
                        new_score = SolutionEvaluator.evaluate(np.concatenate([best_positions.flatten(), best_angles]), use_spatial_index=False)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                        else:
                            best_angles[i] = old_angle

                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        return best_positions, best_angles

    @staticmethod
    def multi_start_optimization():
        """Run optimization from multiple starting points"""
        best_solution = None
        best_score = float('inf')

        # Try several different initial configurations
        for start_seed in range(5):
            # Create different initial configurations
            np.random.seed(start_seed)

            # Start with different base configurations
            if start_seed == 0:
                initial_positions, initial_angles = OptimizationStages.create_initial_configuration()
            elif start_seed == 1:
                # Perturb the initial configuration
                initial_positions, initial_angles = OptimizationStages.create_initial_configuration()
                for i in range(len(initial_positions)):
                    initial_positions[i][0] += np.random.normal(0, 0.2)
                    initial_positions[i][1] += np.random.normal(0, 0.2)
            elif start_seed == 2:
                # Spiral-like pattern
                initial_positions = [[0, 0], [-2.5, 0], [2.5, 0], [0, 2.5], [0, -2.5]]
                initial_positions.extend([
                    [-1.25, 2.17], [1.25, 2.17], [-1.25, -2.17], [1.25, -2.17],
                    [-3.75, 2.17], [3.75, 2.17]
                ])
                initial_angles = [0.0] * 11
            elif start_seed == 3:
                # More distributed pattern
                initial_positions = [
                    [0, 0], [-3, 0], [3, 0], [0, 3], [0, -3],
                    [-1.5, 2.6], [1.5, 2.6], [-1.5, -2.6], [1.5, -2.6],
                    [-2.5, 2.17], [2.5, 2.17]
                ]
                initial_angles = [0.0] * 11
            else:
                # Random configuration
                initial_positions = [[np.random.uniform(-2, 2), np.random.uniform(-2, 2)] for _ in range(11)]
                initial_angles = [np.random.uniform(0, 360) for _ in range(11)]

            # Flatten initial solution
            initial_solution = []
            for pos in initial_positions:
                initial_solution.extend(pos)
            initial_solution.extend(initial_angles)
            initial_solution = np.array(initial_solution)

            # Run coarse optimization
            try:
                result = OptimizationStages.adaptive_coarse_optimization(initial_solution, max_generations=25)

                # Local refinement
                final_positions = result.x[:22].reshape(-1, 2)
                final_angles = result.x[22:]
                refined_positions, refined_angles = OptimizationStages.adaptive_local_refinement(final_positions, final_angles)

                # Evaluate refined solution
                final_score = SolutionEvaluator.evaluate(np.concatenate([refined_positions.flatten(), refined_angles]), use_spatial_index=False)

                if final_score < best_score:
                    best_score = final_score
                    best_solution = (refined_positions, refined_angles)

            except Exception as e:
                continue  # Skip this start if it fails

        return best_solution if best_solution else (initial_positions, initial_angles)

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
        outer_radius = SolutionEvaluator.calculate_outer_radius(refined_positions, refined_angles)
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