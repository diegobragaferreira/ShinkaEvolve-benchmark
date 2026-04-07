# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
from joblib import Parallel, delayed

# Constants
UNIT_HEX_RADIUS = 1.0  # Side length of unit hexagon
UNIT_HEX_APOGEE = np.sqrt(3)/2  # Distance from center to corner of unit hexagon

class HexagonPacker:
    def __init__(self):
        self.n_inner = 11
        self.unit_hex_radius = UNIT_HEX_RADIUS
        self.unit_hex_apogee = UNIT_HEX_APOGEE

    def create_unit_hexagon(self, center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon"""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + self.unit_hex_radius * np.cos(angle)
            y = center[1] + self.unit_hex_radius * np.sin(angle)
            points.append((x, y))
        return Polygon(points)

    def validate_polygon(self, polygon):
        """Ensure polygon is valid for geometric operations"""
        if not polygon.is_valid:
            return make_valid(polygon)
        return polygon

    def get_hexagon_axes(self, hexagon):
        """Get the axes (normals) of a hexagon for SAT collision detection"""
        coords = list(hexagon.exterior.coords)
        axes = []
        for i in range(len(coords) - 1):
            p1 = np.array(coords[i])
            p2 = np.array(coords[i+1])
            edge = p2 - p1
            # Normal vector (perpendicular to edge)
            normal = np.array([-edge[1], edge[0]])
            # Normalize
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
                axes.append(normal)
        return axes

    def project_polygon_onto_axis(self, polygon, axis):
        """Project polygon onto an axis and return min/max projection"""
        coords = list(polygon.exterior.coords)
        projections = [np.dot(np.array(coord), axis) for coord in coords]
        return min(projections), max(projections)

    def sat_collision_check(self, hex1, hex2):
        """Use Separating Axis Theorem to detect collision between hexagons"""
        # Get axes for both polygons
        axes1 = self.get_hexagon_axes(hex1)
        axes2 = self.get_hexagon_axes(hex2)
        all_axes = axes1 + axes2

        # Check each axis
        for axis in all_axes:
            min1, max1 = self.project_polygon_onto_axis(hex1, axis)
            min2, max2 = self.project_polygon_onto_axis(hex2, axis)

            # Check for separation
            if max1 < min2 or max2 < min1:
                return False  # No overlap on this axis, so they don't collide

        return True  # Overlap on all axes, so they collide

    def check_containment(self, inner_hexagon, outer_hexagon):
        """Check if inner hexagon is fully contained within outer hexagon"""
        # Check if all vertices of inner hexagon are inside outer hexagon
        for point in inner_hexagon.exterior.coords[:-1]:
            if not outer_hexagon.contains(Point(point)):
                return False
        return True

    def check_overlap(self, hex1, hex2):
        """Check if two hexagons overlap using robust SAT-based detection with buffer validation"""
        # Add buffer to polygons to handle floating-point precision issues
        buffered_hex1 = hex1.buffer(1e-6)
        buffered_hex2 = hex2.buffer(1e-6)

        # Early rejection using bounding boxes
        bbox1 = buffered_hex1.bounds
        bbox2 = buffered_hex2.bounds
        if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
            bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
            return False

        # Use SAT for precise collision detection
        return self.sat_collision_check(buffered_hex1, buffered_hex2)

    def evaluate_constraints(self, inner_params, outer_radius):
        """Comprehensive constraint evaluation with optimized early termination and enhanced validation"""
        # Create all inner hexagons with buffering for robustness
        inner_hexagons = []
        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)

        # Create outer hexagon (unit size then scaled)
        outer_hexagon = self.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)

        # Check containment with buffer for robustness
        for hexagon in inner_hexagons:
            buffered_hexagon = hexagon.buffer(1e-6)
            if not self.check_containment(buffered_hexagon, outer_hexagon_scaled):
                return False, False, 0.0  # containment violated

        # Check overlaps in parallel for better performance
        def check_pair_overlap(i, j):
            return self.check_overlap(inner_hexagons[i], inner_hexagons[j])

        # Parallel overlap checking
        overlap_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(check_pair_overlap)(i, j)
            for i in range(self.n_inner) for j in range(i+1, self.n_inner)
        )

        # Check if any overlaps were found (early termination)
        if any(overlap_results):
            return False, False, 0.0  # overlap violated

        return True, True, 1.0 / outer_radius  # valid solution

    def objective_function(self, params):
        """Objective function to minimize: negative of 1/outer_radius (i.e., maximize 1/outer_radius)"""
        # params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_radius]
        n = self.n_inner
        outer_radius = params[-1]

        # Extract inner hexagon parameters
        inner_params = params[:-1]

        # Check constraints
        containment_ok, overlap_ok, inv_radius = self.evaluate_constraints(inner_params, outer_radius)

        # If any constraint violated, return large penalty
        if not (containment_ok and overlap_ok):
            # Much larger penalty for constraint violations to discourage invalid solutions
            return 100000.0 + abs(outer_radius) * 100.0

        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius

    def generate_initial_population(self, pop_size=50):
        """Generate diverse initial configurations using geometric heuristics"""
        initial_configs = []

        # Generate multiple starting configurations
        for _ in range(pop_size):
            config = []

            # More strategic initial placement based on optimal packing patterns
            # Central hexagon plus surrounding in a more optimized pattern
            # Using a configuration inspired by known good solutions
            base_positions = [
                (0.0, 0.0),       # center
                (-1.8, 0.0),      # left
                (1.8, 0.0),       # right
                (0.0, 1.8),       # top
                (0.0, -1.8),      # bottom
                (-1.3, 1.3),      # top-left
                (1.3, 1.3),       # top-right
                (-1.3, -1.3),     # bottom-left
                (1.3, -1.3),      # bottom-right
                (-2.2, 0.0),      # further left
                (2.2, 0.0),       # further right
            ]

            # Add positions with more strategic spacing
            for i, (cx, cy) in enumerate(base_positions):
                # Add small random variation to avoid symmetry issues
                jitter_x = np.random.normal(0, 0.15)
                jitter_y = np.random.normal(0, 0.15)
                config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])

            # Add outer radius estimate - start with something reasonable
            config.append(4.0 + np.random.uniform(0, 1.0))
            initial_configs.append(config)

        return initial_configs

    def optimize_with_local_search(self, initial_params):
        """Refine solution using local optimization after global search"""
        bounds = []
        # Bounds for inner hexagon positions and rotations - tightened for better convergence
        for _ in range(self.n_inner):
            bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (0, 360)])  # x, y, angle
        # Bound for outer radius - tightened for better convergence
        bounds.append((3.0, 7.5))  # Reasonable range for outer radius (tightened)

        options = {'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}

        try:
            # Use L-BFGS-B for fine-tuning
            result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options=options,
                callback=lambda x: None  # Empty callback
            )

            if result.success:
                return result.x

        except Exception as e:
            warnings.warn(f"Local search failed: {str(e)}")

        return initial_params

    def optimize_solution(self):
        """Main optimization routine using differential evolution followed by local search"""
        # Generate tighter bounds for optimization to improve convergence
        bounds = []

        # Bounds for inner hexagon positions and rotations - tighter ranges
        for _ in range(self.n_inner):
            bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])  # x, y, angle

        # Bound for outer radius - more constrained
        bounds.append((3.0, 8.0))  # Reasonable range for outer radius

        # Initial guess from more informed heuristic placement
        initial_guess = []

        # Better positioned centers that reflect a more optimal packing
        # These positions are chosen to be closer to what a good solution might look like
        centers = [
            (0.0, 0.0),       # center
            (-1.8, 0.0),      # left
            (1.8, 0.0),       # right
            (0.0, 1.8),       # top
            (0.0, -1.8),      # bottom
            (-1.3, 1.3),      # top-left
            (1.3, 1.3),       # top-right
            (-1.3, -1.3),     # bottom-left
            (1.3, -1.3),      # bottom-right
            (-2.2, 0.0),      # further left
            (2.2, 0.0),       # further right
        ]

        for i, (cx, cy) in enumerate(centers):
            initial_guess.extend([cx, cy, np.random.uniform(0, 360)])

        # Estimate outer radius based on the initial configuration
        # Calculate the maximum distance from center to any hexagon center + apogee
        max_dist = 0
        for cx, cy in centers:
            dist = np.sqrt(cx**2 + cy**2) + self.unit_hex_apogee
            max_dist = max(max_dist, dist)

        initial_guess.append(max_dist + 0.3)  # Add a small margin

        # Optimization settings - more aggressive for better convergence
        try:
            # Use differential evolution for global optimization
            result = differential_evolution(
                self.objective_function,
                bounds,
                seed=42,
                maxiter=200,
                popsize=25,
                tol=1e-8,
                mutation=(0.6, 1.0),
                recombination=0.8,
                disp=False
            )

            if result.success:
                # Refine with local search
                refined_params = self.optimize_with_local_search(result.x)
                return refined_params

        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")

        # Return initial guess if optimization fails
        return initial_guess

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    packer = HexagonPacker()

    try:
        # Run optimization
        final_params = packer.optimize_solution()

        # Extract results
        n = 11
        inner_params = final_params[:-1]
        outer_radius = final_params[-1]

        # Validate solution
        containment_ok, overlap_ok, inv_radius = packer.evaluate_constraints(inner_params, outer_radius)

        if containment_ok and overlap_ok:
            # Format output
            inner_hex_data = np.zeros((n, 3))
            for i in range(n):
                inner_hex_data[i] = inner_params[3*i:3*i+3]

            outer_hex_data = np.array([0, 0, 0])

            return inner_hex_data, outer_hex_data, outer_radius

    except Exception as e:
        warnings.warn(f"Error in optimization: {str(e)}")
        pass

    # Fallback to original method if optimization fails
    inner_hex_data = np.array([
        [0, 0, 0],        # center
        [-2.5, 0, 0],     # left
        [2.5, 0, 0],      # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0], # bottom-right
        [-3.75, 2.17, 0], # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0], # far bottom-right
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END