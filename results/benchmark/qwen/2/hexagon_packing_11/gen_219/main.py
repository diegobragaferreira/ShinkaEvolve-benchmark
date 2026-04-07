# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
from joblib import Parallel, delayed
import copy
from functools import lru_cache

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2

class HexagonGeometry:
    """Handles geometric operations for hexagons with caching optimizations"""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def create_unit_hexagon_cached(center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon with caching"""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            points.append((x, y))
        return Polygon(points)

    @staticmethod
    def create_unit_hexagon(center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon without caching"""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            points.append((x, y))
        return Polygon(points)

    @staticmethod
    def hexagon_vertices(center=(0,0), rotation=0):
        """Get vertices of a unit hexagon for SAT calculations"""
        angle_offset = np.deg2rad(rotation)
        vertices = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            vertices.append((x, y))
        return vertices

    @staticmethod
    def get_hexagon_edges(vertices):
        """Get edges from vertices for SAT calculation"""
        edges = []
        for i in range(len(vertices)):
            p1 = np.array(vertices[i])
            p2 = np.array(vertices[(i+1) % len(vertices)])
            edge = p2 - p1
            edges.append(edge)
        return edges

class SATConstraintChecker:
    """Implements Separating Axis Theorem for accurate overlap detection"""
    
    @staticmethod
    def check_overlap_sat(hex1_vertices, hex2_vertices):
        """Check overlap using Separating Axis Theorem for better precision"""
        # Get edges of both hexagons
        edges1 = HexagonGeometry.get_hexagon_edges(hex1_vertices)
        edges2 = HexagonGeometry.get_hexagon_edges(hex2_vertices)

        # Combine all axes
        all_axes = edges1 + edges2

        # Normalize axes
        for i, axis in enumerate(all_axes):
            norm = np.linalg.norm(axis)
            if norm > 1e-10:
                all_axes[i] = axis / norm

        # Project both polygons onto each axis
        for axis in all_axes:
            # Project hex1
            projections1 = [np.dot(vertex, axis) for vertex in hex1_vertices]
            min1, max1 = min(projections1), max(projections1)

            # Project hex2
            projections2 = [np.dot(vertex, axis) for vertex in hex2_vertices]
            min2, max2 = min(projections2), max(projections2)

            # Check for separation
            if max1 < min2 or max2 < min1:
                return False  # Separating axis found, no overlap

        return True  # No separating axis found, overlap exists

    @staticmethod
    def check_containment_precise(inner_vertices, outer_vertices):
        """Precise containment check using vertex-by-vertex test with buffer"""
        outer_polygon = Polygon(outer_vertices)
        # Use a small buffer to avoid floating point precision issues
        buffered_outer = outer_polygon.buffer(1e-8)
        for vertex in inner_vertices:
            point = Point(vertex)
            if not buffered_outer.contains(point):
                return False
        return True

class ImprovedHexagonPacker:
    """Main class for hexagon packing optimization with advanced techniques"""
    
    def __init__(self):
        self.n_inner = 11
        self.unit_hex_radius = UNIT_HEX_RADIUS
        self.unit_hex_apogee = UNIT_HEX_APOGEE
        self.geometry_utils = HexagonGeometry()
        self.sat_checker = SATConstraintChecker()
        
    def calculate_better_tight_outer_radius(self, inner_params):
        """Calculate a tighter outer radius using vertex analysis"""
        # Get all hexagon vertices and compute actual bounding box
        all_vertices = []

        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.geometry_utils.create_unit_hexagon((x, y), angle)
            # Get all vertices of this hexagon
            for point in hexagon.exterior.coords[:-1]:  # exclude closing point
                all_vertices.append(point)

        if not all_vertices:
            return 1.0

        # Convert to numpy array for easier computation
        vertices_array = np.array(all_vertices)

        # Calculate bounding box directly
        min_x, max_x = np.min(vertices_array[:, 0]), np.max(vertices_array[:, 0])
        min_y, max_y = np.min(vertices_array[:, 1]), np.max(vertices_array[:, 1])

        # Calculate center of bounding box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Calculate distance from center to corners of bounding box
        distances = []
        for vertex in vertices_array:
            dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
            distances.append(dist)

        # Use the maximum distance from center to any vertex
        # Add a small margin for numerical stability
        outer_radius = np.max(distances) * 1.01  # 1% margin for safety

        return outer_radius

    def evaluate_constraints(self, inner_params, outer_radius):
        """Comprehensive constraint evaluation with early termination and SAT overlap checking"""
        inner_vertices_list = []
        inner_hexagons = []
        
        # Create inner hexagons and collect vertices for SAT checks
        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.geometry_utils.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)
            
            # Also collect vertices for SAT checks
            vertices = self.geometry_utils.hexagon_vertices((x, y), angle)
            inner_vertices_list.append(vertices)

        # Create outer hexagon
        outer_hexagon = self.geometry_utils.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)

        # Check containment (early termination) - using precise checking
        for i, hexagon in enumerate(inner_hexagons):
            # Use buffered containment check
            buffered_inner = hexagon.buffer(-1e-10)
            if not outer_hexagon_scaled.contains(buffered_inner):
                return False, False, 0.0  # containment violated

        # Check overlaps (early termination) - using SAT for better precision
        for i in range(self.n_inner):
            for j in range(i+1, self.n_inner):
                if self.sat_checker.check_overlap_sat(inner_vertices_list[i], inner_vertices_list[j]):
                    return False, False, 0.0  # overlap violated

        # Calculate actual tight radius for better objective function
        actual_tight_radius = self.calculate_better_tight_outer_radius(inner_params)
        return True, True, 1.0 / actual_tight_radius  # valid solution
    
    def objective_function(self, params):
        """Objective function to minimize: negative of 1/outer_radius (i.e., maximize 1/outer_radius)"""
        # params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_radius]
        outer_radius = params[-1]

        # Extract inner hexagon parameters
        inner_params = params[:-1]

        # Check constraints
        containment_ok, overlap_ok, inv_radius = self.evaluate_constraints(inner_params, outer_radius)

        # If any constraint violated, return a sophisticated penalty
        if not (containment_ok and overlap_ok):
            # Use a more sophisticated penalty based on the severity
            penalty = 10000.0 + abs(outer_radius) * 10 + 10000.0 * (not containment_ok) + 10000.0 * (not overlap_ok)
            return penalty

        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius

    def generate_diverse_initial_configs(self):
        """Generate multiple diverse initial configurations with different strategies"""
        configs = []
        
        # Strategy 1: Hexagonal ring arrangement
        ring_pattern = [
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
        
        # Strategy 2: Modified hexagonal ring with different spacing
        modified_ring = [
            (0.0, 0.0),       # center
            (-1.7, 0.0),      # left
            (1.7, 0.0),       # right
            (0.0, 1.7),       # top
            (0.0, -1.7),      # bottom
            (-1.2, 1.2),      # top-left
            (1.2, 1.2),       # top-right
            (-1.2, -1.2),     # bottom-left
            (1.2, -1.2),      # bottom-right
            (-2.0, 0.0),      # further left
            (2.0, 0.0),       # further right
        ]
        
        # Strategy 3: Optimized compact arrangement
        compact_pattern = [
            (0.0, 0.0),       # center
            (-1.9, 0.0),      # left
            (1.9, 0.0),       # right
            (0.0, 1.9),       # top
            (0.0, -1.9),      # bottom
            (-1.4, 1.4),      # top-left
            (1.4, 1.4),       # top-right
            (-1.4, -1.4),     # bottom-left
            (1.4, -1.4),      # bottom-right
            (-2.3, 0.0),      # further left
            (2.3, 0.0),       # further right
        ]
        
        # Strategy 4: Spiral-like arrangement
        spiral_pattern = [
            (0.0, 0.0),       # center
            (-1.85, 0.0),     # left
            (1.85, 0.0),      # right
            (0.0, 1.85),      # top
            (0.0, -1.85),     # bottom
            (-1.35, 1.35),    # top-left
            (1.35, 1.35),     # top-right
            (-1.35, -1.35),   # bottom-left
            (1.35, -1.35),    # bottom-right
            (-2.25, 0.0),     # further left
            (2.25, 0.0),      # further right
        ]
        
        patterns = [ring_pattern, modified_ring, compact_pattern, spiral_pattern]
        
        # Generate diverse configurations from each strategy
        for i, pattern in enumerate(patterns):
            for j in range(15):  # 15 configs per pattern
                config = []
                for k, (cx, cy) in enumerate(pattern):
                    # Add small random variation
                    jitter_x = np.random.uniform(-0.15, 0.15)
                    jitter_y = np.random.uniform(-0.15, 0.15)
                    config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])
                # Use a more accurate initial radius estimate
                config.append(3.8 + np.random.uniform(0.1, 0.5))  # Tighter initial estimate
                configs.append(config)
        
        return configs

    def optimize_with_multi_start(self):
        """Optimize using multi-start with enhanced strategies"""
        # Generate bounds for optimization
        bounds = []
        # Bounds for inner hexagon positions and rotations
        for _ in range(self.n_inner):
            bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])  # x, y, angle
        # Bound for outer radius
        bounds.append((3.0, 8.0))  # Reasonable range for outer radius

        # Generate multiple initial configurations
        initial_configs = self.generate_diverse_initial_configs()

        # Create a wrapper function for parallel execution
        def evaluate_config(config):
            try:
                # Use DE with moderate iterations for faster screening
                # Use a lower popsize to speed things up, but maintain quality
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    seed=np.random.randint(0, 10000),
                    maxiter=30,  # Reduced iterations for screening
                    popsize=12,  # Smaller population for speed
                    tol=1e-5,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    disp=False,
                    polish=False  # Disable polishing for speed
                )

                if result.success:
                    return result.x, result.fun
            except Exception as e:
                # Log error but continue
                return None, float('inf')
            return None, float('inf')

        # Evaluate all configurations in parallel with limited jobs
        results = Parallel(n_jobs=min(4, len(initial_configs)))(
            delayed(evaluate_config)(config) for config in initial_configs[:20]  # Test top 20 configs
        )

        # Filter out None results
        valid_results = [(params, fun) for params, fun in results if params is not None]
        if not valid_results:
            return None

        # Find the best result based on objective function value
        best_params, _ = min(valid_results, key=lambda x: x[1])
        
        # Apply a local refinement step with L-BFGS-B
        refined_params = self.local_refinement(best_params, bounds)
        
        return refined_params

    def local_refinement(self, initial_params, bounds):
        """Apply local refinement with careful bounds handling"""
        options = {'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}

        try:
            # Use L-BFGS-B for fine-tuning with strictly enforced bounds
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
            else:
                return initial_params
                
        except Exception as e:
            warnings.warn(f"Local refinement failed: {str(e)}")
            return initial_params

    def optimize_solution(self):
        """Main optimization routine"""
        # Try multi-start approach first
        best_params = self.optimize_with_multi_start()

        # If found, return the result
        if best_params is not None:
            return best_params

        # Fallback: single DE run
        try:
            bounds = []
            for _ in range(self.n_inner):
                bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])
            bounds.append((3.0, 8.0))

            result = differential_evolution(
                self.objective_function,
                bounds,
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False,
                polish=False
            )

            if result.success:
                # Apply local refinement
                refined_params = self.local_refinement(result.x, bounds)
                return refined_params

        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")

        # Return default initial guess if everything fails
        initial_guess = []
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

        initial_guess.append(4.0)  # Initial outer radius estimate
        return initial_guess

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid optimization with multi-start and local refinement to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    packer = ImprovedHexagonPacker()

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
