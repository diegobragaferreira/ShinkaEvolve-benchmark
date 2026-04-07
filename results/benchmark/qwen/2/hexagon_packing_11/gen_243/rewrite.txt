# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
import multiprocessing as mp
from functools import partial
from scipy.spatial.distance import cdist

# Constants
UNIT_HEX_RADIUS = 1.0  # Side length of unit hexagon
UNIT_HEX_APOGEE = np.sqrt(3)/2  # Distance from center to corner of unit hexagon

class HexagonPacker:
    """Core hexagon packing optimizer with hybrid evolutionary approach."""
    
    def __init__(self):
        self.n_inner = 11
        self.bounds = self._generate_bounds()
        self._initialize_geometric_tools()
        
    def _initialize_geometric_tools(self):
        """Initialize geometric utilities for efficient computations."""
        # Pre-compute hexagon vertices for unit hexagon
        self.unit_hex_vertices = []
        for i in range(6):
            angle = i * np.pi/3
            x = UNIT_HEX_RADIUS * np.cos(angle)
            y = UNIT_HEX_RADIUS * np.sin(angle)
            self.unit_hex_vertices.append((x, y))
            
    def _generate_bounds(self):
        """Generate optimization bounds for 11 hexagons (x,y,angle each) + outer radius."""
        bounds = []
        # Bounds for inner hexagon positions and rotations
        for _ in range(self.n_inner):
            bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])  # x, y, angle
        # Bound for outer radius
        bounds.append((3.0, 8.0))  # Reasonable range for outer radius
        return bounds
    
    def create_unit_hexagon(self, center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon efficiently."""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            points.append((x, y))
        return Polygon(points)
    
    def get_all_vertices(self, hexagon):
        """Extract all vertices from a hexagon polygon."""
        return list(hexagon.exterior.coords)[:-1]  # Exclude closing point
    
    def get_hexagon_axes(self, vertices):
        """Get the normal vectors (axes) for SAT collision detection."""
        axes = []
        for i in range(len(vertices)):
            p1 = np.array(vertices[i])
            p2 = np.array(vertices[(i + 1) % len(vertices)])
            edge = p2 - p1
            # Normal vector (perpendicular to edge)
            normal = np.array([-edge[1], edge[0]])
            # Normalize
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                axes.append(normal / norm)
        return axes
    
    def project_polygon_onto_axis(self, vertices, axis):
        """Project polygon vertices onto an axis and return min/max."""
        projections = [np.dot(np.array(v), axis) for v in vertices]
        return min(projections), max(projections)
    
    def sat_collision_detection(self, hex1_vertices, hex2_vertices):
        """Use Separating Axis Theorem to detect collision between two polygons."""
        # Get all axes from both polygons
        axes1 = self.get_hexagon_axes(hex1_vertices)
        axes2 = self.get_hexagon_axes(hex2_vertices)
        all_axes = axes1 + axes2

        # Check each axis
        for axis in all_axes:
            min1, max1 = self.project_polygon_onto_axis(hex1_vertices, axis)
            min2, max2 = self.project_polygon_onto_axis(hex2_vertices, axis)

            # Check for separation
            if max1 < min2 or max2 < min1:
                return False  # Separating axis found, no collision

        return True  # No separating axis found, collision exists
    
    def check_containment(self, inner_hexagon, outer_hexagon):
        """Check if inner hexagon is fully contained within outer hexagon."""
        # Check if all vertices of inner hexagon are inside outer hexagon
        inner_vertices = self.get_all_vertices(inner_hexagon)
        for vertex in inner_vertices:
            point = Point(vertex)
            # Use a small buffer to be more robust to floating point errors
            buffered_outer = outer_hexagon.buffer(-1e-8)
            if not buffered_outer.contains(point):
                return False
        return True
    
    def check_overlap(self, hex1, hex2):
        """Check if two hexagons overlap using SAT for precision."""
        hex1_vertices = self.get_all_vertices(hex1)
        hex2_vertices = self.get_all_vertices(hex2)
        return self.sat_collision_detection(hex1_vertices, hex2_vertices)
    
    def calculate_tight_outer_radius(self, inner_params):
        """Calculate tightest possible outer hexagon radius using actual vertex positions."""
        # Get all hexagon vertices and find bounding circle
        all_vertices = []

        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.create_unit_hexagon((x, y), angle)
            # Get all vertices of this hexagon
            for point in self.get_all_vertices(hexagon):
                all_vertices.append(point)

        if not all_vertices:
            return 1.0

        # Convert to numpy array for easier computation
        vertices_array = np.array(all_vertices)

        # Find centroid of all vertices
        centroid = np.mean(vertices_array, axis=0)

        # Calculate distances from centroid to all vertices
        distances = np.sqrt(np.sum((vertices_array - centroid)**2, axis=1))

        # Use adaptive margin based on the average distance and number of hexagons
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)

        # Adaptive margin: smaller for tightly packed arrangements, larger for scattered ones
        # Scale margin based on how much the distribution deviates from uniform
        deviation_ratio = max_distance / (avg_distance + 1e-10)  # Avoid division by zero
        adaptive_margin = 1e-6 + (0.01 * deviation_ratio)  # Dynamic margin

        # Outer radius is the maximum distance plus adaptive margin for numerical stability
        outer_radius = max_distance + adaptive_margin

        return outer_radius
    
    def evaluate_constraints(self, inner_params, outer_radius):
        """Comprehensive constraint evaluation with early termination."""
        # Create inner hexagons
        inner_hexagons = []
        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)

        # Create outer hexagon
        outer_hexagon = self.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)

        # Check containment (early termination)
        for hexagon in inner_hexagons:
            if not self.check_containment(hexagon, outer_hexagon_scaled):
                return False, False, 0.0  # containment violated

        # Check overlaps (early termination)
        for i in range(self.n_inner):
            for j in range(i+1, self.n_inner):
                if self.check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    return False, False, 0.0  # overlap violated

        # Calculate tight radius for better objective function
        return True, True, 1.0 / outer_radius  # valid solution
    
    def objective_function(self, params):
        """Objective function to minimize: negative of 1/outer_radius."""
        outer_radius = params[-1]
        inner_params = params[:-1]

        # Check constraints
        containment_ok, overlap_ok, inv_radius = self.evaluate_constraints(
            inner_params, outer_radius
        )

        # If any constraint violated, return large penalty
        if not (containment_ok and overlap_ok):
            return 10000.0 + abs(outer_radius)  # penalty for constraint violations

        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius
    
    def generate_initial_configs(self):
        """Generate diverse initial configurations."""
        configs = []
        
        # Strategy 1: Hexagonal cluster pattern
        cluster_centers = [
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
        
        # Generate 20 diverse configurations
        for _ in range(20):
            config = []
            for i, (cx, cy) in enumerate(cluster_centers):
                # Add small random variation with controlled magnitude
                jitter_x = np.random.uniform(-0.2, 0.2)
                jitter_y = np.random.uniform(-0.2, 0.2)
                config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])
            config.append(4.0 + np.random.uniform(0.2, 0.8))  # outer radius estimate
            configs.append(config)
            
        # Strategy 2: Spiral pattern
        for _ in range(10):
            config = []
            # Center
            config.extend([0.0, 0.0, np.random.uniform(0, 360)])
            # Spiral pattern
            for i in range(10):
                angle = i * 0.6
                radius = 1.2 + i * 0.3
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                config.extend([x, y, np.random.uniform(0, 360)])
            config.append(4.5 + np.random.uniform(0.1, 0.5))  # outer radius estimate
            configs.append(config)
            
        return configs
        
    def hybrid_local_search(self, initial_params):
        """Enhanced local refinement using gradient-based optimization."""
        # First, tighten bounds for local search
        local_bounds = []
        for _ in range(self.n_inner):
            local_bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])
        local_bounds.append((3.0, 8.0))

        try:
            # Use L-BFGS-B with high precision
            result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=local_bounds,
                options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-10},
                callback=lambda x: None
            )

            if result.success:
                return result.x
        except Exception as e:
            warnings.warn(f"Local search failed: {str(e)}")

        return initial_params

    def optimize_with_parallel_de(self):
        """Run multiple differential evolution instances in parallel."""
        def run_de_instance(seed_val):
            try:
                result = differential_evolution(
                    self.objective_function,
                    self.bounds,
                    seed=seed_val,
                    maxiter=80,
                    popsize=20,
                    tol=1e-6,
                    mutation=(0.8, 1.0),
                    recombination=0.9,
                    disp=False
                )
                if result.success:
                    return result.x
            except Exception as e:
                warnings.warn(f"DE instance with seed {seed_val} failed: {str(e)}")
            return None

        # Run multiple instances in parallel
        seeds = [42, 123, 456, 789]  # Different random seeds
        with mp.Pool(processes=min(len(seeds), mp.cpu_count())) as pool:
            results = pool.map(run_de_instance, seeds)

        # Filter out None results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            return None

        # Find the best result based on objective function value
        best_result = min(valid_results, key=lambda x: self.objective_function(x))
        return best_result

    def optimize_solution(self):
        """Main optimization routine using hybrid approach."""
        # Generate initial configurations
        initial_configs = self.generate_initial_configs()
        
        # Try parallel differential evolution first
        best_params = self.optimize_with_parallel_de()
        
        # If parallel approach failed or gave poor results, try sequential approach
        if best_params is None:
            try:
                # Use differential evolution for global optimization
                result = differential_evolution(
                    self.objective_function,
                    self.bounds,
                    seed=42,
                    maxiter=120,
                    popsize=25,
                    tol=1e-7,
                    mutation=(0.9, 1.0),
                    recombination=0.95,
                    disp=False
                )

                if result.success:
                    best_params = result.x
            except Exception as e:
                warnings.warn(f"DE optimization failed: {str(e)}")
                pass

        # Refine with local search if we have a good candidate
        if best_params is not None:
            refined_params = self.hybrid_local_search(best_params)
            return refined_params

        # Fallback to generating initial random guess
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
    Uses evolutionary optimization to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        packer = HexagonPacker()
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