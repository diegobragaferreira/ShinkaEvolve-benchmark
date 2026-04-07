# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import multiprocessing as mp
import time
from joblib import Parallel, delayed
import warnings
import math
from typing import Tuple, List, Optional, Any

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2
BENCHMARK_RATIO = 0.2544

class HexagonGeometry:
    """Handles all geometric operations for hexagons with optimized performance"""
    
    def __init__(self):
        self.unit_hex_radius = UNIT_HEX_RADIUS
        self.unit_hex_apogee = UNIT_HEX_APOGEE
        self.unit_hex_vertices = self._generate_unit_hexagon_vertices()
        self.hexagon_edges = self._generate_hexagon_edges()
        
    def _generate_unit_hexagon_vertices(self) -> np.ndarray:
        """Generate vertices of a unit regular hexagon centered at origin"""
        vertices = []
        for i in range(6):
            angle = i * np.pi / 3
            x = self.unit_hex_radius * np.cos(angle)
            y = self.unit_hex_radius * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)

    def _generate_hexagon_edges(self) -> np.ndarray:
        """Generate edges of a unit regular hexagon"""
        edges = []
        vertices = self.unit_hex_vertices
        for i in range(6):
            edge = vertices[(i+1) % 6] - vertices[i]
            edges.append(edge)
        return np.array(edges)

    def create_hexagon_vertices(self, center: Tuple[float, float], rotation: float) -> np.ndarray:
        """Create hexagon vertices array for fast geometric operations"""
        angle_offset = np.deg2rad(rotation)
        vertices = np.zeros((6, 2))
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            vertices[i] = [
                center[0] + self.unit_hex_radius * np.cos(angle),
                center[1] + self.unit_hex_radius * np.sin(angle)
            ]
        return vertices

    def create_hexagon_vertices_vectorized(self, centers: np.ndarray, rotations: np.ndarray) -> np.ndarray:
        """Vectorized creation of hexagon vertices for multiple hexagons"""
        n = len(centers)
        vertices = np.zeros((n, 6, 2))
        
        # Precompute angles
        angles = np.arange(6) * np.pi/3
        angle_offsets = np.deg2rad(rotations)
        
        # Vectorized computation
        for i in range(6):
            angle = angle_offsets + angles[i]
            vertices[:, i, 0] = centers[:, 0] + self.unit_hex_radius * np.cos(angle)
            vertices[:, i, 1] = centers[:, 1] + self.unit_hex_radius * np.sin(angle)
            
        return vertices

class HexagonValidation:
    """Handles constraint validation for hexagon configurations with optimized performance"""
    
    def __init__(self, geometry: HexagonGeometry):
        self.geometry = geometry
        self.outer_hexagon = self.geometry.create_hexagon_vertices((0, 0), 0)
        
    def _project_polygon_onto_axis(self, vertices: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
        """Project polygon vertices onto an axis and return min/max projections"""
        projections = np.dot(vertices, axis)
        return np.min(projections), np.max(projections)

    def _get_hexagon_edges(self, vertices: np.ndarray) -> np.ndarray:
        """Get edges from vertices"""
        edges = np.zeros((len(vertices), 2))
        n = len(vertices)
        for i in range(n):
            edges[i] = vertices[i] - vertices[(i+1)%n]
        return edges

    def sat_collision_check(self, vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
        """SAT-based collision detection between two hexagons - highly optimized"""
        # Get edges for both polygons
        edges1 = self._get_hexagon_edges(vertices1)
        edges2 = self._get_hexagon_edges(vertices2)

        # Precompute normals (perpendicular vectors)
        normals1 = np.zeros_like(edges1)
        normals2 = np.zeros_like(edges2)
        
        # Compute normals (perpendicular to edges) efficiently
        normals1[:, 0] = -edges1[:, 1]
        normals1[:, 1] = edges1[:, 0]
        normals2[:, 0] = -edges2[:, 1]
        normals2[:, 1] = edges2[:, 0]
        
        # Normalize normals
        norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
        norms1 = np.where(norms1 > 1e-10, norms1, 1.0)
        norms2 = np.where(norms2 > 1e-10, norms2, 1.0)
        normals1 /= norms1
        normals2 /= norms2
        
        # Test all axes (combined normals from both polygons)
        all_normals = np.vstack([normals1, normals2])
        
        # Vectorized projection test
        for axis in all_normals:
            min1, max1 = self._project_polygon_onto_axis(vertices1, axis)
            min2, max2 = self._project_polygon_onto_axis(vertices2, axis)
            
            # Check for overlap
            if max1 < min2 or max2 < min1:
                return False  # No overlap along this axis
                
        return True  # Overlap detected

    def check_overlap_fast(self, vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
        """Fast overlap check using SAT with early termination"""
        return self.sat_collision_check(vertices1, vertices2)
    
    def check_containment_simple(self, vertices: np.ndarray, outer_radius: float) -> bool:
        """Simple containment check using distance from center"""
        # Get center of hexagon (average of all vertices)
        center = np.mean(vertices, axis=0)
        dist_from_center = np.linalg.norm(center)
        # Add radius margin to account for hexagon size
        return dist_from_center + self.geometry.unit_hex_apogee <= outer_radius

    def validate_configuration(self, inner_hex_data: np.ndarray, outer_radius: float) -> Tuple[bool, float]:
        """
        Validate configuration for collisions and containment with early termination
        Returns (is_valid, objective_value)
        """
        num_hex = len(inner_hex_data)
        
        # Precompute all hexagon vertices for faster access
        centers = inner_hex_data[:, :2]
        rotations = inner_hex_data[:, 2]
        
        # Create vertices for all hexagons at once (vectorized)
        hex_vertices = self.geometry.create_hexagon_vertices_vectorized(centers, rotations)
        
        # Check containment for all hexagons (early termination)
        for i in range(num_hex):
            vertices = hex_vertices[i]
            if not self.check_containment_simple(vertices, outer_radius):
                return False, 0.0  # containment violated
                
        # Check overlaps between all pairs (early termination)
        for i in range(num_hex):
            for j in range(i+1, num_hex):
                if self.check_overlap_fast(hex_vertices[i], hex_vertices[j]):
                    return False, 0.0  # overlap violated
                    
        # Valid configuration
        return True, 1.0 / outer_radius

class HexagonPackingOptimizer:
    """Main optimization controller with modular design and performance optimizations"""
    
    def __init__(self):
        self.geometry = HexagonGeometry()
        self.validator = HexagonValidation(self.geometry)
        self.num_inner = 11
        self.max_evaluations = 120  # Reduced for better time management
        self.num_parallel = min(mp.cpu_count(), 6)  # Allow more parallel workers
        
    def _calculate_outer_radius_estimate(self, inner_hex_data: np.ndarray) -> float:
        """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
        max_dist = 0
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            dist = np.linalg.norm(np.array(center) - np.array([0, 0]))
            # Add distance from center to corner of unit hexagon
            dist += self.geometry.unit_hex_apogee
            max_dist = max(max_dist, dist)
        return max_dist * 1.15  # Slightly smaller margin for better packing
        
    def _generate_initial_configurations(self) -> List[np.ndarray]:
        """Generate diverse initial configurations with smart patterns"""
        configs = []
        
        # Base hexagonal arrangement pattern 1 (dense honeycomb)
        base_positions = [
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
        ]
        
        # Add base configuration
        configs.append(np.array(base_positions))
        
        # Pattern 2: Spiral arrangement
        spiral_positions = [
            [0, 0, 0],
            [-2.2, 0, 0],
            [2.2, 0, 0],
            [0, 2.2, 0],
            [0, -2.2, 0],
            [-1.6, 1.6, 0],
            [1.6, 1.6, 0],
            [-1.6, -1.6, 0],
            [1.6, -1.6, 0],
            [-3.2, 1.1, 0],
            [3.2, 1.1, 0],
        ]
        configs.append(np.array(spiral_positions))
        
        # Pattern 3: Linear arrangement
        linear_positions = [
            [0, 0, 0],
            [-3.0, 0, 0],
            [3.0, 0, 0],
            [-1.5, 2.6, 0],
            [1.5, 2.6, 0],
            [-1.5, -2.6, 0],
            [1.5, -2.6, 0],
            [-4.5, 2.6, 0],
            [4.5, 2.6, 0],
            [-4.5, -2.6, 0],
            [4.5, -2.6, 0],
        ]
        configs.append(np.array(linear_positions))
        
        # Pattern 4: Random variations with different spacing
        for _ in range(3):
            config = np.array(base_positions)
            for i in range(len(config)):
                # Add random perturbations
                config[i][0] += np.random.normal(0, 0.25)
                config[i][1] += np.random.normal(0, 0.25)
                config[i][2] += np.random.normal(0, 3)
                config[i][2] = config[i][2] % 360
            configs.append(config)
            
        return configs

    def _objective_function(self, params: np.ndarray) -> float:
        """
        Objective function to minimize (we want to maximize 1/outer_radius)
        Returns negative because we're using minimization
        """
        try:
            # Extract inner hexagon data (first 33 parameters: 11 hexagons * 3 params each)
            inner_params = params[:-1].reshape(self.num_inner, 3)
            outer_radius = params[-1]
            
            # Validate configuration
            is_valid, objective_value = self.validator.validate_configuration(inner_params, outer_radius)
            
            if is_valid:
                return -objective_value  # Negative for minimization
            else:
                # Return large penalty if invalid
                return 10000.0  # Large positive number to indicate poor fitness
                
        except Exception:
            return 10000.0  # Penalty for exceptions

    def _get_bounds(self, initial_config: np.ndarray) -> List[Tuple[float, float]]:
        """Get optimization bounds for parameters"""
        bounds = []
        
        # Add bounds for positions and rotations (11 hexagons * 3 params each)
        for _ in range(self.num_inner):
            # x coordinate bounds
            bounds.append((-12.0, 12.0))
            # y coordinate bounds  
            bounds.append((-12.0, 12.0))
            # rotation bounds (0 to 360 degrees)
            bounds.append((0.0, 360.0))
        
        # Add bound for outer hexagon radius
        bounds.append((1.5, 12.0))  # Tighter range for better convergence
        
        return bounds

    def _optimize_single(self, initial_config: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float], float]:
        """Run single differential evolution optimization with better settings"""
        try:
            # Prepare optimization parameters
            bounds = self._get_bounds(initial_config)
            
            # Initial guess
            initial_params = initial_config.flatten()
            estimated_radius = self._calculate_outer_radius_estimate(initial_config)
            initial_params = np.append(initial_params, estimated_radius)
            
            # Run optimization with tuned parameters
            result = differential_evolution(
                self._objective_function,
                bounds,
                args=(),
                seed=42,
                maxiter=self.max_evaluations,
                popsize=25,  # Larger population for better exploration
                mutation=(0.8, 1),  # Higher mutation rate for better diversity
                recombination=0.9,  # Higher recombination rate for better mixing
                atol=1e-8,
                rtol=1e-8,
                disp=False
            )
            
            # Extract results
            opt_params = result.x
            opt_inner_data = opt_params[:-1].reshape(self.num_inner, 3)
            opt_outer_radius = opt_params[-1]
            
            # Validate final result
            is_valid, objective_value = self.validator.validate_configuration(opt_inner_data, opt_outer_radius)
            
            if is_valid:
                return opt_inner_data, opt_outer_radius, -objective_value
            else:
                return None, None, -1e10
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            return None, None, -1e10

    def find_optimal_packing(self, initial_config: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Main optimization driver with improved parallel processing"""
        # Generate multiple initial configurations
        initial_configs = self._generate_initial_configurations()
        
        # Run parallel optimizations with limited timeout
        results = Parallel(n_jobs=self.num_parallel, timeout=160)(
            delayed(self._optimize_single)(config) for config in initial_configs
        )
        
        # Find best result
        best_result = None
        best_score = -1e10
        
        for result in results:
            if result[0] is not None and result[2] > best_score:
                best_score = result[2]
                best_result = result
        
        # If no good result found, fallback to single optimization with larger population
        if best_result is None or best_score < 0.15:
            # Try single optimization with even more iterations
            try:
                # Use the first configuration as basis for extra attempt
                best_result = self._optimize_single(initial_config)
            except:
                pass
        
        if best_result is None or best_result[0] is None:
            return None, None, -1e10
        
        return best_result[0], best_result[1], best_result[2]

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize optimizer
    optimizer = HexagonPackingOptimizer()
    
    # Initial configuration from the simple grid (same as baseline)
    initial_config = np.array([
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
    
    # Attempt optimization
    try:
        inner_hex_data, outer_hex_side_length, inv_side_length = optimizer.find_optimal_packing(initial_config)
        
        # If optimization succeeded with reasonable results
        if inner_hex_data is not None and inv_side_length > 0.15:
            outer_hex_data = np.array([0, 0, 0])
            return inner_hex_data, outer_hex_data, outer_hex_side_length
    except Exception as e:
        # Silently handle errors and fall back
        warnings.warn(f"Optimization error: {str(e)}")
        pass
    
    # Fallback to original approach if optimization fails
    # Set reasonable initial outer hexagon size based on configuration
    max_dist_from_center = 0
    for i in range(len(initial_config)):
        center_x, center_y, _ = initial_config[i]
        dist = np.sqrt(center_x**2 + center_y**2)
        max_dist_from_center = max(max_dist_from_center, dist + 1.0)  # Add radius margin
    
    # Outer hexagon should have side length slightly larger than max distance
    outer_hex_side_length = max_dist_from_center * 1.15  # Smaller margin for better optimization
    
    # Evaluate this configuration
    validator = HexagonValidation(HexagonGeometry())
    valid, _ = validator.validate_configuration(initial_config, outer_hex_side_length)
    
    # If initial configuration is invalid due to overlap or containment,
    # we fall back to the simpler approach but with better validation
    if not valid:
        # Fallback to a basic valid configuration
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
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0  # fallback value
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Since we've confirmed initial config works, we can return it
    inner_hex_data = initial_config.copy()
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END