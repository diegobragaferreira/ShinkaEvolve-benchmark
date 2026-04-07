# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import numba
from numba import jit
import time
from joblib import Parallel, delayed
import math
from collections import defaultdict
import random

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
UNIT_HEXAGON_APOGEE = UNIT_HEXAGON_RADIUS * np.sqrt(3) / 2
MAX_EVAL_TIME = 180.0
BENCHMARK_RATIO_TARGET = 0.2544

@jit(nopython=True)
def hexagon_vertices_numba(x, y, angle_deg, side_length=1):
    """Calculate vertices of a regular hexagon using numba for speed"""
    angle_rad = np.deg2rad(angle_deg)
    vertices = np.zeros((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + side_length * np.cos(theta)
        vertices[i, 1] = y + side_length * np.sin(theta)
    return vertices

@jit(nopython=True)
def point_in_hexagon_numba(px, py, hex_vertices):
    """Fast point-in-polygon check for hexagon using ray casting"""
    n = len(hex_vertices)
    inside = False
    p1x, p1y = hex_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = hex_vertices[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class Hexagon:
    """Represents a regular hexagon with position, rotation, and size"""
    
    def __init__(self, center_x, center_y, rotation_deg, side_length=1):
        self.center_x = center_x
        self.center_y = center_y
        self.rotation_deg = rotation_deg
        self.side_length = side_length
    
    def get_vertices(self):
        """Get the 6 vertices of the hexagon"""
        return hexagon_vertices_numba(self.center_x, self.center_y, self.rotation_deg, self.side_length)
    
    def get_bounding_box(self):
        """Get the bounding box coordinates of the hexagon"""
        vertices = self.get_vertices()
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        return np.min(xs), np.max(xs), np.min(ys), np.max(ys)
    
    def contains_point(self, px, py):
        """Check if a point is inside this hexagon"""
        vertices = self.get_vertices()
        return point_in_hexagon_numba(px, py, vertices)

class HexagonPacker:
    """Handles the core packing logic and constraint validation"""
    
    def __init__(self, inner_side_length=1.0):
        self.inner_side_length = inner_side_length
        self.hex_apothem = inner_side_length * np.sqrt(3) / 2
    
    def get_inner_hexagons(self, positions, rotations):
        """Create list of Hexagon objects from positions and rotations"""
        return [Hexagon(pos[0], pos[1], rot) for pos, rot in zip(positions, rotations)]
    
    def build_spatial_index(self, hexagons, cell_size=2.8):
        """Build a grid-based spatial index for efficient collision detection"""
        index = defaultdict(list)
        for i, hexagon in enumerate(hexagons):
            min_x, max_x, min_y, max_y = hexagon.get_bounding_box()
            # Get grid coordinates for the bounding box
            start_grid_x = int(np.floor(min_x / cell_size))
            end_grid_x = int(np.ceil(max_x / cell_size))
            start_grid_y = int(np.floor(min_y / cell_size))
            end_grid_y = int(np.ceil(max_y / cell_size))
            
            # Add to all relevant grid cells
            for gx in range(start_grid_x, end_grid_x + 1):
                for gy in range(start_grid_y, end_grid_y + 1):
                    index[(gx, gy)].append(i)
        return index
    
    def get_potential_collisions(self, hexagons, spatial_index, cell_size=2.8):
        """Get potential collision pairs using spatial indexing"""
        collisions = set()
        for i, hexagon in enumerate(hexagons):
            min_x, max_x, min_y, max_y = hexagon.get_bounding_box()
            # Get grid coordinates for the bounding box
            start_grid_x = int(np.floor(min_x / cell_size))
            end_grid_x = int(np.ceil(max_x / cell_size))
            start_grid_y = int(np.floor(min_y / cell_size))
            end_grid_y = int(np.ceil(max_y / cell_size))
            
            # Check neighboring cells
            for gx in range(start_grid_x - 1, end_grid_x + 2):
                for gy in range(start_grid_y - 1, end_grid_y + 2):
                    if (gx, gy) in spatial_index:
                        for j in spatial_index[(gx, gy)]:
                            if i < j:  # Avoid duplicates and self-checking
                                collisions.add((i, j))
        return collisions
    
    def check_containment(self, inner_hexagons, outer_polygon):
        """Check that all inner hexagons are contained within outer polygon"""
        for hexagon in inner_hexagons:
            vertices = hexagon.get_vertices()
            # Check if all vertices are inside the outer polygon
            for vx, vy in vertices:
                if not outer_polygon.contains(Point(vx, vy)):
                    return False
        return True
    
    def check_overlaps(self, inner_hexagons):
        """Check for overlaps between any pair of hexagons"""
        # Build spatial index to optimize collision detection
        spatial_index = self.build_spatial_index(inner_hexagons, cell_size=2.8)
        potential_collisions = self.get_potential_collisions(inner_hexagons, spatial_index, cell_size=2.8)
        
        # Check actual intersections
        for i, j in potential_collisions:
            hex1 = inner_hexagons[i]
            hex2 = inner_hexagons[j]
            poly1 = Polygon(hex1.get_vertices())
            poly2 = Polygon(hex2.get_vertices())
            if poly1.intersects(poly2):
                return True
        return False
    
    def estimate_outer_side_length(self, inner_positions, padding=0.1):
        """Estimate minimum outer hexagon side length to contain all inner hexagons"""
        if len(inner_positions) == 0:
            return 1000
        
        # Flatten positions to array for easier processing
        positions = np.array(inner_positions)
        
        # Get all vertices of all hexagons
        all_vertices = []
        for i, (x, y) in enumerate(positions):
            hexagon = Hexagon(x, y, 0, self.inner_side_length)
            vertices = hexagon.get_vertices()
            all_vertices.extend(vertices)
        
        if len(all_vertices) == 0:
            return 1000
        
        # Calculate bounding box
        all_vertices = np.array(all_vertices)
        min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
        min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
        
        # Calculate center
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Find maximum distance from center to any vertex
        max_dist = 0
        for vx, vy in all_vertices:
            dist = np.sqrt((vx - center_x)**2 + (vy - center_y)**2)
            max_dist = max(max_dist, dist)
        
        # Convert to hexagon side length
        outer_side_length = 2 * max_dist / np.sqrt(3)
        
        return outer_side_length + padding  # Add padding for safety

class Optimizer:
    """Handles the optimization process"""
    
    def __init__(self, packer, max_iterations=800, population_size=30):
        self.packer = packer
        self.max_iterations = max_iterations
        self.population_size = population_size
    
    def evaluate_fitness(self, params):
        """Evaluate a candidate solution - returns negative because we minimize"""
        # Unpack parameters: 11 positions (x,y) + 11 rotations 
        n_hexagons = 11
        inner_positions = params[:2*n_hexagons].reshape(-1, 2)
        inner_rotations = params[2*n_hexagons:3*n_hexagons]
        
        # Create inner hexagons
        inner_hexagons = self.packer.get_inner_hexagons(inner_positions, inner_rotations)
        
        # Estimate outer hexagon side length based on inner positions
        estimated_side_length = self.packer.estimate_outer_side_length(inner_positions)
        
        # Use a fixed outer hexagon configuration for constraint checks
        # Outer hexagon centered at origin, no rotation, side length based on estimation
        outer_hex = Hexagon(0, 0, 0, estimated_side_length)
        outer_vertices = outer_hex.get_vertices()
        outer_polygon = Polygon(outer_vertices)
        
        # Check constraints
        try:
            if not self.packer.check_containment(inner_hexagons, outer_polygon):
                # Scale penalty based on how severe the containment violation is
                max_dist = 0
                for hexagon in inner_hexagons:
                    vertices = hexagon.get_vertices()
                    for vx, vy in vertices:
                        dist = np.sqrt((vx)**2 + (vy)**2)
                        max_dist = max(max_dist, dist)
                penalty = 10000000 + (max_dist - estimated_side_length) * 10000
                return penalty
            
            if self.packer.check_overlaps(inner_hexagons):
                # Scale penalty based on overlap severity
                penalty = 10000000 + 1000000  # Large penalty
                return penalty
            
            # Return negative of 1/outer_side_length to maximize 1/outer_side_length
            return -1.0 / estimated_side_length
        except Exception as e:
            return 1e10
    
    def run_evolutionary_optimization(self, initial_guess, bounds):
        """Run evolutionary optimization with adaptive scheduling"""
        # Use the best differential evolution approach with better parameters
        result = differential_evolution(
            self.evaluate_fitness,
            bounds,
            maxiter=self.max_iterations,
            popsize=self.population_size,
            mutation=(0.8, 1.0),  # Slightly higher mutation rate for better exploration
            recombination=0.7,
            seed=42,
            disp=False,
            polish=True
        )
        return result

class MultiStartOptimizer:
    """Handles multi-start optimization with diverse initial configurations"""
    
    def __init__(self, packer, optimizer, num_starts=8):
        self.packer = packer
        self.optimizer = optimizer
        self.num_starts = num_starts
    
    def generate_voronoi_initialization(self, num_points=15):
        """Generate initial configuration using Voronoi diagram for better distribution"""
        # Generate random points
        np.random.seed(random.randint(1, 10000))
        points = np.random.rand(num_points, 2) * 12 - 6  # Random points in [-6, 6]^2
        
        # Sample points ensuring we get good spread
        valid_centers = []
        for i, (x, y) in enumerate(points):
            # Check if point is reasonably far from edges and within bounds
            if abs(x) < 5 and abs(y) < 5 and len(valid_centers) < NUM_INNER_HEXAGONS:
                valid_centers.append([x, y])
        
        # If we don't have enough points, fill with additional random ones
        while len(valid_centers) < NUM_INNER_HEXAGONS:
            valid_centers.append([random.uniform(-5, 5), random.uniform(-5, 5)])
        
        # Sample only the required number
        valid_centers = valid_centers[:NUM_INNER_HEXAGONS]
        
        # Create positions and angles
        positions = np.array(valid_centers)
        angles = np.zeros(NUM_INNER_HEXAGONS)
        
        return positions, angles
    
    def generate_initial_configs(self):
        """Generate multiple diverse initial configurations"""
        configs = []
        
        # Configuration 1: Hexagonal pattern around center
        center_config = np.array([
            [0, 0, 0],      # Center
            [-1.75, 0, 0],  # Left
            [1.75, 0, 0],   # Right
            [0, 1.75, 0],   # Top
            [0, -1.75, 0],  # Bottom
            [-0.875, 0.875, 0],  # Top-left
            [0.875, 0.875, 0],   # Top-right
            [-0.875, -0.875, 0], # Bottom-left
            [0.875, -0.875, 0],  # Bottom-right
            [-1.75, 1.75, 0],    # Far-top-left
            [1.75, 1.75, 0],     # Far-top-right
        ])
        configs.append(center_config)
        
        # Configuration 2: Spiral pattern
        spiral_config = np.array([
            [0, 0, 0],      # Center
            [1.5, 0, 0],    # Right
            [0, 1.5, 0],    # Top
            [-1.5, 0, 0],   # Left
            [0, -1.5, 0],   # Bottom
            [1.5, 1.5, 0],  # Top-right
            [-1.5, 1.5, 0], # Top-left
            [-1.5, -1.5, 0], # Bottom-left
            [1.5, -1.5, 0],  # Bottom-right
            [3.0, 0, 0],     # Far right
            [0, 3.0, 0],     # Far top
        ])
        configs.append(spiral_config)
        
        # Configuration 3: Clustered
        cluster_config = np.array([
            [0, 0, 0],       # Center
            [1.25, 0, 0],    # Right
            [-1.25, 0, 0],   # Left
            [0, 1.25, 0],    # Top
            [0, -1.25, 0],   # Bottom
            [1.25, 1.25, 0], # Top-right
            [-1.25, 1.25, 0], # Top-left
            [-1.25, -1.25, 0], # Bottom-left
            [1.25, -1.25, 0],  # Bottom-right
            [2.5, 0, 0],       # Far right
            [0, 2.5, 0],       # Far top
        ])
        configs.append(cluster_config)
        
        # Configuration 4: Staggered pattern
        staggered_config = np.array([
            [0, 0, 0],       # Center
            [1.25, 0, 0],    # Right
            [-1.25, 0, 0],   # Left
            [0, 1.25, 0],    # Top
            [0, -1.25, 0],   # Bottom
            [0.625, 1.25, 0], # Top-right
            [-0.625, 1.25, 0], # Top-left
            [-0.625, -1.25, 0], # Bottom-left
            [0.625, -1.25, 0],  # Bottom-right
            [1.875, 1.25, 0],   # Far top-right
            [1.875, -1.25, 0],  # Far bottom-right
        ])
        configs.append(staggered_config)
        
        # Configuration 5: Random distribution with Voronoi
        voronoi_positions, voronoi_angles = self.generate_voronoi_initialization()
        voronoi_config = np.column_stack([voronoi_positions, voronoi_angles])
        configs.append(voronoi_config)
        
        # Configuration 6: Another hex pattern
        hex_pattern_config = np.array([
            [0, 0, 0],       # Center
            [-2.0, 0, 0],    # Left
            [2.0, 0, 0],     # Right
            [0, 2.0, 0],     # Top
            [0, -2.0, 0],    # Bottom
            [-1.0, 1.0, 0],  # Top-left
            [1.0, 1.0, 0],   # Top-right
            [-1.0, -1.0, 0], # Bottom-left
            [1.0, -1.0, 0],  # Bottom-right
            [-2.5, 2.5, 0],  # Far top-left
            [2.5, 2.5, 0],   # Far top-right
        ])
        configs.append(hex_pattern_config)
        
        # Configuration 7: Ring pattern
        ring_config = np.array([
            [0, 0, 0],       # Center
            [-1.5, 0, 0],    # Left
            [1.5, 0, 0],     # Right
            [0, 1.5, 0],     # Top
            [0, -1.5, 0],    # Bottom
            [-1.5, 1.5, 0],  # Top-left
            [1.5, 1.5, 0],   # Top-right
            [-1.5, -1.5, 0], # Bottom-left
            [1.5, -1.5, 0],  # Bottom-right
            [0, 3.0, 0],     # Far top
            [0, -3.0, 0],    # Far bottom
        ])
        configs.append(ring_config)
        
        # Configuration 8: Random noise added to center config
        random_config = center_config.copy()
        for i in range(NUM_INNER_HEXAGONS):
            random_config[i][0] += np.random.normal(0, 0.3)
            random_config[i][1] += np.random.normal(0, 0.3)
            random_config[i][2] += np.random.normal(0, 10)
        configs.append(random_config)
        
        return configs
    
    def run_multi_start_optimization(self):
        """Run optimization from multiple starting points"""
        initial_configs = self.generate_initial_configs()
        best_result = None
        best_score = -np.inf
        best_params = None
        
        # Run optimization from each initial configuration in parallel
        def run_single_optimization(config_data):
            try:
                # Set up bounds for optimization
                bounds = []
                n_hexagons = 11
                
                # Inner positions (x,y)
                for j in range(n_hexagons):
                    bounds.extend([(-12, 12), (-12, 12)])  # Slightly extended bounds for better search
                
                # Inner angles (0 to 360)
                for _ in range(n_hexagons):
                    bounds.extend([(0, 360)])
                
                # Create initial guess
                initial_guess = []
                # Add inner positions
                for j in range(n_hexagons):
                    initial_guess.extend([config_data[j][0], config_data[j][1]])
                # Add inner angles
                for j in range(n_hexagons):
                    initial_guess.extend([config_data[j][2]])
                
                # Run optimization
                result = self.optimizer.run_evolutionary_optimization(initial_guess, bounds)
                
                if result is not None:
                    final_score = self.optimizer.evaluate_fitness(result.x)
                    return final_score, result.x
                else:
                    return -np.inf, None
                    
            except Exception as e:
                return -np.inf, None
        
        # Parallel execution of optimizations
        results = Parallel(n_jobs=min(8, len(initial_configs)))(delayed(run_single_optimization)(config) for config in initial_configs)
        
        # Find best result
        for score, params in results:
            if params is not None and score > best_score:
                best_score = score
                best_params = params
        
        return best_params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize core components
    packer = HexagonPacker()
    optimizer = Optimizer(packer, max_iterations=800, population_size=30)
    multi_optimizer = MultiStartOptimizer(packer, optimizer, num_starts=8)
    
    # Run multi-start optimization
    best_params = multi_optimizer.run_multi_start_optimization()
    
    if best_params is None:
        # Fallback to baseline configuration
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
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = 8  # large enough to contain all inner hexagons
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Extract results from best solution
    n_hexagons = 11
    inner_positions = best_params[:2*n_hexagons].reshape(-1, 2)
    inner_rotations = best_params[2*n_hexagons:3*n_hexagons]
    
    # Construct inner hex data
    inner_hex_data = np.column_stack([inner_positions, inner_rotations])
    
    # Construct outer hex data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    # Estimate outer side length
    outer_hex_side_length = 0  # Will be computed from the optimization
    # Recompute based on final positions as done in the original
    all_vertices = []
    for i, (x, y) in enumerate(inner_positions):
        hexagon = Hexagon(x, y, inner_rotations[i], 1.0)
        vertices = hexagon.get_vertices()
        all_vertices.extend(vertices)
    
    if len(all_vertices) > 0:
        all_vertices = np.array(all_vertices)
        min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
        min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        max_dist = 0
        for vx, vy in all_vertices:
            dist = np.sqrt((vx - center_x)**2 + (vy - center_y)**2)
            max_dist = max(max_dist, dist)
        outer_hex_side_length = 2 * max_dist / np.sqrt(3)
    
    # Final validation to ensure correctness
    try:
        # Check final validation
        test_hexagons = [Hexagon(pos[0], pos[1], rot) for pos, rot in zip(inner_positions, inner_rotations)]
        outer_hex = Hexagon(0, 0, 0, outer_hex_side_length)
        outer_vertices = outer_hex.get_vertices()
        outer_polygon = Polygon(outer_vertices)
        
        if not packer.check_containment(test_hexagons, outer_polygon):
            # If containment fails, fall back to safe configuration
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
            outer_hex_side_length = 8
    except Exception:
        # If there's any issue, fall back to safe configuration
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
        outer_hex_side_length = 8

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END