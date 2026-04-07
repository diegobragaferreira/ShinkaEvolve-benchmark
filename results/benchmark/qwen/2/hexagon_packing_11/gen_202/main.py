# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
from scipy.spatial.distance import cdist
import time
import warnings
import math

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2
BENCHMARK_RATIO = 0.2544

class HexagonPacker:
    def __init__(self, num_inner=11):
        self.num_inner = num_inner
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
    
    def point_in_polygon_buffered(self, point, polygon, buffer_distance=1e-6):
        """Check if point is inside polygon with small buffer"""
        buffered = polygon.buffer(buffer_distance)
        return buffered.contains(Point(point))
    
    def check_containment(self, inner_hexagon, outer_hexagon, buffer=1e-6):
        """Check if inner hexagon is fully contained within outer hexagon with buffer"""
        # Check if all vertices of inner hexagon are inside outer hexagon
        for point in inner_hexagon.exterior.coords[:-1]:
            if not self.point_in_polygon_buffered(point, outer_hexagon, buffer):
                return False
        return True

    def check_overlap(self, hex1, hex2, buffer=1e-6):
        """Check if two hexagons overlap with buffer"""
        # Add small buffer to both polygons to prevent floating-point precision issues
        buffered_hex1 = hex1.buffer(buffer)
        buffered_hex2 = hex2.buffer(buffer)
        return buffered_hex1.intersects(buffered_hex2)
    
    def calculate_outer_hex_radius(self, inner_hex_data, outer_center=(0,0)):
        """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
        max_dist = 0
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            dist = np.linalg.norm(np.array(center) - np.array(outer_center))
            # Add distance from center to corner of unit hexagon
            dist += self.unit_hex_apogee
            max_dist = max(max_dist, dist)
        return max_dist

    def evaluate_constraints(self, inner_params, outer_radius):
        """Constraint evaluation with early termination"""
        inner_hexagons = []
        
        # Create inner hexagons
        for i in range(self.num_inner):
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
        for i in range(self.num_inner):
            for j in range(i+1, self.num_inner):
                if self.check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    return False, False, 0.0  # overlap violated
                    
        return True, True, 1.0 / outer_radius  # valid solution

def generate_voronoi_initialization(packer, num_samples=20):
    """Generate diverse initial configurations using Voronoi-based spatial sampling"""
    configs = []
    
    # Generate Voronoi points with specific geometric properties
    # Generate initial points using a modified grid to avoid regularity
    voronoi_points = []
    # Center point
    voronoi_points.append([0.0, 0.0])
    
    # Generate points in rings around center
    num_rings = 4
    ring_sizes = [1, 4, 6, 10]  # Points per ring (approximately matching hexagonal packing)
    
    for ring_idx in range(1, num_rings):
        num_points = ring_sizes[ring_idx - 1]  # Adjust for indexing
        radius = 1.5 + ring_idx * 1.0
        for i in range(num_points):
            angle = (i * 2 * np.pi / num_points) + (np.pi if ring_idx % 2 == 0 else 0)  
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # Add slight noise to break symmetry
            noise_scale = 0.3
            x += np.random.normal(0, noise_scale)
            y += np.random.normal(0, noise_scale)
            voronoi_points.append([x, y])
    
    # Ensure we have exactly 11 points  
    voronoi_points = voronoi_points[:11]
    
    # Create diverse samples with Voronoi-inspired arrangements
    for _ in range(num_samples):
        config = []
        # Sort points to ensure consistent ordering
        sorted_points = sorted(voronoi_points, key=lambda p: np.arctan2(p[1], p[0]))
        
        for i, (cx, cy) in enumerate(sorted_points):
            # Add strategic jitter
            jitter_x = np.random.normal(0, 0.3)
            jitter_y = np.random.normal(0, 0.3)
            config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])
        
        # Add outer radius estimate with better distribution
        # Estimate based on the spread of points
        points_array = np.array([(x, y) for x, y in sorted_points])
        centroid = points_array.mean(axis=0)
        distances = np.sqrt(np.sum((points_array - centroid)**2, axis=1))
        max_dist = np.max(distances) + 1.5  # Add extra margin for hexagon size
        
        config.append(max_dist + np.random.uniform(0.5, 2.0))
        configs.append(config)
    
    return configs

def voronoi_constraint_evaluator(individual, packer):
    """Enhanced constraint evaluator with better handling of edge cases"""
    try:
        # Extract parameters 
        n = packer.num_inner
        inner_params = individual[:-1]
        outer_radius = individual[-1]
        
        # Check constraints
        containment_ok, overlap_ok, inv_radius = packer.evaluate_constraints(inner_params, outer_radius)
        
        # If any constraint violated, return large penalty
        if not (containment_ok and overlap_ok):
            return 10000.0 + abs(outer_radius)  # penalty for constraint violations
        
        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius
    
    except Exception as e:
        return 10000.0  # penalty for exceptions

def hexagon_packing_voronoi():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses Voronoi-based initialization and multi-stage optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize packer
    packer = HexagonPacker()
    
    # Step 1: Generate Voronoi-based initial population
    population = generate_voronoi_initialization(packer, 30)
    
    # Track best solution
    best_inv_radius = -float('inf')
    best_individual = None
    best_score = float('inf')
    
    # Multi-start optimization with Voronoi-based seeds
    # Use specific seeds that should produce good starting points
    seeds = [42, 123, 456, 789, 999, 555, 111, 222, 333, 666]
    
    for seed in seeds:
        try:
            np.random.seed(seed)
            
            # Generate new population for this seed
            current_population = generate_voronoi_initialization(packer, 15)
            
            # Evaluate population sequentially (more stable than parallel)
            for individual in current_population:
                score = voronoi_constraint_evaluator(individual, packer)
                if score < best_score and score < 10000.0:
                    best_score = score
                    best_individual = individual[:]
                    
        except Exception as e:
            warnings.warn(f"Voronoi optimization run failed: {str(e)}")
            continue
    
    # Enhanced local refinement
    if best_individual is not None:
        try:
            # Prepare bounds for local optimization
            bounds = []
            for _ in range(packer.num_inner):
                bounds.extend([(-10.0, 10.0), (-10.0, 10.0), (0, 360)])  # x, y, angle
            bounds.append((2.0, 15.0))  # outer radius
            
            # Define objective function for local optimization
            def local_objective(params):
                return voronoi_constraint_evaluator(params, packer)
            
            # Run L-BFGS-B optimization with more precise settings
            result = minimize(
                local_objective,
                best_individual,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            
            if result.success:
                refined_individual = result.x
                # Check if refined solution is better
                current_inv_radius = -best_score
                refined_inv_radius = -local_objective(refined_individual)
                
                if refined_inv_radius > current_inv_radius and refined_inv_radius > 0:
                    best_individual = refined_individual
                    best_score = -refined_inv_radius
                    
        except Exception as e:
            warnings.warn(f"Local refinement failed: {str(e)}")
    
    # Final evaluation and formatting
    if best_individual is not None:
        inner_params = best_individual[:-1]
        outer_radius = best_individual[-1]
        
        # Validate final solution
        containment_ok, overlap_ok, inv_radius = packer.evaluate_constraints(inner_params, outer_radius)
        
        if containment_ok and overlap_ok:
            # Format output
            inner_hex_data = np.zeros((packer.num_inner, 3))
            for i in range(packer.num_inner):
                inner_hex_data[i] = inner_params[3*i:3*i+3]
            
            outer_hex_data = np.array([0, 0, 0])
            
            return inner_hex_data, outer_hex_data, outer_radius
    
    # Fallback to best available configuration from initial samples
    best_sample = None
    best_score_fallback = float('inf')
    
    for sample in population:
        try:
            inner_params = sample[:-1]
            outer_radius = sample[-1]
            containment_ok, overlap_ok, inv_radius = packer.evaluate_constraints(inner_params, outer_radius)
            
            if containment_ok and overlap_ok and inv_radius > 0 and inv_radius < best_score_fallback:
                best_score_fallback = inv_radius
                best_sample = sample
        except Exception:
            continue
    
    if best_sample is not None:
        inner_params = best_sample[:-1]
        outer_radius = best_sample[-1]
        
        inner_hex_data = np.zeros((packer.num_inner, 3))
        for i in range(packer.num_inner):
            inner_hex_data[i] = inner_params[3*i:3*i+3]
        
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_radius
    
    # Ultimate fallback to simple configuration
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

def hexagon_packing_11():
    return hexagon_packing_voronoi()

# EVOLVE-BLOCK-END