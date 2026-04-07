# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.validation import make_valid
import time
from collections import defaultdict
import math

class HexagonPacker:
    def __init__(self):
        self.hexagon_cache = {}
        self.bbox_grid = defaultdict(list)
        self.grid_size = 2.0  # Size of grid cells for spatial indexing
    
    def get_hexagon_vertices(self, center_x, center_y, side_length=1, rotation_deg=0):
        """Get vertices of a regular hexagon with caching"""
        cache_key = (center_x, center_y, side_length, rotation_deg)
        if cache_key in self.hexagon_cache:
            return self.hexagon_cache[cache_key]
        
        rotation_rad = np.radians(rotation_deg)
        angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
        x_coords = center_x + side_length * np.cos(angles)
        y_coords = center_y + side_length * np.sin(angles)
        vertices = list(zip(x_coords, y_coords))
        self.hexagon_cache[cache_key] = vertices
        return vertices
    
    def create_regular_hexagon(self, center=(0, 0), side_length=1, rotation=0):
        """Create a regular hexagon as a shapely polygon"""
        vertices = self.get_hexagon_vertices(center[0], center[1], side_length, rotation)
        return Polygon(vertices)
    
    def get_bbox(self, vertices):
        """Get bounding box of hexagon vertices"""
        xs, ys = zip(*vertices)
        return (min(xs), min(ys), max(xs), max(ys))
    
    def get_grid_cell(self, x, y):
        """Get grid cell coordinates for point"""
        return (int(x // self.grid_size), int(y // self.grid_size))
    
    def update_grid(self, hex_id, vertices):
        """Update spatial grid with hexagon"""
        bbox = self.get_bbox(vertices)
        x1, y1, x2, y2 = bbox
        for x in range(int(x1 // self.grid_size), int(x2 // self.grid_size) + 1):
            for y in range(int(y1 // self.grid_size), int(y2 // self.grid_size) + 1):
                self.bbox_grid[(x, y)].append(hex_id)
    
    def get_potential_collisions(self, vertices):
        """Get potential collision candidates using spatial indexing"""
        bbox = self.get_bbox(vertices)
        x1, y1, x2, y2 = bbox
        candidates = set()
        for x in range(int(x1 // self.grid_size), int(x2 // self.grid_size) + 1):
            for y in range(int(y1 // self.grid_size), int(y2 // self.grid_size) + 1):
                candidates.update(self.bbox_grid[(x, y)])
        return candidates
    
    def clear_grid(self):
        """Clear the spatial grid"""
        self.bbox_grid.clear()
        self.hexagon_cache.clear()

class ConstraintChecker:
    def __init__(self, packer):
        self.packer = packer
    
    def check_containment(self, hexagon_poly, outer_hex_poly):
        """Check if hexagon is fully contained within outer hexagon"""
        try:
            return outer_hex_poly.contains(hexagon_poly)
        except:
            try:
                valid_outer = make_valid(outer_hex_poly)
                valid_hex = make_valid(hexagon_poly)
                return valid_outer.contains(valid_hex)
            except:
                return False
    
    def check_overlap_fast(self, hex1_vertices, hex2_vertices):
        """Fast overlap check using spatial indexing and bounding boxes"""
        # Quick bounding box test first
        bbox1 = self.packer.get_bbox(hex1_vertices)
        bbox2 = self.packer.get_bbox(hex2_vertices)
        
        if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
            bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
            return False
        
        # Spatial indexing check
        candidates = self.packer.get_potential_collisions(hex1_vertices)
        for hex_id in candidates:
            if hex_id == hash(tuple(hex2_vertices)):  # Skip self
                continue
            # Full polygon intersection test
            try:
                poly1 = Polygon(hex1_vertices)
                poly2 = Polygon(hex2_vertices)
                if poly1.intersects(poly2):
                    return True
            except:
                try:
                    valid_poly1 = make_valid(Polygon(hex1_vertices))
                    valid_poly2 = make_valid(Polygon(hex2_vertices))
                    if valid_poly1.intersects(valid_poly2):
                        return True
                except:
                    pass
        
        return False
    
    def check_overlap(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely"""
        try:
            poly1 = Polygon(hex1_vertices)
            poly2 = Polygon(hex2_vertices)
            return poly1.intersects(poly2)
        except:
            try:
                valid_poly1 = make_valid(Polygon(hex1_vertices))
                valid_poly2 = make_valid(Polygon(hex2_vertices))
                return valid_poly1.intersects(valid_poly2)
            except:
                return True  # if we can't validate, assume they overlap

class Optimizer:
    def __init__(self, packer, checker):
        self.packer = packer
        self.checker = checker
        self.best_score = float('-inf')
        self.best_params = None
    
    def evaluate_solution(self, params, outer_hex_radius=None):
        """Evaluate a solution and return negative combined score (for minimization)"""
        # Clear spatial grid for fresh computation
        self.packer.clear_grid()
        
        # Parse parameters
        inner_positions = params[:22].reshape(-1, 2)  # x,y pairs
        inner_rotations = params[22:33]  # 11 rotations
        if outer_hex_radius is None:
            outer_radius = params[33]  # outer hex radius
        else:
            outer_radius = outer_hex_radius

        # Create outer hexagon
        outer_hex = self.packer.create_regular_hexagon((0, 0), outer_radius, 0)

        # Check if all inner hexagons fit inside outer hexagon
        num_inner_hexes = len(inner_positions)

        # Create all inner hexagon polygons
        inner_hexes = []
        for i in range(num_inner_hexes):
            pos = tuple(inner_positions[i])
            rot = inner_rotations[i]
            vertices = self.packer.get_hexagon_vertices(pos[0], pos[1], 1, rot)
            inner_hexes.append(vertices)
            
            # Update spatial grid
            self.packer.update_grid(hash(tuple(vertices)), vertices)

        # Check containment and overlaps
        total_penalty = 0
        for i, hex_vertices in enumerate(inner_hexes):
            hex_poly = Polygon(hex_vertices)
            if not self.checker.check_containment(hex_poly, outer_hex):
                total_penalty += 1000  # Large penalty for containment violations

            # Check overlap with other hexes using spatial indexing
            for j in range(i+1, len(inner_hexes)):
                if self.checker.check_overlap_fast(hex_vertices, inner_hexes[j]):
                    total_penalty += 10000  # Large penalty for overlaps

        # Calculate combined score (negative since we want to minimize)
        inv_radius = 1.0 / outer_radius if outer_radius > 0 else 0
        score = -inv_radius + total_penalty
        
        # Track best solution
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
        
        return score

def generate_diverse_initial_configs():
    """Generate multiple diverse initial configurations"""
    configs = []
    
    # Configuration 1: Hexagonal lattice arrangement
    base_config = [
        (0, 0, 0),  # center
        (-2.5, 0, 0),  # left
        (2.5, 0, 0),  # right
        (-1.25, 2.17, 0),  # top-left
        (1.25, 2.17, 0),  # top-right
        (-1.25, -2.17, 0),  # bottom-left
        (1.25, -2.17, 0),  # bottom-right
        (-3.75, 2.17, 0),  # far top-left
        (3.75, 2.17, 0),  # far top-right
        (-3.75, -2.17, 0),  # far bottom-left
        (3.75, -2.17, 0),  # far bottom-right
    ]
    
    # Configuration 2: Spiral pattern
    spiral_config = [
        (0, 0, 0),
        (2, 0, 0),
        (1, 1.732, 0),
        (-1, 1.732, 0),
        (-2, 0, 0),
        (-1, -1.732, 0),
        (1, -1.732, 0),
        (3, 0, 0),
        (2, 2.17, 0),
        (-2, 2.17, 0),
        (-3, 0, 0)
    ]
    
    # Configuration 3: Clustered arrangement
    cluster_config = [
        (0, 0, 0),
        (1.5, 0, 0),
        (-1.5, 0, 0),
        (0, 1.5, 0),
        (0, -1.5, 0),
        (1.5, 1.5, 0),
        (-1.5, 1.5, 0),
        (1.5, -1.5, 0),
        (-1.5, -1.5, 0),
        (3, 0, 0),
        (0, 3, 0)
    ]
    
    # Configuration 4: Cross pattern
    cross_config = [
        (0, 0, 0),
        (0, 2.5, 0),
        (0, -2.5, 0),
        (2.5, 0, 0),
        (-2.5, 0, 0),
        (1.25, 2.17, 0),
        (-1.25, 2.17, 0),
        (1.25, -2.17, 0),
        (-1.25, -2.17, 0),
        (3.75, 0, 0),
        (0, 3.75, 0)
    ]
    
    # Configuration 5: Random-like but structured
    random_like_config = [
        (0, 0, 0),
        (2, 1, 30),
        (-1.5, 2, 60),
        (-2.5, -1, 90),
        (1, -2, 120),
        (3, 1.5, 150),
        (-2, 1.5, 180),
        (1.5, -2.5, 210),
        (-1.5, -2.5, 240),
        (3.5, -1, 270),
        (0, 3.5, 300)
    ]
    
    configs.extend([base_config, spiral_config, cluster_config, cross_config, random_like_config])
    
    return configs

def create_initial_guess(config):
    """Create initial parameter vector from configuration"""
    guess = []
    for x, y, angle in config:
        guess.extend([x, y, angle])
    guess.append(8.0)  # Initial outer radius guess
    return np.array(guess)

def adaptive_mutation_schedule(iteration, maxiter):
    """Adaptive mutation rate with improved scheduling for better exploration-exploitation balance"""
    if maxiter <= 0:
        return 0.5
    progress = iteration / maxiter

    # Use exponential decay with better tuning
    return 0.1 + 0.7 * np.exp(-3 * progress)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization to find a better solution than the simple grid arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize components
    packer = HexagonPacker()
    checker = ConstraintChecker(packer)
    optimizer = Optimizer(packer, checker)
    
    # Set up bounds for optimization
    bounds = []
    # Position bounds (-10, 10) for each of 11 hexagons
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10)])
    # Rotation bounds (0, 360) for each of 11 hexagons
    for _ in range(11):
        bounds.append((0, 360))
    # Outer radius bounds (1, 20)
    bounds.append((1, 20))

    # Generate diverse initial populations
    initial_configs = generate_diverse_initial_configs()
    initial_populations = [create_initial_guess(config) for config in initial_configs]
    
    best_result = None
    best_fitness = float('inf')
    
    # Try each initial population
    for i, initial_guess in enumerate(initial_populations):
        if time.time() - start_time > 170:  # Leave some buffer
            break
            
        # Run optimization with adaptive mutation
        result = differential_evolution(
            optimizer.evaluate_solution,
            bounds,
            maxiter=150,
            popsize=20,
            seed=42+i,  # Different seed for each run
            callback=lambda x, convergence: None,  # Suppress output
            disp=False,
            mutation=(0.5, 1.0)  # Default mutation range
        )
        
        # Update best result if this one is better
        if result.fun < best_fitness:
            best_fitness = result.fun
            best_result = result
    
    # If we still don't have a good result, do a fallback optimization
    if best_result is None:
        # Use default initial guess
        default_config = [
            (0, 0, 0),
            (1.732, 0, 0),
            (-1.732, 0, 0),
            (0.866, 1.5, 0),
            (-0.866, 1.5, 0),
            (0.866, -1.5, 0),
            (-0.866, -1.5, 0),
            (2.598, 1.5, 0),
            (-2.598, 1.5, 0),
            (2.598, -1.5, 0),
            (-2.598, -1.5, 0)
        ]
        
        initial_guess = create_initial_guess(default_config)
        
        result = differential_evolution(
            optimizer.evaluate_solution,
            bounds,
            maxiter=150,
            popsize=20,
            seed=42,
            callback=lambda x, convergence: None,  # Suppress output
            disp=False,
            mutation=(0.5, 1.0)
        )
        best_result = result
    
    # Extract results
    best_params = best_result.x
    inner_positions_angles = best_params[:-1].reshape(-1, 3)
    outer_radius = best_params[-1]

    # Convert to desired output format
    inner_hex_data = inner_positions_angles.copy()

    # Create outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin

    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END