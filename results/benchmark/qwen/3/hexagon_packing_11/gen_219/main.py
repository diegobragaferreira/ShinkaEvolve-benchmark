# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from scipy.optimize import differential_evolution, minimize
import time
from typing import Tuple, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Hexagon:
    """Represents a regular hexagon with center, rotation and side length"""
    
    def __init__(self, center_x: float, center_y: float, angle_degrees: float, side_length: float = 1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.angle_degrees = angle_degrees
        self.side_length = side_length
        
    def get_vertices(self) -> np.ndarray:
        """Get vertices of the hexagon with current transformation"""
        # Precomputed values for efficiency
        sqrt3 = math.sqrt(3)
        side = self.side_length
        
        # Base vertices of a unit hexagon centered at origin
        base_vertices = np.array([
            [side, 0.0],
            [side/2.0, sqrt3/2.0 * side],
            [-side/2.0, sqrt3/2.0 * side],
            [-side, 0.0],
            [-side/2.0, -sqrt3/2.0 * side],
            [side/2.0, -sqrt3/2.0 * side]
        ])
        
        # Apply rotation
        angle_rad = math.radians(self.angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_vertices = base_vertices @ rotation_matrix.T
        
        # Apply translation
        return rotated_vertices + np.array([self.center_x, self.center_y])
        
    def to_polygon(self) -> Polygon:
        """Convert hexagon to shapely polygon"""
        return Polygon(self.get_vertices())

class HexagonEvaluator:
    """Handles geometric validation and fitness evaluation"""
    
    def __init__(self, hex_side_length: float = 1.0):
        self.hex_side_length = hex_side_length
        
    def check_containment(self, hexagons: List[Hexagon], outer_radius: float) -> bool:
        """Check if all hexagons are contained within outer hexagon of given radius"""
        # Create outer hexagon centered at origin
        outer_hex = Hexagon(0.0, 0.0, 0.0, outer_radius)
        outer_polygon = outer_hex.to_polygon()
        
        for hexagon in hexagons:
            hex_polygon = hexagon.to_polygon()
            if not outer_polygon.contains(hex_polygon):
                return False
        return True
        
    def check_overlap(self, hexagons: List[Hexagon]) -> bool:
        """Check if any hexagons overlap"""
        # Convert to polygons once
        polygons = [h.to_polygon() for h in hexagons]
        
        # Check pairwise overlaps efficiently with early exit
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if polygons[i].intersects(polygons[j]):
                    return True
        return False
        
    def evaluate_fitness(self, hexagons: List[Hexagon], outer_radius: float) -> float:
        """Evaluate fitness based on geometric constraints and packing density"""
        # Check constraints
        if not self.check_containment(hexagons, outer_radius):
            return -float('inf')  # Invalid - penalty
            
        if self.check_overlap(hexagons):
            return -float('inf')  # Invalid - penalty
            
        # Valid configuration - maximize 1/outer_radius (minimize outer_radius)
        return 1.0 / outer_radius

class HexagonPackingOptimizer:
    """Main optimization class for hexagon packing"""
    
    def __init__(self, n_inner_hexagons: int = 11, hex_side_length: float = 1.0):
        self.n_inner_hexagons = n_inner_hexagons
        self.hex_side_length = hex_side_length
        self.evaluator = HexagonEvaluator(hex_side_length)
        self._cached_results = {}
        
    def create_hexagons_from_array(self, hex_data: np.ndarray) -> List[Hexagon]:
        """Convert array data to list of Hexagon objects"""
        return [Hexagon(row[0], row[1], row[2], self.hex_side_length) for row in hex_data]
        
    def create_array_from_hexagons(self, hexagons: List[Hexagon]) -> np.ndarray:
        """Convert list of Hexagon objects to array data"""
        return np.array([[h.center_x, h.center_y, h.angle_degrees] for h in hexagons])
        
    def find_optimal_radius(self, hexagons: List[Hexagon], min_radius: float = 1.0, max_radius: float = 20.0) -> float:
        """Find minimum radius that contains all hexagons using optimized binary search"""
        if not hexagons:
            return min_radius
            
        # Cache key for memoization
        cache_key = tuple((h.center_x, h.center_y, h.angle_degrees) for h in hexagons)
        
        if cache_key in self._cached_results:
            return self._cached_results[cache_key]
            
        # Quick check for containment to avoid expensive binary search
        quick_check_radius = min_radius
        temp_outer_hex = Hexagon(0.0, 0.0, 0.0, quick_check_radius)
        temp_outer_polygon = temp_outer_hex.to_polygon()
        
        # If we can't contain all hexagons even with minimum radius, 
        # compute a reasonable starting point
        all_contained = all(temp_outer_polygon.contains(h.to_polygon()) for h in hexagons)
        if all_contained:
            self._cached_results[cache_key] = quick_check_radius
            return quick_check_radius
            
        # More precise approach: compute a good starting estimate
        # Get all vertices of all hexagons
        all_vertices = []
        for hexagon in hexagons:
            vertices = hexagon.get_vertices()
            all_vertices.extend(vertices)
            
        if not all_vertices:
            self._cached_results[cache_key] = min_radius
            return min_radius
            
        # Find center of bounding box
        xs = [p[0] for p in all_vertices]
        ys = [p[1] for p in all_vertices]
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        
        # Compute max distance from center to any vertex
        max_dist = 0
        for x, y in all_vertices:
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = max(max_dist, dist)
            
        # Set reasonable bounds
        low = max_dist
        high = max(low * 2, max_radius)
        
        # Optimize binary search with adaptive tolerance
        tolerance = 1e-4
        max_iterations = 25
        prev_diff = float('inf')
        tolerance_factor = 0.9
        
        for iteration in range(max_iterations):
            # Adaptive tolerance adjustment
            current_diff = high - low
            if iteration > 0 and current_diff < prev_diff * tolerance_factor:
                tolerance = max(1e-8, tolerance * 0.8)
            elif current_diff < 1e-3:
                tolerance = max(1e-8, tolerance * 0.5)
                
            prev_diff = current_diff
            
            if current_diff < tolerance:
                break
                
            mid = (low + high) / 2
            # Create test outer hexagon
            test_outer_hex = Hexagon(0, 0, mid, 0)
            test_outer_polygon = test_outer_hex.to_polygon()
            
            # Check if all inner hexagons fit
            all_contained = True
            for hexagon in hexagons:
                if not test_outer_polygon.contains(hexagon.to_polygon()):
                    all_contained = False
                    break
                    
            if all_contained:
                high = mid
            else:
                low = mid
                
        result = (low + high) / 2 + 0.01  # Add small padding
        self._cached_results[cache_key] = result
        return result
        
    def generate_initial_patterns(self) -> List[np.ndarray]:
        """Generate multiple initial pattern configurations"""
        patterns = []
        
        # Pattern 1: Standard hexagonal packing
        pattern1 = np.array([
            [0, 0, 0],          # center
            [-2.0, 0, 0],       # left
            [2.0, 0, 0],        # right
            [0, 2.0, 0],        # top
            [0, -2.0, 0],       # bottom
            [-1.0, 1.0, 0],     # top-left
            [1.0, 1.0, 0],      # top-right
            [-1.0, -1.0, 0],    # bottom-left
            [1.0, -1.0, 0],     # bottom-right
            [-2.0, 1.0, 0],     # far top-left
            [2.0, 1.0, 0],      # far top-right
        ])
        patterns.append(pattern1)
        
        # Pattern 2: More spread-out arrangement
        pattern2 = np.array([
            [0, 0, 0],          # center
            [-2.5, 0, 0],       # left
            [2.5, 0, 0],        # right
            [0, 2.5, 0],        # top
            [0, -2.5, 0],       # bottom
            [-1.5, 1.5, 0],     # top-left
            [1.5, 1.5, 0],      # top-right
            [-1.5, -1.5, 0],    # bottom-left
            [1.5, -1.5, 0],     # bottom-right
            [-2.5, 1.5, 0],     # far top-left
            [2.5, 1.5, 0],      # far top-right
        ])
        patterns.append(pattern2)
        
        # Pattern 3: Ring arrangement with diagonals
        pattern3 = np.array([
            [0, 0, 0],          # center
            [-2.0, 0, 0],       # left
            [2.0, 0, 0],        # right
            [0, 2.0, 0],        # top
            [0, -2.0, 0],       # bottom
            [-1.0, 1.732, 0],   # top-left
            [1.0, 1.732, 0],    # top-right
            [-1.0, -1.732, 0],  # bottom-left
            [1.0, -1.732, 0],   # bottom-right
            [-2.0, 1.0, 0],     # far top-left
            [2.0, 1.0, 0],      # far top-right
        ])
        patterns.append(pattern3)
        
        return patterns
        
    def optimize_local(self, individual: np.ndarray, outer_radius: float) -> np.ndarray:
        """Refine solution locally using optimization"""
        def objective(params):
            # Reshape params back to hexagon data
            new_data = individual.copy()
            for i in range(len(new_data)):
                new_data[i][0] = params[i*3]
                new_data[i][1] = params[i*3+1]
                new_data[i][2] = params[i*3+2]
                
            # Convert to hexagon objects for evaluation
            hexagons = self.create_hexagons_from_array(new_data)
            
            # Evaluate fitness
            fitness = self.evaluator.evaluate_fitness(hexagons, outer_radius)
            return -fitness  # minimize negative fitness
            
        # Flatten the data for optimization
        initial_params = individual.flatten()
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B',
                            bounds=[(-10, 10), (-10, 10), (0, 360)] * len(individual),
                            options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-9})
            if result.success:
                # Reshape optimized result back
                refined_data = individual.copy()
                for i in range(len(refined_data)):
                    refined_data[i][0] = result.x[i*3]
                    refined_data[i][1] = result.x[i*3+1]
                    refined_data[i][2] = result.x[i*3+2]
                return refined_data
        except Exception as e:
            logger.warning(f"Local optimization failed: {e}")
        return individual
        
    def optimize_with_staged_approach(self, initial_config: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize using a staged approach with progressive refinement"""
        best_fitness = -float('inf')
        best_config = initial_config.copy()
        best_radius = 10.0
        
        # Stage 1: Coarse global optimization with differential evolution
        logger.info("Starting coarse global optimization...")
        bounds = [(-8, 8), (-8, 8), (0, 360)] * self.n_inner_hexagons
        
        def objective_global(params):
            positions_angles = []
            for i in range(self.n_inner_hexagons):
                x = params[i*3]
                y = params[i*3 + 1]
                angle = params[i*3 + 2]
                positions_angles.append([x, y, angle])
            score, _ = evaluate_layout_internal(positions_angles)
            return score
            
        try:
            result = differential_evolution(
                objective_global,
                bounds,
                maxiter=30,
                popsize=20,
                seed=42,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                disp=False
            )
            
            # Extract best solution from global optimization
            best_params = result.x
            final_positions_angles = []
            for i in range(self.n_inner_hexagons):
                x = best_params[i*3]
                y = best_params[i*3 + 1]
                angle = best_params[i*3 + 2]
                final_positions_angles.append([x, y, angle])
                
            score, side_length = evaluate_layout_internal(final_positions_angles)
            
            if score < best_fitness and side_length > 0:
                best_fitness = score
                best_config = np.array(final_positions_angles)
                best_radius = side_length
                
        except Exception as e:
            logger.warning(f"Global optimization failed: {e}")
            
        # Stage 2: Fine-grained local optimization  
        logger.info("Starting fine-grained local optimization...")
        
        if best_fitness != -float('inf'):
            # Apply local refinement
            refined_config = self.optimize_local(best_config, best_radius)
            
            # Re-evaluate after local refinement
            score, side_length = evaluate_layout_internal(refined_config)
            
            if score < best_fitness and side_length > 0:
                best_fitness = score
                best_config = refined_config
                best_radius = side_length
                
        # Stage 3: Multi-start refinement to escape local minima
        logger.info("Running multi-start optimization...")
        
        # Generate additional starting configurations
        additional_configs = self.generate_initial_patterns()
        
        for i, candidate_config in enumerate(additional_configs):
            if i >= 3:  # Only use first 3 patterns
                break
                
            # Perturb the candidate slightly
            np.random.seed(42 + i)
            perturbed_config = candidate_config.copy()
            for j in range(self.n_inner_hexagons):
                perturbed_config[j][0] += np.random.normal(0, 0.2)
                perturbed_config[j][1] += np.random.normal(0, 0.2)
                perturbed_config[j][2] = perturbed_config[j][2] % 360
                
            try:
                # Local refinement
                refined_config = self.optimize_local(perturbed_config, best_radius)
                
                # Re-evaluate
                score, side_length = evaluate_layout_internal(refined_config)
                
                if score < best_fitness and side_length > 0:
                    best_fitness = score
                    best_config = refined_config
                    best_radius = side_length
                    
            except Exception as e:
                logger.warning(f"Multi-start refinement {i} failed: {e}")
                
        return best_config, best_radius

def evaluate_layout_internal(inner_positions_angles):
    """Internal evaluation function without the outer scope dependency"""
    # Create hexagons
    hexagons = []
    for pos_angle in inner_positions_angles:
        x, y, angle = pos_angle
        hexagon = Hexagon(x, y, angle, 1.0)
        hexagons.append(hexagon)
        
    # Create outer hexagon with current radius
    # Using a simpler approach to compute radius first
    all_vertices = []
    for hexagon in hexagons:
        vertices = hexagon.get_vertices()
        all_vertices.extend(vertices)
        
    if not all_vertices:
        return -float('inf'), 10.0
        
    # Find center of bounding box
    xs = [p[0] for p in all_vertices]
    ys = [p[1] for p in all_vertices]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    
    # Compute max distance from center to any vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)
        
    outer_radius = max_dist + 0.01  # Add padding
    
    # Validate constraints
    evaluator = HexagonEvaluator(1.0)
    outer_hex = Hexagon(0, 0, 0, outer_radius)
    outer_polygon = outer_hex.to_polygon()
    
    # Check containment
    for hexagon in hexagons:
        if not outer_polygon.contains(hexagon.to_polygon()):
            return -float('inf'), 0.0
            
    # Check overlap
    polygons = [h.to_polygon() for h in hexagons]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                return -float('inf'), 0.0
                
    # Valid configuration - maximize 1/outer_radius (minimize outer_radius)
    inv_radius = 1.0 / outer_radius
    return -inv_radius, outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize optimizer
    optimizer = HexagonPackingOptimizer(n_inner_hexagons=11, hex_side_length=1.0)
    
    # Generate several initial configurations to try multiple patterns
    initial_patterns = optimizer.generate_initial_patterns()
    best_score = float('inf')
    best_config = None
    best_radius = 10.0
    
    # Try each pattern and see which works better
    for i, pattern in enumerate(initial_patterns):
        logger.info(f"Trying initial pattern {i+1}")
        
        try:
            # Direct optimization of this pattern
            refined_config, radius = optimizer.optimize_with_staged_approach(pattern)
            
            # Evaluate the refined result
            score, side_length = evaluate_layout_internal(refined_config)
            
            if score < best_score and side_length > 0:
                best_score = score
                best_config = refined_config.copy()
                best_radius = side_length
                
        except Exception as e:
            logger.warning(f"Pattern {i+1} failed: {e}")
            continue
    
    # If no pattern worked, use the first pattern as fallback
    if best_config is None:
        best_config = initial_patterns[0].copy()
        _, best_radius = evaluate_layout_internal(best_config)
        
    # Perform final validation and ensure we have the correct data structures
    inner_hex_data = best_config if best_config is not None else initial_patterns[0]
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = best_radius if 'best_radius' in locals() else 8.0
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Validate result
    try:
        hexagons = optimizer.create_hexagons_from_array(inner_hex_data)
        if not optimizer.evaluator.checkOverlap(hexagons):
            logger.info("No overlaps detected in final configuration")
        else:
            logger.warning("Overlapping hexagons detected!")
    except Exception as e:
        logger.error(f"Validation error: {e}")
        
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END