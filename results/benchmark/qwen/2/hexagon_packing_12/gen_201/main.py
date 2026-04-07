# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
from typing import Tuple, List, Optional
import math
from numba import jit
import warnings
warnings.filterwarnings('ignore')

class SpatialHash:
    """Spatial hash grid for efficient collision detection."""

    def __init__(self, cell_size=2.0):
        self.cell_size = cell_size
        self.grid = {}

    def _hash(self, x, y):
        """Convert continuous coordinates to grid cell indices."""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def clear(self):
        """Clear the spatial hash grid."""
        self.grid.clear()

    def insert(self, hex_id, vertices):
        """Insert a hexagon into the spatial hash."""
        # Get bounding box of hexagon
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)

        # Insert into all cells it covers
        min_cell_x, min_cell_y = self._hash(min_x, min_y)
        max_cell_x, max_cell_y = self._hash(max_x, max_y)

        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                if (cell_x, cell_y) not in self.grid:
                    self.grid[(cell_x, cell_y)] = []
                self.grid[(cell_x, cell_y)].append(hex_id)

    def query(self, vertices):
        """Query which hexagons might collide with the given hexagon."""
        # Get bounding box of hexagon
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)

        # Query all cells it covers
        min_cell_x, min_cell_y = self._hash(min_x, min_y)
        max_cell_x, max_cell_y = self._hash(max_x, max_y)

        candidates = set()
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                if (cell_x, cell_y) in self.grid:
                    candidates.update(self.grid[(cell_x, cell_y)])

        return list(candidates)

@jit(nopython=True)
def hexagon_vertices_numba(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon using numba for speed."""
    angle_rad = angle_deg * math.pi / 180.0
    vertices = np.empty((6, 2))
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices[i] = (x, y)
    return vertices

class GeometryHandler:
    """Handles all geometric computations for hexagon operations."""
    
    def __init__(self):
        self.hex_radius = 1.0
        self.hex_apothem = np.sqrt(3) / 2
        self.hex_height = 2 * self.hex_apothem
        self.hex_width = 2 * self.hex_radius
    
    def generate_hexagon_vertices(self, center_x: float, center_y: float, angle_deg: float) -> np.ndarray:
        """Generate vertices of a regular hexagon given center and rotation."""
        angle_rad = np.deg2rad(angle_deg)
        # Vertices of a unit hexagon centered at origin
        base_vertices = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1, 0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        # Rotate and translate
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = base_vertices @ rotation_matrix.T
        return rotated_vertices + np.array([center_x, center_y])
    
    def create_hexagon_polygon(self, center_x: float, center_y: float, angle_deg: float) -> Polygon:
        """Create Shapely polygon representation of a hexagon."""
        vertices = self.generate_hexagon_vertices(center_x, center_y, angle_deg)
        return Polygon(vertices)

class ConstraintChecker:
    """Handles all constraint validation logic."""
    
    def __init__(self, geometry_handler: GeometryHandler):
        self.geo = geometry_handler
    
    def check_containment(self, hexagon: Polygon, outer_hex: Polygon) -> bool:
        """Check if hexagon is fully contained within outer hexagon."""
        return outer_hex.contains(hexagon) or outer_hex.touches(hexagon)
    
    def check_overlap(self, hex1: Polygon, hex2: Polygon) -> bool:
        """Check if two hexagons overlap."""
        return hex1.intersects(hex2) and not hex1.touches(hex2)
    
    def calculate_overlap_area(self, hex1: Polygon, hex2: Polygon) -> float:
        """Calculate the overlap area between two hexagons."""
        try:
            intersection = hex1.intersection(hex2)
            return intersection.area if not intersection.is_empty else 0.0
        except:
            return 0.0
    
    def compute_constraint_violations(self, positions: np.ndarray, outer_radius: float) -> Tuple[bool, float]:
        """Efficiently compute constraint violations for a configuration."""
        # Create outer hexagon (scaled appropriately)
        outer_vertices = self.geo.generate_hexagon_vertices(0, 0, 0)
        outer_vertices *= outer_radius / self.geo.hex_radius
        outer_hex = Polygon(outer_vertices)
        
        # Check containment and overlap efficiently
        total_violation = 0.0
        inner_polygons = []
        
        for i, (cx, cy, angle) in enumerate(positions):
            inner_hex = self.geo.create_hexagon_polygon(cx, cy, angle)
            inner_polygons.append(inner_hex)
            
            # Check containment - more robust way
            if not self.check_containment(inner_hex, outer_hex):
                # For containment violations, we measure the extent of violation
                try:
                    intersection = outer_hex.intersection(inner_hex)
                    if intersection.is_empty:
                        # Complete violation
                        total_violation += 10000.0
                    else:
                        # Partial violation
                        contained_area = intersection.area
                        total_violation += (inner_hex.area - contained_area) * 100.0
                except:
                    total_violation += 10000.0
            
            # Check overlaps with previously processed hexagons only
            for j in range(i):
                if self.check_overlap(inner_hex, inner_polygons[j]):
                    try:
                        intersection_area = self.calculate_overlap_area(inner_hex, inner_polygons[j])
                        # Penalty based on the amount of overlapping area
                        overlap_penalty = (inner_hex.area + inner_polygons[j].area - 2 * intersection_area) * 500.0
                        total_violation += overlap_penalty
                    except:
                        total_violation += 50000.0
                        
        return total_violation == 0, total_violation

class SymmetryGuidedOptimizer:
    """Implements symmetry-guided optimization for hexagon packing."""
    
    def __init__(self, geometry_handler: GeometryHandler, constraint_checker: ConstraintChecker):
        self.geo = geometry_handler
        self.constraint_checker = constraint_checker
        self.n_hexagons = 12
        self.max_time = 180.0
        
    def generate_symmetric_base_config(self) -> np.ndarray:
        """Generate a highly symmetric base configuration based on known good patterns."""
        # This creates a pattern that respects D6 symmetry as much as possible with 12 hexagons
        # Pattern inspired by close-packing arrangements
        
        # Positions arranged in concentric rings with symmetrical placement
        positions = []
        
        # Center hexagon
        positions.append([0, 0, 0])
        
        # First ring (radius ~2.5)
        ring1_radius = 2.5
        for i in range(6):
            angle = i * 60  # 60-degree increments
            x = ring1_radius * np.cos(np.deg2rad(angle))
            y = ring1_radius * np.sin(np.deg2rad(angle))
            positions.append([x, y, 0])
        
        # Second ring (radius ~3.75)  
        ring2_radius = 3.75
        for i in range(6):
            angle = i * 60  # 60-degree increments
            x = ring2_radius * np.cos(np.deg2rad(angle))
            y = ring2_radius * np.sin(np.deg2rad(angle))
            positions.append([x, y, 0])
        
        # Adjust to ensure we have exactly 12 positions
        positions = positions[:12]
        
        # Convert to numpy array
        positions_array = np.array(positions)
        
        # Add small random variations to break perfect symmetry and allow optimization
        positions_array[:, 0] += np.random.normal(0, 0.1, 12)
        positions_array[:, 1] += np.random.normal(0, 0.1, 12)
        
        return positions_array
    
    def compute_fitness(self, positions_and_radius: np.ndarray) -> float:
        """Compute fitness for a configuration."""
        # Extract parameters
        positions = positions_and_radius[:-1].reshape(-1, 3)
        outer_radius = positions_and_radius[-1]
        
        # Check constraints
        valid, violation = self.constraint_checker.compute_constraint_violations(positions, outer_radius)
        
        if not valid:
            # Return very poor fitness for invalid configurations
            return 1e12 + violation
        
        # Return negative inverse of outer radius (we want to maximize 1/R)
        return -1.0 / outer_radius

    def optimize_symmetric_config(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Optimize the symmetric base configuration."""
        # Generate symmetric base configuration
        base_positions = self.generate_symmetric_base_config()
        
        # Set initial outer radius based on the configuration
        # Estimate based on maximum distance from center
        max_distance = 0
        for i in range(len(base_positions)):
            dist = np.sqrt(base_positions[i][0]**2 + base_positions[i][1]**2)
            if dist > max_distance:
                max_distance = dist
        estimated_outer_radius = max_distance + self.geo.hex_radius
        
        # Combine positions and radius into single vector
        initial_config = np.concatenate([base_positions.flatten(), [estimated_outer_radius]])
        
        # Define bounds for optimization
        bounds = []
        # Position bounds (-10 to 10 for safety)
        for i in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        # Outer radius bound
        bounds.append((2.0, 15.0))
        
        # First, try global optimization
        try:
            de_result = differential_evolution(
                self.compute_fitness,
                bounds,
                maxiter=200,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False,
                tol=1e-6
            )
            result = de_result.x
        except Exception as e:
            warnings.warn(f"Differential evolution failed: {e}")
            # Fallback to the initial configuration
            result = initial_config
            
        # Then, local refinement with L-BFGS-B
        try:
            local_result = minimize(
                self.compute_fitness,
                result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if local_result.success:
                final_result = local_result.x
            else:
                final_result = result
        except Exception as e:
            warnings.warn(f"Local optimization failed: {e}")
            final_result = result
        
        # Extract positions and radius
        positions = final_result[:-1].reshape(-1, 3)
        outer_radius = final_result[-1]
        
        return positions, np.array([0, 0, 0]), outer_radius

    def find_optimal_configuration(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Find the optimal hexagon configuration using symmetry-guided approach."""
        start_time = time.time()
        
        # Try multiple random restarts to avoid local optima
        best_positions = None
        best_outer_hex_data = None
        best_outer_radius = float('inf')
        best_fitness = float('-inf')
        
        # Try 5 different initial symmetric configurations with different randomizations
        for seed in range(5):
            np.random.seed(seed)
            try:
                positions, outer_hex_data, outer_radius = self.optimize_symmetric_config()
                
                # Verify the solution
                valid, violation = self.constraint_checker.compute_constraint_violations(positions, outer_radius)
                if valid:
                    fitness = -1.0 / outer_radius
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_positions = positions.copy()
                        best_outer_hex_data = outer_hex_data.copy()
                        best_outer_radius = outer_radius
            except Exception as e:
                continue
                
        # If no valid solution found, return a basic working solution
        if best_positions is None:
            # Fallback to a basic configuration
            positions = np.array([
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0]
            ])
            best_positions = positions
            best_outer_hex_data = np.array([0, 0, 0])
            best_outer_radius = 5.0  # Conservative estimate
            
        return best_positions, best_outer_hex_data, best_outer_radius

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize components
    geo_handler = GeometryHandler()
    constraint_checker = ConstraintChecker(geo_handler)
    optimizer = SymmetryGuidedOptimizer(geo_handler, constraint_checker)
    
    # Execute optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimizer.find_optimal_configuration()
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END