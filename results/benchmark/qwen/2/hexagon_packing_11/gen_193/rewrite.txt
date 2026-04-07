# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import math
from collections import deque

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2

class HexagonTilingOptimizer:
    """Hierarchical geometric optimization for hexagon packing using constraint satisfaction"""
    
    def __init__(self):
        self.n_inner = 11
        self.hex_radius = UNIT_HEX_RADIUS
        self.hex_apogee = UNIT_HEX_APOGEE
        
    @staticmethod
    def create_unit_hexagon(center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon"""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            points.append((x, y))
        return Polygon(points)

    @staticmethod
    def get_hexagon_vertices(center, rotation, radius=UNIT_HEX_RADIUS):
        """Get vertices of a hexagon"""
        angle_offset = np.deg2rad(rotation)
        vertices = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            vertices.append((x, y))
        return vertices

    @staticmethod
    def is_contained(hexagon, outer_hexagon):
        """Check if hexagon is fully contained in outer hexagon"""
        # Buffer for floating point precision
        buffered_hex = hexagon.buffer(-1e-12)
        return outer_hexagon.contains(buffered_hex)

    @staticmethod
    def do_intersect(hex1, hex2):
        """Check if two hexagons intersect"""
        # Small buffer for precision
        buffered_hex1 = hex1.buffer(1e-12)
        buffered_hex2 = hex2.buffer(1e-12)
        return buffered_hex1.intersects(buffered_hex2)

    def calculate_bounding_radius(self, positions_and_angles):
        """Calculate tightest bounding circle for all hexagons"""
        all_vertices = []
        for i in range(self.n_inner):
            x, y, angle = positions_and_angles[3*i:3*i+3]
            vertices = self.get_hexagon_vertices((x, y), angle)
            all_vertices.extend(vertices)

        if not all_vertices:
            return 1.0
            
        # Find centroid
        coords = np.array(all_vertices)
        centroid = np.mean(coords, axis=0)
        
        # Calculate max distance from centroid
        distances = np.sqrt(np.sum((coords - centroid)**2, axis=1))
        max_distance = np.max(distances)
        
        return max_distance + 1e-6

    def build_tile_pattern(self):
        """Build a geometrically sound initial pattern"""
        # Start with central hexagon
        positions = [(0.0, 0.0)]
        
        # Place 6 surrounding hexagons (first ring)
        for i in range(6):
            angle = i * np.pi/3
            radius = 2.0  # Slightly more than diameter to allow for gaps
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))
            
        # Place 4 more in second ring
        for i in range(4):
            angle = i * np.pi/2
            radius = 3.5
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))
            
        # Take first 11 positions and add rotation angles
        final_positions = []
        for i, (x, y) in enumerate(positions[:11]):
            final_positions.extend([x, y, np.random.uniform(0, 360)])
            
        return np.array(final_positions)

    def validate_configuration(self, positions_and_angles):
        """Validate complete configuration with efficient checks"""
        # Create all hexagons
        hexagons = []
        for i in range(self.n_inner):
            x, y, angle = positions_and_angles[3*i:3*i+3]
            hexagon = self.create_unit_hexagon((x, y), angle)
            hexagons.append(hexagon)
            
        # Create outer hexagon
        bounding_radius = self.calculate_bounding_radius(positions_and_angles)
        outer_hex = self.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hex.exterior.coords)
        scaled_coords = [(x*bounding_radius, y*bounding_radius) for x, y in outer_coords]
        outer_hex_scaled = Polygon(scaled_coords)
        
        # Check containment
        for hexagon in hexagons:
            if not self.is_contained(hexagon, outer_hex_scaled):
                return False, False, 0.0
                
        # Check overlaps (only necessary pairs)
        for i in range(self.n_inner):
            for j in range(i+1, self.n_inner):
                if self.do_intersect(hexagons[i], hexagons[j]):
                    return False, False, 0.0
                    
        # Return valid configuration
        tight_radius = self.calculate_bounding_radius(positions_and_angles)
        return True, True, 1.0 / tight_radius

    def construct_geometrically_optimal_packing(self):
        """Construct a geometrically driven packing using hierarchical rules"""
        # Phase 1: Build basic structure with known good patterns
        base_positions = self.build_tile_pattern()
        
        # Phase 2: Apply geometric constraints with iterative refinement
        best_config = base_positions.copy()
        best_score = 0.0
        
        # Geometric optimization loop - focus on improving specific arrangements
        for iteration in range(50):
            # Perturb and test
            test_positions = base_positions.copy()
            
            # Randomly modify some positions and angles
            for i in range(11):
                if np.random.random() < 0.3:  # 30% chance to modify
                    idx = 3 * i
                    # Small random perturbations
                    test_positions[idx] += np.random.normal(0, 0.05)  # x
                    test_positions[idx+1] += np.random.normal(0, 0.05)  # y
                    test_positions[idx+2] += np.random.normal(0, 10)  # angle
                    
            # Validate and score
            valid, _, score = self.validate_configuration(test_positions)
            if valid and score > best_score:
                best_score = score
                best_config = test_positions.copy()
                
        return best_config

    def optimize_structure(self):
        """Main optimization method using geometric constraints"""
        # Try multiple geometric strategies
        strategies = []
        for _ in range(10):
            # Different initial configurations
            strategy = self.construct_geometrically_optimal_packing()
            strategies.append(strategy)
            
        # Select best strategy
        best_strategy = None
        best_score = 0.0
        
        for strategy in strategies:
            valid, _, score = self.validate_configuration(strategy)
            if valid and score > best_score:
                best_score = score
                best_strategy = strategy
                
        return best_strategy if best_strategy is not None else self.build_tile_pattern()

    def refine_with_constraint_propagation(self, initial_positions):
        """Refine solution using constraint propagation principles"""
        # Convert to geometric representation
        positions = []
        for i in range(self.n_inner):
            positions.append((initial_positions[3*i], initial_positions[3*i+1], initial_positions[3*i+2]))
            
        # Apply geometric constraints iteratively
        for _ in range(20):  # Limited refinement iterations
            improved = False
            
            # Try to move each hexagon to reduce conflicts
            for i in range(self.n_inner):
                x, y, angle = positions[i]
                original_pos = (x, y, angle)
                
                # Try small adjustments
                for dx in [-0.1, -0.05, 0, 0.05, 0.1]:
                    for dy in [-0.1, -0.05, 0, 0.05, 0.1]:
                        # Test nearby positions
                        new_x, new_y = x + dx, y + dy
                        test_positions = positions[:]
                        test_positions[i] = (new_x, new_y, angle)
                        
                        # Create test configuration
                        test_config = []
                        for pos in test_positions:
                            test_config.extend([pos[0], pos[1], pos[2]])
                            
                        valid, _, score = self.validate_configuration(np.array(test_config))
                        if valid and score > 1e-6:
                            # Accept improvement
                            positions[i] = (new_x, new_y, angle)
                            improved = True
                            break
                    if improved:
                        break
                        
        # Convert back to flat array
        final_config = []
        for pos in positions:
            final_config.extend([pos[0], pos[1], pos[2]])
            
        return np.array(final_config)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses geometric constraint satisfaction instead of evolutionary optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        optimizer = HexagonTilingOptimizer()
        
        # Use geometric construction approach
        best_config = optimizer.optimize_structure()
        
        # Further refine with constraint propagation
        refined_config = optimizer.refine_with_constraint_propagation(best_config)
        
        # Validate final configuration
        valid, _, score = optimizer.validate_configuration(refined_config)
        
        if valid and score > 1e-6:
            # Format output
            inner_hex_data = np.zeros((11, 3))
            for i in range(11):
                inner_hex_data[i] = refined_config[3*i:3*i+3]
                
            outer_hex_data = np.array([0, 0, 0])
            outer_radius = 1.0 / score
            
            return inner_hex_data, outer_hex_data, outer_radius

    except Exception as e:
        warnings.warn(f"Error in geometric optimization: {str(e)}")
        pass

    # Fallback to simple configuration
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