# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import math
import time
from scipy.spatial.distance import cdist

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
        """Build a geometrically sound initial pattern using hexagonal tiling principles"""
        # Create a pattern based on three concentric rings
        positions = []
        
        # Central hexagon
        positions.append((0.0, 0.0))
        
        # First ring: 6 hexagons around center
        for i in range(6):
            angle = i * np.pi/3
            radius = 2.0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))
            
        # Second ring: 4 hexagons arranged in diamond formation
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

    def build_alternative_patterns(self, count=5):
        """Generate alternative geometric patterns for diversity"""
        patterns = []
        
        # Pattern 1: Star-like arrangement
        star_pattern = []
        star_pattern.append((0.0, 0.0))  # center
        # Arrange around center like a star
        for i in range(5):
            angle = i * 2*np.pi/5
            radius = 2.0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            star_pattern.append((x, y))
        # Add more points to reach 11
        for i in range(6):
            angle = i * np.pi/3 + np.pi/6  # Offset for variety
            radius = 3.0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            star_pattern.append((x, y))
            
        # Pattern 2: Linear arrangement with branching
        linear_pattern = []
        # Main line
        for i in range(7):
            linear_pattern.append((i * 1.8, 0.0))
        # Branches
        for i in range(4):
            linear_pattern.append((-1.0, 1.5 + i*0.8))
            
        # Pattern 3: Hexagonal lattice pattern
        lattice_pattern = []
        # Create a hexagonal pattern
        for i in range(3):
            for j in range(3):
                x = i * 2.0 + (j % 2) * 1.0
                y = j * 1.732  # sqrt(3)
                lattice_pattern.append((x, y))
        
        patterns.append(star_pattern[:11])
        patterns.append(linear_pattern[:11])  
        patterns.append(lattice_pattern[:11])
        
        # Convert to proper format
        formatted_patterns = []
        for pattern in patterns:
            if len(pattern) >= 11:
                formatted = []
                for i, (x, y) in enumerate(pattern[:11]):
                    formatted.extend([x, y, np.random.uniform(0, 360)])
                formatted_patterns.append(np.array(formatted))
                
        return formatted_patterns

    def validate_configuration(self, positions_and_angles):
        """Validate complete configuration with efficient checks"""
        # Create all hexagons
        hexagons = []
        for i in range(self.n_inner):
            x, y, angle = positions_and_angles[3*i:3*i+3]
            hexagon = self.create_unit_hexagon((x, y), angle)
            hexagons.append(hexagon)
            
        # Create outer hexagon - using tight bounding radius
        bounding_radius = self.calculate_bounding_radius(positions_and_angles)
        outer_hex = self.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hex.exterior.coords)
        scaled_coords = [(x*bounding_radius, y*bounding_radius) for x, y in outer_coords]
        outer_hex_scaled = Polygon(scaled_coords)
        
        # Check containment - early termination
        for hexagon in hexagons:
            if not self.is_contained(hexagon, outer_hex_scaled):
                return False, False, 0.0
                
        # Check overlaps - early termination with optimized pair checking
        for i in range(self.n_inner):
            for j in range(i+1, self.n_inner):
                if self.do_intersect(hexagons[i], hexagons[j]):
                    return False, False, 0.0
                    
        # Return valid configuration
        tight_radius = self.calculate_bounding_radius(positions_and_angles)
        return True, True, 1.0 / tight_radius

    def construct_geometrically_optimal_packing(self, base_positions=None):
        """Construct a geometrically driven packing using hierarchical refinement"""
        if base_positions is None:
            base_positions = self.build_tile_pattern()
        
        # Phase 1: Global pattern refinement
        best_config = base_positions.copy()
        best_score = 0.0
        
        # Try several perturbation iterations
        for iteration in range(30):
            # Perturb positions with strategic changes
            test_positions = base_positions.copy()
            
            # Apply selective modifications (focus on boundary positions)
            for i in range(11):
                if np.random.random() < 0.4:  # 40% chance to modify
                    idx = 3 * i
                    # Modify positions with varying magnitudes
                    if i == 0:  # Center hexagon - small changes
                        test_positions[idx] += np.random.normal(0, 0.03)
                        test_positions[idx+1] += np.random.normal(0, 0.03)
                    elif i < 7:  # Outer ring - medium changes
                        test_positions[idx] += np.random.normal(0, 0.08)
                        test_positions[idx+1] += np.random.normal(0, 0.08)
                    else:  # Second ring - larger changes
                        test_positions[idx] += np.random.normal(0, 0.12)
                        test_positions[idx+1] += np.random.normal(0, 0.12)
                    
                    # Angle changes
                    test_positions[idx+2] += np.random.normal(0, 15)
                    
            # Validate and score
            valid, _, score = self.validate_configuration(test_positions)
            if valid and score > best_score:
                best_score = score
                best_config = test_positions.copy()
                
        return best_config

    def optimize_structure(self):
        """Main optimization method using geometric strategies and pattern diversity"""
        # Generate diverse initial patterns
        patterns = self.build_alternative_patterns(3)
        patterns.append(self.build_tile_pattern())
        
        # Try multiple geometric strategies
        best_strategy = None
        best_score = 0.0
        
        for i, pattern in enumerate(patterns):
            # Apply refinement to each pattern
            refined_pattern = self.construct_geometrically_optimal_packing(pattern)
            
            # Validate refined pattern
            valid, _, score = self.validate_configuration(refined_pattern)
            if valid and score > best_score:
                best_score = score
                best_strategy = refined_pattern
                
        # Return best found configuration
        return best_strategy if best_strategy is not None else self.build_tile_pattern()

    def refine_with_local_search(self, initial_positions):
        """Apply local geometric refinement focusing on improving specific arrangements"""
        # Convert to structured format
        positions = []
        for i in range(self.n_inner):
            positions.append((initial_positions[3*i], initial_positions[3*i+1], initial_positions[3*i+2]))
            
        # Apply iterative local improvement
        for iteration in range(15):  # Limited iterations for efficiency
            improved = False
            
            # Test moving each hexagon to better positions
            for i in range(self.n_inner):
                x, y, angle = positions[i]
                original_pos = (x, y, angle)
                
                # Try small adjustments in vicinity
                best_adjustment = (x, y, angle)
                best_score = 0.0
                
                # Sample nearby positions
                for dx in [-0.1, -0.05, 0, 0.05, 0.1]:
                    for dy in [-0.1, -0.05, 0, 0.05, 0.1]:
                        if abs(dx) + abs(dy) < 0.01: continue  # Skip center
                        
                        new_x, new_y = x + dx, y + dy
                        
                        # Test this position
                        test_positions = positions[:]
                        test_positions[i] = (new_x, new_y, angle)
                        
                        # Create test configuration
                        test_config = []
                        for pos in test_positions:
                            test_config.extend([pos[0], pos[1], pos[2]])
                            
                        valid, _, score = self.validate_configuration(np.array(test_config))
                        if valid and score > best_score:
                            best_score = score
                            best_adjustment = (new_x, new_y, angle)
                            
                # Apply best adjustment if improvement
                if best_score > 0 and best_score > 1e-6:
                    positions[i] = best_adjustment
                    improved = True
                    
            # Stop if no improvement
            if not improved:
                break
                        
        # Convert back to flat array
        final_config = []
        for pos in positions:
            final_config.extend([pos[0], pos[1], pos[2]])
            
        return np.array(final_config)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses geometric constraint satisfaction with pattern-based construction and local refinement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        optimizer = HexagonTilingOptimizer()
        
        # Use geometric construction approach
        best_config = optimizer.optimize_structure()
        
        # Further refine with local search
        refined_config = optimizer.refine_with_local_search(best_config)
        
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