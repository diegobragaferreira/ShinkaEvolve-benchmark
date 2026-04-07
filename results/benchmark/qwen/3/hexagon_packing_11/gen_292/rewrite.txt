# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
from numba import jit
import math

# Constants
UNIT_HEX_RADIUS = 1.0  # radius of unit hexagon circumcircle
UNIT_HEX_SIDE = 1.0    # side length of unit hexagon
PI = np.pi

class HexagonTilingOptimizer:
    """Completely different approach using geometric tiling and analytical optimization"""

    def __init__(self):
        self.best_solution = None
        self.best_score = 0.0
        self.eval_time = 0.0

    @staticmethod
    @jit(nopython=True)
    def hexagon_vertices_jit(center_x, center_y, angle_rad, side_length):
        """Compute hexagon vertices efficiently using numba"""
        vertices = np.zeros((6, 2))
        for i in range(6):
            angle = angle_rad + i * PI / 3
            vertices[i, 0] = center_x + side_length * np.cos(angle)
            vertices[i, 1] = center_y + side_length * np.sin(angle)
        return vertices

    @staticmethod
    def hexagon_vertices(center_x, center_y, angle_rad, side_length):
        """Compute hexagon vertices"""
        return HexagonTilingOptimizer.hexagon_vertices_jit(center_x, center_y, angle_rad, side_length)

    @staticmethod
    def compute_hexagon_distance(c1_x, c1_y, a1, c2_x, c2_y, a2):
        """Compute minimum distance between two hexagons using analytical approach"""
        # Simplified distance calculation for hexagons with specific constraints
        # This assumes unit hexagons with known properties
        dx = c1_x - c2_x
        dy = c1_y - c2_y
        return np.sqrt(dx*dx + dy*dy)

    @staticmethod
    def check_overlap_hexagons(h1_center_x, h1_center_y, h1_angle, h1_side,
                              h2_center_x, h2_center_y, h2_angle, h2_side):
        """Check if two hexagons overlap using vertices inclusion test"""
        vertices1 = HexagonTilingOptimizer.hexagon_vertices(h1_center_x, h1_center_y, np.radians(h1_angle), h1_side)
        vertices2 = HexagonTilingOptimizer.hexagon_vertices(h2_center_x, h2_center_y, np.radians(h2_angle), h2_side)

        # Create shapely polygons
        poly1 = Polygon(vertices1)
        poly2 = Polygon(vertices2)

        # Check if they intersect
        return poly1.intersects(poly2)

    @staticmethod
    def check_all_overlaps(inner_hex_data):
        """Check all pairs of hexagons for overlaps"""
        n = len(inner_hex_data)
        # Early return if too few hexagons
        if n < 2:
            return False

        # Check only unique pairs
        for i in range(n):
            for j in range(i+1, n):
                cx1, cy1, angle1 = inner_hex_data[i]
                cx2, cy2, angle2 = inner_hex_data[j]

                if HexagonTilingOptimizer.check_overlap_hexagons(cx1, cy1, angle1, UNIT_HEX_SIDE,
                                                                cx2, cy2, angle2, UNIT_HEX_SIDE):
                    return True
        return False

    @staticmethod
    def check_containment(inner_hex_data, outer_center=(0,0), outer_side=10):
        """Check if all inner hexagons are contained in outer hexagon"""
        outer_vertices = HexagonTilingOptimizer.hexagon_vertices(outer_center[0], outer_center[1], 0, outer_side)
        outer_polygon = Polygon(outer_vertices)

        for i in range(len(inner_hex_data)):
            cx, cy, angle = inner_hex_data[i]
            vertices = HexagonTilingOptimizer.hexagon_vertices(cx, cy, np.radians(angle), UNIT_HEX_SIDE)

            # Create hexagon polygon
            inner_polygon = Polygon(vertices)

            # Check if it's contained
            if not outer_polygon.contains(inner_polygon):
                return False

        return True

    @staticmethod
    def compute_outer_hexagon_radius(inner_hex_data, outer_hex_center=(0,0)):
        """Estimate minimum outer hexagon radius needed to contain all inner hexagons"""
        max_distance = 0
        for i in range(len(inner_hex_data)):
            cx, cy, angle = inner_hex_data[i]
            # Get all vertices of this hexagon
            vertices = HexagonTilingOptimizer.hexagon_vertices(cx, cy, np.radians(angle), UNIT_HEX_SIDE)
            # Find maximum distance from center
            for vx, vy in vertices:
                dist = np.sqrt((vx - outer_hex_center[0])**2 + (vy - outer_hex_center[1])**2)
                max_distance = max(max_distance, dist)
        return max_distance * 1.05

    @staticmethod
    def calculate_geometric_constraints(hex_positions):
        """Calculate geometric constraints for the system"""
        # Analytical approach to understand spacing constraints
        n = len(hex_positions)
        constraints = []
        
        # Compute minimum distances between hexagons (should be at least 2 units apart for non-overlapping)
        for i in range(n):
            for j in range(i+1, n):
                c1_x, c1_y = hex_positions[i]
                c2_x, c2_y = hex_positions[j]
                distance = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
                constraints.append(distance)
        
        return constraints

    @staticmethod
    def generate_tiling_pattern():
        """Generate a tiling pattern that respects geometric constraints"""
        # This uses mathematical insights about hexagonal packing
        # Create a strategic pattern that maximizes packing efficiency
        
        # Core tiling structure based on hexagonal lattice principles
        pattern = []
        # Central hexagon
        pattern.append([0.0, 0.0, 0.0])
        
        # First ring - 6 hexagons arranged in a hexagonal pattern
        # Distance between centers in hexagonal lattice is 2 (for unit hexagons)
        for i in range(6):
            angle = i * 60  # degrees
            rad_angle = np.radians(angle)
            x = 2.0 * np.cos(rad_angle)
            y = 2.0 * np.sin(rad_angle)
            pattern.append([x, y, 0.0])
            
        # Second ring - 12 hexagons (approximately)
        for i in range(12):
            angle = i * 30  # degrees
            rad_angle = np.radians(angle)
            x = 4.0 * np.cos(rad_angle)
            y = 4.0 * np.sin(rad_angle)
            pattern.append([x, y, 0.0])
            
        # Trim to exactly 11 hexagons with specific positions
        pattern = pattern[:11]
        
        # Apply small random perturbations to avoid degeneracy
        result = np.array(pattern)
        np.random.seed(42)
        for i in range(len(result)):
            result[i][0] += np.random.normal(0, 0.05)
            result[i][1] += np.random.normal(0, 0.05)
            
        return result

    @staticmethod
    def analytical_refinement(initial_config):
        """Apply analytical geometric refinement to the configuration"""
        # Use geometric reasoning to determine optimal positions
        # This is a simplified approach but captures the essence of analytical optimization
        
        # Convert to array for easier manipulation
        config = initial_config.copy()
        
        # Apply geometric constraints more carefully
        # Ensure minimal distances between centers are respected
        # (at least 2 units for unit hexagons)
        
        # Group by clusters based on distance for more targeted optimization
        clusters = []
        for i in range(len(config)):
            cluster_found = False
            for cluster in clusters:
                # Check if this point belongs to an existing cluster
                for pt in cluster:
                    distance = np.sqrt((config[i][0] - pt[0])**2 + (config[i][1] - pt[1])**2)
                    if distance < 3.0:  # Within cluster threshold
                        cluster.append(config[i])
                        cluster_found = True
                        break
            if not cluster_found:
                clusters.append([config[i]])
                
        # For each cluster, apply centroid-based refinement
        refined_config = config.copy()
        for i, cluster in enumerate(clusters):
            if len(cluster) > 1:
                # Compute centroid
                centroid_x = sum([pt[0] for pt in cluster]) / len(cluster)
                centroid_y = sum([pt[1] for pt in cluster]) / len(cluster)
                
                # Adjust positions to be closer to centroid while respecting constraints
                for j, pt in enumerate(cluster):
                    # Move towards centroid but not too much (maintain separation)
                    factor = 0.3  # Adjustment factor
                    refined_config[clusters.index(cluster)][j][0] = (
                        refined_config[clusters.index(cluster)][j][0] * (1 - factor) + 
                        centroid_x * factor
                    )
                    refined_config[clusters.index(cluster)][j][1] = (
                        refined_config[clusters.index(cluster)][j][1] * (1 - factor) + 
                        centroid_y * factor
                    )
                    
        return refined_config

    @staticmethod
    def region_based_sampling(current_config, max_iterations=50):
        """Region-based sampling approach based on geometric analysis"""
        # Divide the plane into regions and sample strategically
        best_config = current_config.copy()
        best_score = -float('inf')
        
        # Sample different regions of the configuration space
        for iteration in range(max_iterations):
            # Create a modified configuration by moving hexagons strategically
            modified_config = current_config.copy()
            
            # Determine what type of adjustment to make
            adjustment_type = iteration % 3
            
            if adjustment_type == 0:  # Global adjustment - shift entire configuration
                shift_x = np.random.uniform(-0.5, 0.5)
                shift_y = np.random.uniform(-0.5, 0.5)
                for i in range(len(modified_config)):
                    modified_config[i][0] += shift_x
                    modified_config[i][1] += shift_y
                    
            elif adjustment_type == 1:  # Local adjustment - move individual hexagons
                # Pick a random hexagon
                idx = np.random.randint(0, len(modified_config))
                modified_config[idx][0] += np.random.uniform(-0.3, 0.3)
                modified_config[idx][1] += np.random.uniform(-0.3, 0.3)
                
            else:  # Rotation adjustment
                # Pick a random hexagon
                idx = np.random.randint(0, len(modified_config))
                modified_config[idx][2] += np.random.uniform(-10, 10)
                modified_config[idx][2] = modified_config[idx][2] % 360
                
            # Score the modified configuration
            score = HexagonTilingOptimizer.evaluate_layout(modified_config)
            
            if score > best_score:
                best_score = score
                best_config = modified_config.copy()
                
            # Adaptive cooling - reduce randomness over time
            if iteration > max_iterations * 0.7:
                # Reduce magnitude of changes
                for i in range(len(modified_config)):
                    modified_config[i][0] += np.random.uniform(-0.1, 0.1)
                    modified_config[i][1] += np.random.uniform(-0.1, 0.1)
                    
        return best_config

    @staticmethod
    def evaluate_layout(inner_hex_data):
        """Evaluate the quality of a given hexagon layout"""
        # Check overlaps first (early rejection)
        if HexagonTilingOptimizer.check_all_overlaps(inner_hex_data):
            return -np.inf  # Large penalty for overlaps

        # Check containment
        estimated_radius = HexagonTilingOptimizer.compute_outer_hexagon_radius(inner_hex_data)
        if not HexagonTilingOptimizer.check_containment(inner_hex_data, (0,0), estimated_radius * 2):
            return -np.inf  # Large penalty for containment violations

        # Return inverse of outer hexagon side length (we want to maximize 1/outer_side)
        return 1.0 / estimated_radius

    @staticmethod
    def adaptive_geometric_optimization():
        """Main optimization method using geometric insights"""
        # Start with a good tiling pattern
        current_config = HexagonTilingOptimizer.generate_tiling_pattern()
        
        # Apply analytical refinement
        refined_config = HexagonTilingOptimizer.analytical_refinement(current_config)
        
        # Apply region-based sampling
        best_config = HexagonTilingOptimizer.region_based_sampling(refined_config, 100)
        
        # Final scoring
        score = HexagonTilingOptimizer.evaluate_layout(best_config)
        
        return best_config, score

    @staticmethod
    def geometric_local_optimization(initial_config):
        """Local optimization focused purely on geometric constraints"""
        def objective_function(params):
            # Reshape parameters
            data = params.reshape(-1, 3)
            
            # Check if valid
            if HexagonTilingOptimizer.check_all_overlaps(data):
                return float('inf')
                
            # Return negative of 1/outer_radius (minimize negative to maximize 1/outer_radius)
            radius = HexagonTilingOptimizer.compute_outer_hexagon_radius(data)
            return -1.0 / radius
            
        # Flatten current configuration
        flat_params = initial_config.flatten()
        
        # Optimize using trust-constr with geometric constraints
        try:
            result = minimize(
                objective_function, 
                flat_params,
                method='trust-constr',
                options={'maxiter': 50, 'verbose': 0}
            )
            
            if result.success:
                # Reshape back
                refined_config = result.x.reshape(-1, 3)
                return refined_config
        except:
            pass
            
        return initial_config

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        # Use geometric tiling approach
        best_config, best_score = HexagonTilingOptimizer.adaptive_geometric_optimization()
        
        # Apply final geometric local optimization
        final_config = HexagonTilingOptimizer.geometric_local_optimization(best_config)
        
        # Final evaluation
        final_score = HexagonTilingOptimizer.evaluate_layout(final_config)
        
        # Ensure valid solution
        if final_score <= -1e5 or not HexagonTilingOptimizer.check_all_overlaps(final_config):
            # Fall back to baseline
            inner_hex_data = np.array([
                [0, 0, 0],          # center
                [-2.5, 0, 0],       # left
                [2.5, 0, 0],        # right
                [-1.25, 2.17, 0],   # top-left
                [1.25, 2.17, 0],    # top-right
                [-1.25, -2.17, 0],  # bottom-left
                [1.25, -2.17, 0],   # bottom-right
                [-3.75, 2.17, 0],   # far top-left
                [3.75, 2.17, 0],    # far top-right
                [-3.75, -2.17, 0],  # far bottom-left
                [3.75, -2.17, 0],   # far bottom-right
            ])
            outer_hex_side_length = 8.0
        else:
            inner_hex_data = final_config
            # Compute actual outer hexagon size
            estimated_radius = HexagonTilingOptimizer.compute_outer_hexagon_radius(inner_hex_data)
            outer_hex_side_length = estimated_radius * 2.0

        # Set outer hexagon at center with zero rotation
        outer_hex_data = np.array([0, 0, 0])

        # Validate solution
        if not HexagonTilingOptimizer.check_all_overlaps(inner_hex_data) and \
           HexagonTilingOptimizer.check_containment(inner_hex_data, (0,0), outer_hex_side_length):
            pass
        else:
            # Fall back to known good configuration
            inner_hex_data = np.array([
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
            ])
            outer_hex_side_length = 8.0
            outer_hex_data = np.array([0, 0, 0])

    except Exception as e:
        print(f"Exception in optimization: {e}")
        # Fallback to baseline approach
        inner_hex_data = np.array([
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
        ])
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0, 0, 0])

    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END