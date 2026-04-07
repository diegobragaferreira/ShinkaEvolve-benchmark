# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit
from joblib import Parallel, delayed

# Geometry module for hexagon operations
class HexagonGeometry:
    @staticmethod
    @jit(nopython=True)
    def vertices(center_x, center_y, angle_rad, side_length=1.0):
        """Fast computation of hexagon vertices using Numba"""
        vertices = np.empty((6, 2))
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vertices[i, 0] = center_x + side_length * np.cos(theta)
            vertices[i, 1] = center_y + side_length * np.sin(theta)
        return vertices

    @staticmethod
    def create_unit_hexagon(center=(0, 0), angle_deg=0):
        """Create a unit regular hexagon centered at center with rotation angle_deg."""
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = np.cos(theta)
            y = np.sin(theta)
            vertices.append((x + center[0], y + center[1]))
        return Polygon(vertices)

# BVH Implementation for spatial acceleration
class BVHNode:
    """A node in the Bounding Volume Hierarchy"""
    def __init__(self, bounds, indices=None, children=None):
        self.bounds = bounds  # [min_x, min_y, max_x, max_y]
        self.indices = indices or []
        self.children = children or []
        self.is_leaf = len(self.children) == 0

    def get_center(self):
        return [(self.bounds[0] + self.bounds[2]) / 2, (self.bounds[1] + self.bounds[3]) / 2]

    def get_area(self):
        return (self.bounds[2] - self.bounds[0]) * (self.bounds[3] - self.bounds[1])

class BVH:
    """Bounding Volume Hierarchy for spatial acceleration"""
    def __init__(self, hexagons, max_depth=10, max_leaf_size=4):
        self.hexagons = hexagons
        self.max_depth = max_depth
        self.max_leaf_size = max_leaf_size
        self.root = None
        self.build()

    def calculate_bounds(self, indices):
        """Calculate bounding box for a list of hexagon indices"""
        if not indices:
            return [0, 0, 0, 0]

        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for idx in indices:
            hexagon = self.hexagons[idx]
            coords = list(hexagon.exterior.coords)
            for x, y in coords:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

        return [min_x, min_y, max_x, max_y]

    def build_recursive(self, indices, depth=0):
        """Recursively build the BVH tree"""
        bounds = self.calculate_bounds(indices)
        node = BVHNode(bounds, indices)

        # Stop splitting if we've reached max depth or leaf size
        if depth >= self.max_depth or len(indices) <= self.max_leaf_size:
            return node

        # Split along the longest axis
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        if width >= height:
            split_pos = (bounds[0] + bounds[2]) / 2
            left_indices = [idx for idx in indices if self.calculate_bounds([idx]).endswith(split_pos)]
            right_indices = [idx for idx in indices if not self.calculate_bounds([idx]).endswith(split_pos)]
        else:
            split_pos = (bounds[1] + bounds[3]) / 2
            left_indices = [idx for idx in indices if self.calculate_bounds([idx])[1] < split_pos]
            right_indices = [idx for idx in indices if self.calculate_bounds([idx])[1] >= split_pos]

        # Create children nodes
        if left_indices:
            node.children.append(self.build_recursive(left_indices, depth + 1))
        if right_indices:
            node.children.append(self.build_recursive(right_indices, depth + 1))

        node.is_leaf = False
        return node

    def build(self):
        """Build the BVH tree"""
        indices = list(range(len(self.hexagons)))
        self.root = self.build_recursive(indices)

    def query_overlapping(self, node, query_bounds, candidates):
        """Query overlapping elements using BVH"""
        # Check if node bounds intersect with query bounds
        if not self.bounds_intersect(node.bounds, query_bounds):
            return

        if node.is_leaf:
            candidates.extend(node.indices)
        else:
            for child in node.children:
                self.query_overlapping(child, query_bounds, candidates)

    def bounds_intersect(self, bounds1, bounds2):
        """Check if two bounding boxes intersect"""
        return not (bounds1[2] < bounds2[0] or bounds1[0] > bounds2[2] or
                   bounds1[3] < bounds2[1] or bounds1[1] > bounds2[3])

    def find_potential_overlaps(self, hexagon_idx):
        """Find potential overlapping hexagons using BVH"""
        query_bounds = self.calculate_bounds([hexagon_idx])
        candidates = []
        self.query_overlapping(self.root, query_bounds, candidates)
        return [i for i in candidates if i != hexagon_idx]

# Constraint checking module with BVH acceleration
class ConstraintChecker:
    @staticmethod
    def containment_check(hexagon, outer_hexagon):
        """Check if hexagon is fully contained within outer_hexagon."""
        return outer_hexagon.contains(hexagon)

    @staticmethod
    def overlap_check(hex1, hex2):
        """Check if two hexagons overlap."""
        return hex1.intersects(hex2)

    @staticmethod
    def parallel_containment_check(hexagons, outer_hexagon, n_jobs=-1):
        """Parallel checking of containment"""
        def check_single(hexagon):
            return ConstraintChecker.containment_check(hexagon, outer_hexagon)

        results = Parallel(n_jobs=n_jobs)(
            delayed(check_single)(h) for h in hexagons
        )
        return results

    @staticmethod
    def accelerated_overlap_check(hexagons, bvh=None, n_jobs=-1):
        """Accelerated overlap checking using BVH when available"""
        if bvh is None:
            # Fallback to parallel approach without acceleration
            def check_pair(i, j):
                return ConstraintChecker.overlap_check(hexagons[i], hexagons[j])

            n_hexagons = len(hexagons)
            results = Parallel(n_jobs=n_jobs)(
                delayed(check_pair)(i, j)
                for i in range(n_hexagons)
                for j in range(i+1, n_hexagons)
            )
            return results

        # Use BVH acceleration
        overlap_results = []
        n_hexagons = len(hexagons)

        def check_with_bvh(index):
            candidates = bvh.find_potential_overlaps(index)
            overlaps = []
            for candidate in candidates:
                if ConstraintChecker.overlap_check(hexagons[index], hexagons[candidate]):
                    overlaps.append(candidate)
            return overlaps

        # For simplicity, we'll do single-threaded processing here but keep the structure in place
        for i in range(n_hexagons):
            overlaps = check_with_bvh(i)
            for overlap in overlaps:
                # Mark as overlapping
                overlap_results.append(True)
            if not overlaps:
                overlap_results.append(False)

        return overlap_results

# Configuration generation module
class ConfigurationGenerator:
    @staticmethod
    def generate_lattice_initial():
        """Generate initial configuration based on hexagonal lattice principles"""
        positions_angles = np.zeros((12, 3))

        # Central hexagon
        positions_angles[0] = [0, 0, 0]

        # First ring: 6 hexagons around center at distance sqrt(3)
        ring1_radius = np.sqrt(3)
        for i in range(1, 7):
            angle = 2 * np.pi * (i-1) / 6
            positions_angles[i] = [ring1_radius * np.cos(angle), ring1_radius * np.sin(angle), 0]

        # Second ring: 5 hexagons arranged in a pentagonal pattern
        # Placed at radius 2*sqrt(3) with slight angular offset for better space utilization
        ring2_radius = 2 * np.sqrt(3)
        for i in range(7, 12):
            angle = 2 * np.pi * (i-7) / 5 + np.pi/10  # Offset to break symmetry
            positions_angles[i] = [ring2_radius * np.cos(angle), ring2_radius * np.sin(angle), 0]

        return positions_angles

    @staticmethod
    def generate_symmetric_initial():
        """Generate traditional symmetric initial configuration"""
        positions_angles = np.zeros((12, 3))

        # Central hexagon
        positions_angles[0] = [0, 0, 0]

        # First ring: 6 hexagons around center
        ring1_radius = 2.0
        for i in range(1, 7):
            angle = 2 * np.pi * (i-1) / 6
            positions_angles[i] = [ring1_radius * np.cos(angle), ring1_radius * np.sin(angle), 0]

        # Second ring: 5 hexagons in a pattern
        ring2_radius = 3.5
        for i in range(7, 12):
            angle = 2 * np.pi * (i-7) / 5 + np.pi/12
            positions_angles[i] = [ring2_radius * np.cos(angle), ring2_radius * np.sin(angle), 0]

        return positions_angles

# Main optimization engine
class HexagonPacker:
    def __init__(self):
        self.geom = HexagonGeometry()
        self.checker = ConstraintChecker()
        self.generator = ConfigurationGenerator()

    def estimate_outer_radius(self, positions, angles):
        """Estimate required outer hexagon radius from positions"""
        # Get all vertices of all hexagons
        all_vertices = []
        for pos, angle in zip(positions, angles):
            vertices = self.geom.vertices(pos[0], pos[1], np.deg2rad(angle))
            all_vertices.extend(vertices)

        if len(all_vertices) == 0:
            return 10.0

        all_coords = np.array(all_vertices)
        min_x, max_x = all_coords[:, 0].min(), all_coords[:, 0].max()
        min_y, max_y = all_coords[:, 1].min(), all_coords[:, 1].max()

        # Calculate distance from center to bounding box corners
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Find maximum distance to any corner
        max_dist = 0
        for vx, vy in all_vertices:
            dist = np.sqrt((vx - center_x)**2 + (vy - center_y)**2)
            max_dist = max(max_dist, dist)

        # Add safety margin
        return max_dist * 1.1

    def evaluate_constraints(self, positions, angles, penalty_scale=1000000):
        """Evaluate all constraints and return penalty value"""
        # Create hexagon objects
        all_hexagons = [self.geom.create_unit_hexagon(pos, angle)
                       for pos, angle in zip(positions, angles)]

        # Create outer hexagon
        outer_hexagon = self.geom.create_unit_hexagon((0, 0), 0)

        total_penalty = 0

        # Check containment (parallelized)
        containment_results = self.checker.parallel_containment_check(all_hexagons, outer_hexagon)
        for result in containment_results:
            if not result:
                total_penalty += penalty_scale

        # Build BVH for accelerated overlap checking
        bvh = BVH(all_hexagons)

        # Check overlaps (accelerated with BVH)
        overlap_results = self.checker.accelerated_overlap_check(all_hexagons, bvh)
        # Convert flat overlap results back to matrix form for checking
        n_hexagons = len(all_hexagons)
        overlap_matrix = [[False]*n_hexagons for _ in range(n_hexagons)]
        idx = 0
        for i in range(n_hexagons):
            for j in range(i+1, n_hexagons):
                overlap_matrix[i][j] = overlap_results[idx]
                overlap_matrix[j][i] = overlap_results[idx]
                idx += 1

        # Check for any overlaps
        for i in range(n_hexagons):
            for j in range(i+1, n_hexagons):
                if overlap_matrix[i][j]:
                    total_penalty += penalty_scale

        return total_penalty

    def objective_function(self, params):
        """Calculate objective function - negative of 1/outer_hex_side_length"""
        # Extract positions and angles
        positions_angles = params.reshape(-1, 3)
        positions = positions_angles[:, :2]
        angles = positions_angles[:, 2]

        # Estimate outer hexagon radius
        outer_radius = self.estimate_outer_radius(positions, angles)

        # Evaluate constraints
        penalty = self.evaluate_constraints(positions, angles)

        # Return negative 1/outer_radius plus penalties
        if outer_radius > 0:
            obj_val = -1.0 / outer_radius + penalty
        else:
            obj_val = np.inf

        return obj_val

    def optimize(self):
        """Main optimization routine"""
        n_hexagons = 12

        # Generate better initial configuration
        initial_positions_angles = self.generator.generate_lattice_initial()

        # Flatten parameters for optimization
        x0 = initial_positions_angles.flatten()

        # Bounds for optimization
        bounds = []
        # Positions: allow movement within reasonable bounds
        for i in range(n_hexagons * 2):
            bounds.append((-15, 15))
        # Angles: 0 to 360 degrees
        for i in range(n_hexagons):
            bounds.append((0, 360))

        # Multi-stage optimization
        # Stage 1: Coarse optimization (global search)
        try:
            result_coarse = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=100,
                popsize=20,
                seed=42,
                disp=False
            )

            # Stage 2: Local refinement
            result_fine = minimize(
                self.objective_function,
                result_coarse.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )

            optimized_params = result_fine.x
            positions_angles = optimized_params.reshape(-1, 3)
            positions = positions_angles[:, :2]
            angles = positions_angles[:, 2]

            # Compute final outer radius
            outer_radius = self.estimate_outer_radius(positions, angles)

        except Exception as e:
            print(f"Optimization failed: {e}")
            # Fallback to good initial configuration
            positions_angles = initial_positions_angles
            outer_radius = self.estimate_outer_radius(initial_positions_angles[:, :2], initial_positions_angles[:, 2])

        return positions_angles, np.array([0, 0, 0]), outer_radius

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Initialize packer
    packer = HexagonPacker()

    # Optimize the hexagon packing
    inner_hex_data, outer_hex_data, outer_hex_side_length = packer.optimize()

    # Validate the solution
    try:
        # Create hexagon objects for validation
        all_hexagons = []
        geom = HexagonGeometry()
        for i, (pos, angle) in enumerate(zip(inner_hex_data[:, :2], inner_hex_data[:, 2])):
            h = geom.create_unit_hexagon(pos, angle)
            all_hexagons.append(h)

        # Check all pairwise overlaps
        checker = ConstraintChecker()
        overlap_results = checker.parallel_overlap_check(all_hexagons)
        for result in overlap_results:
            if result:
                raise ValueError("Overlapping hexagons detected")

        # Check containment in outer hexagon
        outer_hex = geom.create_unit_hexagon((0, 0), 0)
        containment_results = checker.parallel_containment_check(all_hexagons, outer_hex)
        for result in containment_results:
            if not result:
                raise ValueError("Some hexagons outside outer hexagon")

    except ValueError as e:
        print(f"Validation error: {e}")
        # Fallback to default configuration if validation fails
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
            [0, -4, 0],
        ])
        outer_hex_side_length = 8.0

    end_time = time.time()

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END