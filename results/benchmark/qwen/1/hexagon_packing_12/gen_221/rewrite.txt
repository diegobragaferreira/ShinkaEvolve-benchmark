# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from numba import jit
from typing import Tuple, List, Optional, Any

class HexagonPacker:
    """Optimized hexagon packing solver with enhanced performance."""
    
    def __init__(self, n_hexagons: int = 12):
        self.n_hexagons = n_hexagons
        self.violation_history = []
        
    @staticmethod
    @jit(nopython=True)
    def _hexagon_vertices(center_x: float, center_y: float, angle_rad: float, 
                         side_length: float = 1.0) -> np.ndarray:
        """Fast computation of hexagon vertices using Numba."""
        vertices = np.empty((6, 2))
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vertices[i, 0] = center_x + side_length * np.cos(theta)
            vertices[i, 1] = center_y + side_length * np.sin(theta)
        return vertices
    
    @staticmethod
    def _create_unit_hexagon(center: Tuple[float, float] = (0, 0), 
                           angle_deg: float = 0) -> Polygon:
        """Create a unit regular hexagon."""
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = np.cos(theta)
            y = np.sin(theta)
            vertices.append((x + center[0], y + center[1]))
        return Polygon(vertices)
    
    @staticmethod
    def _estimate_outer_hexagon_radius(positions: np.ndarray, angles: np.ndarray) -> float:
        """Estimate required outer hexagon radius from positions."""
        # Get all vertices of all hexagons
        all_vertices = []
        for pos, angle in zip(positions, angles):
            vertices = HexagonPacker._hexagon_vertices(pos[0], pos[1], np.deg2rad(angle))
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
        return max_dist * 1.05
    
    @staticmethod
    def _get_hexagon_centers(positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Get centers of all hexagons for spatial indexing."""
        return np.array(positions)
    
    @staticmethod
    def _fast_check_overlap_pair_fast(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Fast overlap check using bounding circles for early rejection."""
        # Calculate centroids
        center1 = np.mean(hex1_vertices, axis=0)
        center2 = np.mean(hex2_vertices, axis=0)

        # Distance between centers
        dist = np.sqrt(np.sum((center1 - center2)**2))

        # If distance is greater than sum of radii, no overlap
        # For unit hexagon, approximate circumradius is 1
        if dist > 2.0:
            return False

        # Fall back to actual polygon intersection test
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    
    @staticmethod
    def _fast_check_overlaps_spatial(positions: np.ndarray, angles: np.ndarray, 
                                   tree: Optional[cKDTree] = None, 
                                   max_distance: float = 2.0) -> bool:
        """Fast overlap checking using spatial indexing."""
        if tree is None:
            # Build spatial tree from centers
            centers = HexagonPacker._get_hexagon_centers(positions, angles)
            tree = cKDTree(centers)

        # Query neighbors within max_distance
        pairs = tree.query_pairs(max_distance, p=np.inf)

        # Check actual overlaps for candidate pairs
        for i, j in pairs:
            # Get vertices for both hexagons
            hex1_vertices = HexagonPacker._hexagon_vertices(positions[i][0], positions[i][1], 
                                                          np.deg2rad(angles[i]))
            hex2_vertices = HexagonPacker._hexagon_vertices(positions[j][0], positions[j][1], 
                                                          np.deg2rad(angles[j]))

            if HexagonPacker._fast_check_overlap_pair_fast(hex1_vertices, hex2_vertices):
                return True  # Found overlap

        return False  # No overlaps found
    
    def _calculate_objective(self, params: np.ndarray, outer_radius_guess: Optional[float] = None,
                           penalty_scale: float = 1000000) -> float:
        """Calculate objective function with adaptive penalty scaling."""
        # Extract positions and angles
        positions_angles = params.reshape(-1, 3)
        positions = positions_angles[:, :2]
        angles = positions_angles[:, 2]

        # Estimate outer hexagon radius
        estimated_radius = self._estimate_outer_hexagon_radius(positions, angles)
        if outer_radius_guess is not None:
            outer_radius = outer_radius_guess
        else:
            outer_radius = estimated_radius

        # Create outer hexagon (centered at origin)
        outer_hex = self._create_unit_hexagon((0, 0), 0)

        # Check constraints
        total_penalty = 0
        violation_count = 0

        # Check containment
        for i, (pos, angle) in enumerate(zip(positions, angles)):
            hexagon = self._create_unit_hexagon(pos, angle)
            if not self._check_containment(hexagon, outer_hex):
                total_penalty += penalty_scale
                violation_count += 1

        # Check overlaps using spatial indexing (much faster)
        centers = self._get_hexagon_centers(positions, angles)
        tree = cKDTree(centers)

        # Check all pairs using the spatial tree
        overlap_found = False
        try:
            # Use spatial indexing for fast overlap checking
            overlap_found = self._fast_check_overlaps_spatial(positions, angles, tree)
        except Exception as e:
            # Fallback to brute force if spatial indexing fails
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    hex1 = self._create_unit_hexagon(positions[i], angles[i])
                    hex2 = self._create_unit_hexagon(positions[j], angles[j])
                    if self._check_overlap(hex1, hex2):
                        overlap_found = True
                        break
                if overlap_found:
                    break

        if overlap_found:
            total_penalty += penalty_scale
            violation_count += 1

        # Adaptive penalty scaling based on violation history
        if len(self.violation_history) > 0:
            avg_violations = np.mean(self.violation_history[-10:])  # Average over last 10 evaluations
            if avg_violations > 0.5:  # If many violations recently
                # Increase penalty to be more strict
                penalty_scale *= 1.2
            elif avg_violations < 0.2:  # If few violations recently
                # Decrease penalty slightly to allow more exploration
                penalty_scale *= 0.95

        # Update violation history
        self.violation_history.append(violation_count)

        # Return negative 1/outer_radius plus penalties
        if outer_radius > 0:
            obj_val = -1.0 / outer_radius + total_penalty
        else:
            obj_val = np.inf

        return obj_val
    
    @staticmethod
    def _check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
        """Check if two hexagons overlap."""
        return hex1.intersects(hex2)
    
    @staticmethod
    def _check_containment(hexagon: Polygon, outer_hexagon: Polygon) -> bool:
        """Check if hexagon is fully contained within outer_hexagon."""
        return outer_hexagon.contains(hexagon)
    
    def _create_better_initial_config(self) -> np.ndarray:
        """Create a much better initial configuration using hexagonal packing principles."""
        # Start with more sophisticated symmetric arrangement
        positions_angles = np.zeros((self.n_hexagons, 3))

        # Central hexagon
        positions_angles[0] = [0, 0, 0]

        # Arrange in concentric rings - optimized layout
        # First ring: 6 hexagons around center (at radius of 2.0)
        ring1_radius = 2.0
        for i in range(1, 7):
            angle = 2 * np.pi * (i-1) / 6
            positions_angles[i] = [ring1_radius * np.cos(angle), ring1_radius * np.sin(angle), 0]

        # Second ring: 5 hexagons in a pattern that leaves room for optimization
        # These should be placed to maximize space efficiency
        ring2_radius = 3.5
        for i in range(7, self.n_hexagons):
            # Adjusted angles to avoid some overlap issues
            angle = 2 * np.pi * (i-7) / 5 + np.pi/12  # Small offset for better distribution
            positions_angles[i] = [ring2_radius * np.cos(angle), ring2_radius * np.sin(angle), 0]

        return positions_angles
    
    def _optimize_hexagon_packing_multistage(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Multi-stage optimization for better results."""
        # Stage 1: Coarse optimization with relaxed constraints
        initial_positions_angles = self._create_better_initial_config()
        x0 = initial_positions_angles.flatten()

        # Bounds for optimization
        bounds = []
        # Positions: allow wider movement
        for i in range(self.n_hexagons * 2):
            bounds.append((-15, 15))
        # Angles: 0 to 360 degrees
        for i in range(self.n_hexagons):
            bounds.append((0, 360))

        # Stage 1: Coarse optimization
        try:
            result1 = differential_evolution(
                self._calculate_objective,
                bounds,
                args=(None, 100000),  # Reduced penalty for coarse stage
                maxiter=150,
                popsize=25,
                seed=42,
                disp=False
            )

            # Stage 2: Refine with tighter constraints
            refined_params = result1.x
            positions_angles = refined_params.reshape(-1, 3)

            # More precise optimization with full penalties
            result2 = differential_evolution(
                self._calculate_objective,
                bounds,
                args=(None, 1000000),  # Full penalty for fine stage
                maxiter=300,
                popsize=30,
                seed=42,
                disp=False
            )

            optimized_params = result2.x
            positions_angles = optimized_params.reshape(-1, 3)
            positions = positions_angles[:, :2]
            angles = positions_angles[:, 2]

            # Compute final outer radius based on optimized positions
            outer_radius = self._estimate_outer_hexagon_radius(positions, angles)

        except Exception as e:
            print(f"Optimization failed: {e}")
            # Fallback to good initial configuration
            positions_angles = initial_positions_angles
            outer_radius = self._estimate_outer_hexagon_radius(initial_positions_angles[:, :2], 
                                                              initial_positions_angles[:, 2])

        return positions_angles, np.array([0, 0, 0]), outer_radius
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Main solving method that returns the optimized packing."""
        # Optimize the hexagon packing using multi-stage approach
        inner_hex_data, outer_hex_data, outer_hex_side_length = self._optimize_hexagon_packing_multistage()

        # Validate the solution
        try:
            # Create hexagon objects for validation
            all_hexagons = []
            for i, (pos, angle) in enumerate(zip(inner_hex_data[:, :2], inner_hex_data[:, 2])):
                h = self._create_unit_hexagon(pos, angle)
                all_hexagons.append(h)

            # Check all pairwise overlaps
            for i in range(len(all_hexagons)):
                for j in range(i+1, len(all_hexagons)):
                    if all_hexagons[i].intersects(all_hexagons[j]):
                        raise ValueError("Overlapping hexagons detected")
                        
            # Check containment in outer hexagon (approximate)
            outer_hex = self._create_unit_hexagon((0, 0), 0)
            for hexagon in all_hexagons:
                if not outer_hex.contains(hexagon):
                    raise ValueError("Some hexagons outside outer hexagon")
                    
        except ValueError as e:
            print(f"Validation error: {e}")
            # Fallback to a reasonable configuration if validation fails
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

        return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Create packer instance and solve
    packer = HexagonPacker(n_hexagons=12)
    inner_hex_data, outer_hex_data, outer_hex_side_length = packer.solve()
    
    end_time = time.time()
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END