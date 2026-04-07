# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List, Optional
from collections import defaultdict
import math

class CirclePackingOptimizer:
    """Main optimizer class for 26-circle packing problem with enhanced performance."""

    def __init__(self, n_circles: int = 26, seed: int = 42):
        self.n_circles = n_circles
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def validate_placement(self, circles: np.ndarray) -> bool:
        """Validate that circles are within bounds and don't overlap."""
        n = len(circles)

        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False

        # Check non-overlap constraints using KDTree for efficiency
        points = circles[:, :2]
        tree = cKDTree(points)

        # Find all pairs within distance 2*r (minimum separation needed to avoid overlap)
        pairs = tree.query_pairs(2 * min(circles[:, 2]), output_type='ndarray')

        for i, j in pairs:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_distance_sq = (r1 + r2)**2
            if distance_sq < min_distance_sq:
                return False

        return True

    def compute_voronoi_areas(self, points: np.ndarray) -> np.ndarray:
        """Estimate Voronoi cell areas for each point."""
        # Simple estimation: use distance to nearest neighbors
        tree = cKDTree(points)
        distances = tree.query(points, k=4)  # Query 4 nearest neighbors (including self)
        areas = []
        for i, (distances_to_neighbors, indices) in enumerate(zip(distances[0], distances[1])):
            # Use average distance to determine relative area
            avg_dist = np.mean(distances_to_neighbors[1:])  # exclude self
            areas.append(avg_dist**2)
        return np.array(areas)
        
    def generate_enhanced_voronoi_initialization(self) -> np.ndarray:
        """Generate initial circle positions with enhanced Voronoi-based spreading."""
        # Create a grid of candidate positions
        grid_size = max(4, int(np.ceil(np.sqrt(self.n_circles)) + 1))
        x_coords = np.linspace(0.05, 0.95, grid_size)
        y_coords = np.linspace(0.05, 0.95, grid_size)

        # Generate all grid points
        grid_points = []
        for x in x_coords:
            for y in y_coords:
                grid_points.append([x, y])

        # If we have more circles than grid points, add some random points
        if len(grid_points) < self.n_circles:
            extra_points = self.n_circles - len(grid_points)
            for _ in range(extra_points):
                grid_points.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])

        # Shuffle the points to avoid systematic bias
        random.shuffle(grid_points)

        # Convert to numpy array for easier processing
        points = np.array(grid_points[:self.n_circles])
        
        # Calculate approximate Voronoi areas for each point
        voronoi_areas = self.compute_voronoi_areas(points)
        
        # Sort points by their estimated Voronoi area in descending order
        # (larger areas get priority for larger initial radii)
        sorted_indices = np.argsort(voronoi_areas)[::-1]
        sorted_points = points[sorted_indices]
        
        # Initialize circles with varying initial radii based on Voronoi area
        circles = np.zeros((self.n_circles, 3))
        circles[:, 0] = sorted_points[:, 0]  # x coordinates
        circles[:, 1] = sorted_points[:, 1]  # y coordinates
        
        # Assign initial radii based on Voronoi area ranking
        # Higher-ranked (larger Voronoi area) get larger initial radii
        for i, idx in enumerate(sorted_indices):
            normalized_area = voronoi_areas[idx] / np.max(voronoi_areas)
            # Assign initial radius inversely proportional to area (smaller area = larger radius)
            initial_radius = 0.02 * (1.0 + 0.5 * normalized_area)
            circles[i, 2] = min(initial_radius, 0.15)  # Cap at 0.15

        return circles

    def generate_spiral_initialization(self) -> np.ndarray:
        """Generate initial circle positions using a spiral pattern."""
        circles = np.zeros((self.n_circles, 3))

        # Spiral parameters
        a = 0.05  # spiral parameter
        b = 0.05  # spiral parameter

        for i in range(self.n_circles):
            angle = 2 * np.pi * i / self.n_circles * 5  # spiral with 5 turns
            radius = a + b * angle
            radius = min(radius, 0.45)  # cap at reasonable value

            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4

            # Clip to valid range
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)

            circles[i, 0] = x
            circles[i, 1] = y
            circles[i, 2] = 0.01  # small initial radius

        return circles

    def generate_greedy_fallback(self) -> np.ndarray:
        """Fallback method to generate a feasible configuration."""
        # Simple greedy approach: place circles in order of decreasing radius
        circles = np.zeros((self.n_circles, 3))

        # Start with small radii and gradually increase
        # Place in a way that they don't overlap initially
        positions = []
        radii = []

        # Try to place circles greedily by spacing them out
        placed = 0
        radius = 0.05
        while placed < self.n_circles and radius > 0.005:
            # Try placing circles in a spiral pattern or grid
            attempt = 0
            while attempt < 100 and placed < self.n_circles:
                # Place in grid-like fashion
                rows = int(np.sqrt(self.n_circles)) + 1
                cols = self.n_circles // rows + 1

                for i in range(rows):
                    for j in range(cols):
                        if placed >= self.n_circles:
                            break
                        x = 0.1 + j * 0.8 / cols
                        y = 0.1 + i * 0.8 / rows

                        # Check if this position is valid
                        valid = True
                        for pos, rad in zip(positions, radii):
                            dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                            if dist_sq < (rad + radius)**2:
                                valid = False
                                break

                        if valid:
                            positions.append([x, y])
                            radii.append(radius)
                            placed += 1
                attempt += 1

            radius *= 0.9  # Decrease radius slightly

        # Fill remaining circles
        while placed < self.n_circles:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            positions.append([x, y])
            radii.append(0.01)
            placed += 1

        circles[:, 0] = [pos[0] for pos in positions]
        circles[:, 1] = [pos[1] for pos in positions]
        circles[:, 2] = radii

        return circles

    def compute_forces(self, circles: np.ndarray,
                      repulsion_strength: float = 150.0,
                      boundary_strength: float = 50.0) -> Tuple[np.ndarray, float]:
        """
        Compute forces acting on each circle including repulsion and boundary attraction.

        Returns:
            forces: array of shape (n_circles, 2) representing force vectors
            potential_energy: total potential energy of the system
        """
        n = len(circles)
        forces = np.zeros((n, 2))
        potential_energy = 0.0

        # Repulsion forces between circles
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]

                dx = x1 - x2
                dy = y1 - y2
                distance_sq = dx*dx + dy*dy

                # Skip if too far apart
                if distance_sq > 4*(r1+r2)**2:  # Approximate cutoff
                    continue

                distance = np.sqrt(distance_sq)

                # Avoid division by zero
                if distance < 1e-10:
                    continue

                # Repulsive force (inverse square law)
                force_magnitude = repulsion_strength / (distance_sq + 1e-10)

                # Apply force direction
                fx = force_magnitude * dx / distance
                fy = force_magnitude * dy / distance

                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy

                # Energy contribution (potential energy)
                potential_energy += force_magnitude * (1.0/distance)

        # Boundary forces (spring-like attraction to boundaries)
        for i in range(n):
            x, y, r = circles[i]

            # Force components towards boundaries
            fx = 0.0
            fy = 0.0

            # Left boundary
            if x - r < 0:
                fx += boundary_strength * (0 - (x - r))
            # Right boundary
            if x + r > 1:
                fx += boundary_strength * (1 - (x + r))

            # Bottom boundary
            if y - r < 0:
                fy += boundary_strength * (0 - (y - r))
            # Top boundary
            if y + r > 1:
                fy += boundary_strength * (1 - (y + r))

            forces[i, 0] += fx
            forces[i, 1] += fy

        return forces, potential_energy

    def update_circles(self, circles: np.ndarray, forces: np.ndarray,
                      dt: float = 0.01, max_velocity: float = 0.05) -> np.ndarray:
        """
        Update circle positions based on forces and velocity limits.
        """
        updated_circles = circles.copy()

        # Update velocities and positions
        for i in range(len(circles)):
            x, y, r = circles[i]

            # Update velocity (force gives acceleration)
            vx = forces[i, 0] * dt
            vy = forces[i, 1] * dt

            # Apply velocity limits
            vel_norm = np.sqrt(vx*vx + vy*vy)
            if vel_norm > max_velocity:
                vx = vx * max_velocity / vel_norm
                vy = vy * max_velocity / vel_norm

            # Update position
            new_x = x + vx
            new_y = y + vy

            # Ensure new positions respect boundaries
            new_x = np.clip(new_x, r, 1-r)
            new_y = np.clip(new_y, r, 1-r)

            updated_circles[i, 0] = new_x
            updated_circles[i, 1] = new_y

        return updated_circles

    def adaptive_simulation_step(self, circles: np.ndarray,
                                target_energy: float = 1e-4,
                                max_steps: int = 1000) -> np.ndarray:
        """
        Perform adaptive simulation to reach equilibrium with smarter damping.
        """
        # Parameters for adaptive dynamics
        dt = 0.01
        damping_factor = 0.95
        energy_threshold = target_energy
        
        # Track energy evolution for adaptive damping
        energy_history = []
        prev_energy = float('inf')
        
        for step in range(max_steps):
            # Compute forces
            forces, current_energy = self.compute_forces(circles, repulsion_strength=150.0)

            # Check for convergence using rate of change in energy
            if len(energy_history) > 10:
                # Calculate average energy change over last 10 steps
                recent_avg = np.mean(energy_history[-10:])
                energy_change = abs(recent_avg - current_energy)
                if energy_change < energy_threshold:
                    break
                    
            # Store energy for tracking
            energy_history.append(current_energy)

            # Update circles
            circles = self.update_circles(circles, forces, dt)

            # Adaptive damping based on energy change rate
            if len(energy_history) >= 2:
                energy_change = abs(energy_history[-1] - energy_history[-2])
                if energy_change < 1e-5:
                    # Slow energy change, increase damping
                    damping_factor = min(damping_factor * 0.98, 0.999)
                else:
                    # Fast energy change, decrease damping
                    damping_factor = max(damping_factor * 1.02, 0.95)
            else:
                damping_factor = 0.95
            
            # Apply damping to prevent oscillation
            if step % 10 == 0:
                dt *= damping_factor
            
            # Occasionally reset time step
            if step % 50 == 0:
                dt = max(0.001, dt * 0.95)

        return circles

    def multi_scale_refinement(self, initial_circles: np.ndarray,
                              max_iterations: int = 200) -> np.ndarray:
        """
        Refine the solution using multi-scale approach with enhanced parameters.
        """
        circles = initial_circles.copy()

        # Phase 1: Coarse optimization using large time steps and strong forces
        circles = self.adaptive_simulation_step(circles, target_energy=1e-2, max_steps=300)

        # Phase 2: Medium optimization with moderate forces
        circles = self.adaptive_simulation_step(circles, target_energy=1e-3, max_steps=200)

        # Phase 3: Fine-tuning with very small forces and high precision
        circles = self.adaptive_simulation_step(circles, target_energy=1e-4, max_steps=150)

        return circles

    def golden_section_search(self, func, a, b, tolerance=1e-6):
        """Perform golden section search to find maximum."""
        phi = (1 + math.sqrt(5)) / 2
        resphi = 2 - phi
        
        x1 = a + (b - a) * resphi
        x2 = b - (b - a) * resphi
        
        f1 = func(x1)
        f2 = func(x2)
        
        while abs(b - a) > tolerance:
            if f1 > f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + (b - a) * resphi
                f1 = func(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = b - (b - a) * resphi
                f2 = func(x2)
        
        return (a + b) / 2

    def constraint_aware_local_search(self, circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """
        Apply enhanced constraint-aware local search with golden section optimization.
        """
        improved_circles = circles.copy()
        n = len(improved_circles)

        for iteration in range(max_iterations):
            improved = False

            # Try to increase radii while maintaining constraints
            for i in range(n):
                x, y, r = improved_circles[i]

                # Calculate maximum possible radius at this position
                max_radius = min(x, 1-x, y, 1-y)

                # Find neighboring circles to check constraints
                neighbors = []
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = improved_circles[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        min_dist_sq = (r + r2)**2
                        neighbors.append((j, dist_sq, min_dist_sq))

                # Calculate maximum increase in radius using golden section search
                def radius_constraint_check(test_radius):
                    valid = True
                    for j, dist_sq, min_dist_sq in neighbors:
                        x2, y2, r2 = improved_circles[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        if dist_sq < (test_radius + r2)**2:
                            valid = False
                            break
                    return -1 if not valid else test_radius  # Return negative if invalid

                # Use golden section search for finding optimal radius
                if max_radius > r + 1e-6:
                    # Find the best safe radius using golden search
                    try:
                        # Only search if we have enough space
                        if max_radius - r > 0.01:
                            # Golden section search to maximize radius
                            best_radius = self.golden_section_search(
                                lambda rad: radius_constraint_check(rad), 
                                r, max_radius, 1e-6
                            )
                            
                            # Validate the returned result
                            if best_radius > r + 1e-6 and radius_constraint_check(best_radius) > 0:
                                improved_circles[i, 2] = best_radius
                                improved = True
                        else:
                            # Direct check with small increments
                            for test_radius in [r + 0.005, r + 0.0025, r + 0.001]:
                                if test_radius <= max_radius and radius_constraint_check(test_radius) > 0:
                                    improved_circles[i, 2] = test_radius
                                    improved = True
                                    break
                    except:
                        # Fallback to simple approach if golden section fails
                        test_radius = min(max_radius, r + 0.005)
                        if radius_constraint_check(test_radius) > 0:
                            improved_circles[i, 2] = test_radius
                            improved = True

            # If no improvement from radius increases, try position adjustments
            if not improved:
                for i in range(n):
                    x, y, r = improved_circles[i]
                    
                    # More systematic exploration of neighborhood
                    movements = [
                        (-0.01, -0.01), (-0.01, 0), (-0.01, 0.01),
                        (0, -0.01),              (0, 0.01),
                        (0.01, -0.01), (0.01, 0), (0.01, 0.01),
                        (-0.005, -0.005), (-0.005, 0), (-0.005, 0.005),
                        (0, -0.005),              (0, 0.005),
                        (0.005, -0.005), (0.005, 0), (0.005, 0.005)
                    ]

                    best_x, best_y = x, y
                    best_score = -float('inf')
                    best_radius = r

                    for dx, dy in movements:
                        new_x, new_y = x + dx, y + dy
                        
                        # Check bounds
                        if new_x - r < 0 or new_x + r > 1 or new_y - r < 0 or new_y + r > 1:
                            continue
                            
                        # Check overlap with neighbors
                        overlap_penalty = 0
                        valid = True
                        for j in range(n):
                            if i != j:
                                x2, y2, r2 = improved_circles[j]
                                dist_sq = (new_x - x2)**2 + (new_y - y2)**2
                                min_dist_sq = (r + r2)**2
                                if dist_sq < min_dist_sq:
                                    overlap_penalty += (min_dist_sq - dist_sq) * 1000
                                    valid = False
                        
                        if valid:
                            # Score based on overlap reduction and radius preservation
                            score = -overlap_penalty + r
                            if score > best_score:
                                best_score = score
                                best_x, best_y = new_x, new_y

                    # Apply the best movement if it helps
                    if best_x != x or best_y != y:
                        improved_circles[i, 0] = best_x
                        improved_circles[i, 1] = best_y
                        improved = True

            # If no improvement made, exit loop
            if not improved:
                break

        return improved_circles

    def improve_with_gradient_descent(self, circles: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """
        Apply enhanced gradient-based improvement focusing on increasing radii.
        """
        improved_circles = circles.copy()
        n = len(improved_circles)
        
        # Precompute neighbor relationships for efficiency
        neighbors_cache = {}
        for i in range(n):
            neighbors = []
            for j in range(n):
                if i != j:
                    neighbors.append(j)
            neighbors_cache[i] = neighbors

        for iteration in range(max_iter):
            # For each circle, compute how much we can increase radius
            updated = False
            
            # Process in randomized order for better exploration
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                x, y, r = improved_circles[i]
                max_radius = min(x, 1-x, y, 1-y)
                
                # Find minimum distance to neighbors for overlap constraints
                min_dist = float('inf')
                for j in neighbors_cache[i]:
                    x2, y2, r2 = improved_circles[j]
                    dist_sq = (x - x2)**2 + (y - y2)**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                
                # If we can increase radius
                if min_dist < (r + 0.001)**2:
                    # We're currently overlapping, reduce radius
                    safe_radius = min(r, max_radius)
                    if safe_radius > r + 1e-6:
                        improved_circles[i, 2] = safe_radius
                        updated = True
                else:
                    # Try to increase radius up to boundary
                    if r < max_radius - 1e-6:
                        # Test if we can safely increase
                        test_radius = min(max_radius, r + 0.005)
                        valid = True
                        
                        # Check for overlaps with all neighbors
                        for j in neighbors_cache[i]:
                            x2, y2, r2 = improved_circles[j]
                            dist_sq = (x - x2)**2 + (y - y2)**2
                            if dist_sq < (test_radius + r2)**2:
                                valid = False
                                break
                        
                        if valid:
                            improved_circles[i, 2] = test_radius
                            updated = True
            
            if not updated:
                break
        
        return improved_circles

    def optimize(self) -> np.ndarray:
        """Main optimization pipeline with improved initialization strategies."""
        # Generate multiple initial configurations and select the best
        initial_configs = []
        
        # Enhanced Voronoi initialization
        initial_configs.append(("enhanced_voronoi", self.generate_enhanced_voronoi_initialization()))
        
        # Spiral initialization
        initial_configs.append(("spiral", self.generate_spiral_initialization()))
        
        # Random initialization
        rand_config = np.zeros((self.n_circles, 3))
        for i in range(self.n_circles):
            rand_config[i, 0] = np.random.uniform(0.05, 0.95)
            rand_config[i, 1] = np.random.uniform(0.05, 0.95)
            rand_config[i, 2] = 0.01
        initial_configs.append(("random", rand_config))

        best_result = None
        best_sum_radii = -float('inf')

        # Evaluate each initial configuration
        for config_name, config in initial_configs:
            # Multi-scale refinement
            refined = self.multi_scale_refinement(config, max_iterations=200)
            
            # Constraint-aware local search
            refined = self.constraint_aware_local_search(refined, max_iterations=200)
            
            # Additional gradient refinement
            refined = self.improve_with_gradient_descent(refined, max_iter=100)
            
            # Validate and check fitness
            if self.validate_placement(refined):
                sum_radii = np.sum(refined[:, 2])
                if sum_radii > best_sum_radii:
                    best_sum_radii = sum_radii
                    best_result = refined.copy()
            else:
                # Attempt repair if validation fails
                try:
                    repaired = self._repair_configuration(config)
                    if self.validate_placement(repaired):
                        sum_radii = np.sum(repaired[:, 2])
                        if sum_radii > best_sum_radii:
                            best_sum_radii = sum_radii
                            best_result = repaired.copy()
                except:
                    pass

        # If no valid configuration was found, use the fallback
        if best_result is None:
            best_result = self.generate_enhanced_voronoi_initialization()
            
        return best_result

    def _repair_configuration(self, circles: np.ndarray) -> np.ndarray:
        """Attempt to repair a configuration that failed validation."""
        # Make a copy to avoid modifying the original
        repaired = circles.copy()
        
        # First, ensure all circles are contained
        for i in range(len(repaired)):
            x, y, r = repaired[i]
            # Fix containment
            r = min(r, x, 1-x, y, 1-y)
            r = max(r, 0.005)
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            repaired[i] = [x, y, r]
            
        # Then, resolve overlaps by iterative adjustment
        max_iter = 100
        for _ in range(max_iter):
            any_changes = False
            for i in range(len(repaired)):
                for j in range(i+1, len(repaired)):
                    x1, y1, r1 = repaired[i]
                    x2, y2, r2 = repaired[j]
                    
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist_sq = (r1 + r2)**2
                    
                    if dist_sq < min_dist_sq:
                        # Resolve overlap by moving circles apart
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = np.sqrt(dist_sq)
                        
                        if distance > 1e-10:
                            # Normalize direction vector
                            nx = dx / distance
                            ny = dy / distance
                            
                            # Move circles apart
                            move_distance = (min_dist_sq**0.5 - distance) / 2.0
                            
                            # Apply movement
                            move_x = nx * move_distance
                            move_y = ny * move_distance
                            
                            repaired[i][0] -= move_x * 0.5
                            repaired[i][1] -= move_y * 0.5
                            repaired[j][0] += move_x * 0.5
                            repaired[j][1] += move_y * 0.5
                            
                            # Keep within bounds
                            repaired[i][0] = np.clip(repaired[i][0], repaired[i][2], 1-repaired[i][2])
                            repaired[i][1] = np.clip(repaired[i][1], repaired[i][2], 1-repaired[i][2])
                            repaired[j][0] = np.clip(repaired[j][0], repaired[j][2], 1-repaired[j][2])
                            repaired[j][1] = np.clip(repaired[j][1], repaired[j][2], 1-repaired[j][2])
                            
                            any_changes = True
            
            if not any_changes:
                break
                
        return repaired

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses an enhanced physics-inspired optimization approach.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer(n_circles=26, seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END