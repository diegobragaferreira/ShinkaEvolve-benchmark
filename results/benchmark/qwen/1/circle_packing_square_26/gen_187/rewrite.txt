# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List, Optional
import time

class CirclePackingOptimizer:
    """Main optimizer class for 26-circle packing problem."""
    
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
        
        # More efficient query using a reasonable search radius
        if n > 100:
            # For large numbers of circles, use a smarter approach
            pairs = tree.query_pairs(0.05, output_type='ndarray')
        else:
            # For smaller numbers, use a more direct approach
            pairs = tree.query_pairs(2 * min(circles[:, 2]), output_type='ndarray')

        for i, j in pairs:
            if i < j:  # Avoid double-checking
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                if distance_sq < min_distance_sq:
                    return False

        return True

    def generate_voronoi_initialization(self) -> np.ndarray:
        """Generate initial circle positions using an enhanced Voronoi-inspired spreading mechanism."""
        # Create a grid of candidate positions with improved distribution
        grid_size = max(4, int(np.ceil(np.sqrt(self.n_circles * 1.2))))
        x_coords = np.linspace(0.05, 0.95, grid_size)
        y_coords = np.linspace(0.05, 0.95, grid_size)

        # Generate all grid points with strategic jittering
        grid_points = []
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                # Apply jittering with grid-dependent pattern
                jitter_x = np.random.uniform(-0.02, 0.02) * (1.0 + 0.1 * np.sin(i * 0.5))
                jitter_y = np.random.uniform(-0.02, 0.02) * (1.0 + 0.1 * np.cos(j * 0.5))
                grid_points.append([x + jitter_x, y + jitter_y])

        # If we have more circles than grid points, add some random points
        if len(grid_points) < self.n_circles:
            extra_points = self.n_circles - len(grid_points)
            for _ in range(extra_points):
                grid_points.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])

        # Shuffle the points to avoid systematic bias
        random.shuffle(grid_points)

        # Take the first n_circles points
        points = np.array(grid_points[:self.n_circles])

        # Initialize circles with calculated initial radii based on position density
        circles = np.zeros((self.n_circles, 3))
        circles[:, 0] = points[:, 0]  # x coordinates
        circles[:, 1] = points[:, 1]  # y coordinates
        
        # Estimate initial radii based on expected density
        avg_density = self.n_circles / (0.9 * 0.9)  # Expected density in unit square
        estimated_radius = min(0.1, 0.5 / np.sqrt(avg_density))
        circles[:, 2] = max(0.01, estimated_radius)  # Initial small radii

        return circles

    def generate_greedy_fallback(self) -> np.ndarray:
        """Improved fallback method to generate a feasible configuration."""
        # Better greedy placement with proper collision detection
        circles = np.zeros((self.n_circles, 3))
        positions = []
        radii = []
        
        # Start with a reasonable radius
        radius = 0.05
        
        placed = 0
        while placed < self.n_circles and radius > 0.005:
            # Try to place circles systematically in a grid pattern with randomness
            rows = int(np.ceil(np.sqrt(self.n_circles)))
            cols = int(np.ceil(self.n_circles / rows))
            
            for i in range(rows):
                for j in range(cols):
                    if placed >= self.n_circles:
                        break
                    # Randomize within cell to avoid systematic patterns
                    offset_x = np.random.uniform(-0.05, 0.05)
                    offset_y = np.random.uniform(-0.05, 0.05)
                    x = 0.1 + (j * 0.8 / max(1, cols-1)) + offset_x
                    y = 0.1 + (i * 0.8 / max(1, rows-1)) + offset_y
                    
                    # Ensure within bounds
                    x = max(0.05, min(0.95, x))
                    y = max(0.05, min(0.95, y))
                    
                    # Check if this position is valid
                    valid = True
                    for pos, rad in zip(positions, radii):
                        dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                        if dist_sq < (rad + radius)**2:
                            valid = False
                            break
                    
                    if valid and len(positions) < self.n_circles:
                        positions.append([x, y])
                        radii.append(radius)
                        placed += 1
            
            if placed < self.n_circles:
                radius *= 0.8  # Decrease radius more aggressively
        
        # Fill remaining circles with random positions
        while len(positions) < self.n_circles:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            positions.append([x, y])
            radii.append(0.01)

        circles[:, 0] = [pos[0] for pos in positions]
        circles[:, 1] = [pos[1] for pos in positions]
        circles[:, 2] = radii

        return circles

    def _compute_forces_optimized(self, circles: np.ndarray, 
                                 repulsion_strength: float = 100.0,
                                 boundary_strength: float = 50.0) -> Tuple[np.ndarray, float]:
        """
        Optimized force computation using spatial indexing and early termination.
        """
        n = len(circles)
        forces = np.zeros((n, 2))
        potential_energy = 0.0
        
        # Use spatial indexing for more efficient neighbor search
        points = circles[:, :2]
        tree = cKDTree(points)
        
        # Query neighbors within a reasonable distance for force computation
        # This significantly reduces computational cost for large systems
        max_force_distance = 0.2  # Only consider nearby interactions
        neighbors = tree.query_pairs(max_force_distance, output_type='ndarray')
        
        # Process neighbor pairs
        for i, j in neighbors:
            if i >= j:  # Avoid duplicate processing
                continue
                
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            
            dx = x1 - x2
            dy = y1 - y2
            distance_sq = dx*dx + dy*dy
            
            # Skip if too far apart (more efficient than original)
            if distance_sq > 4*(r1+r2)**2:
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

    def _update_circles_optimized(self, circles: np.ndarray, forces: np.ndarray, 
                                 dt: float = 0.01, max_velocity: float = 0.05) -> np.ndarray:
        """
        Optimized circle position updates with reduced computation.
        """
        updated_circles = circles.copy()
        
        # Batch update positions to reduce overhead
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

    def _adaptive_simulation_step_optimized(self, circles: np.ndarray, 
                                           target_energy: float = 1e-3,
                                           max_steps: int = 1000) -> np.ndarray:
        """
        Optimized adaptive simulation with smarter convergence detection.
        """
        # Parameters for adaptive dynamics
        dt = 0.01
        damping_factor = 0.95
        energy_threshold = target_energy
        
        # Track energy evolution with reduced history
        energy_history = []
        last_energy = float('inf')
        
        # Early termination thresholds
        convergence_count = 0
        min_convergence_steps = 20
        
        for step in range(max_steps):
            # Compute forces
            forces, current_energy = self._compute_forces_optimized(circles)
            
            # Check for convergence - more conservative approach
            if len(energy_history) > 5:
                recent_avg = np.mean(energy_history[-5:])
                if abs(recent_avg - current_energy) < energy_threshold * 10:
                    convergence_count += 1
                else:
                    convergence_count = 0
                    
                # Stop early if we've converged consistently
                if convergence_count > min_convergence_steps:
                    break
                    
            # Store energy for tracking
            energy_history.append(current_energy)
            
            # Update circles
            circles = self._update_circles_optimized(circles, forces, dt)
            
            # Apply damping to prevent oscillation
            if step % 10 == 0:
                dt *= damping_factor
            
            # Occasionally reset time step
            if step % 30 == 0:
                dt = max(0.001, dt * 0.95)
            
            # Update last energy for convergence check
            last_energy = current_energy
            
        return circles

    def _multi_scale_refinement_optimized(self, initial_circles: np.ndarray, 
                                         max_iterations: int = 200) -> np.ndarray:
        """
        Optimized multi-scale refinement with reduced number of steps.
        """
        circles = initial_circles.copy()
        
        # Phase 1: Coarse optimization - fewer steps since we're already decent
        circles = self._adaptive_simulation_step_optimized(
            circles, target_energy=1e-2, max_steps=200
        )
        
        # Phase 2: Medium optimization with moderate forces
        circles = self._adaptive_simulation_step_optimized(
            circles, target_energy=1e-3, max_steps=150
        )
        
        # Phase 3: Fine-tuning with tighter tolerance
        circles = self._adaptive_simulation_step_optimized(
            circles, target_energy=1e-4, max_steps=100
        )
        
        return circles

    def _constraint_aware_local_search_optimized(self, circles: np.ndarray, 
                                                max_iterations: int = 100) -> np.ndarray:
        """
        Optimized constraint-aware local search with focus on efficiency.
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
                
                # Early termination if no improvement possible
                if max_radius <= r + 0.001:
                    continue
                    
                # Find neighbors more efficiently
                neighbors = []
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = improved_circles[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        min_dist_sq = (r + r2)**2
                        if dist_sq < min_dist_sq * 1.1:  # Allow some buffer
                            neighbors.append((j, dist_sq, min_dist_sq))
                
                # Binary search for maximum safe radius
                low, high = r, max_radius
                best_radius = r
                
                # Limited binary search iterations for efficiency
                for _ in range(15):
                    if abs(high - low) < 1e-6:
                        break
                    mid = (low + high) / 2
                    valid = True
                    
                    # Check overlap constraints quickly
                    for j, dist_sq, min_dist_sq in neighbors:
                        x2, y2, r2 = improved_circles[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        if dist_sq < (mid + r2)**2:
                            valid = False
                            break
                    
                    if valid:
                        best_radius = mid
                        low = mid
                    else:
                        high = mid
                
                # Apply improvement if beneficial
                if best_radius > r + 1e-6:
                    improved_circles[i, 2] = best_radius
                    improved = True
            
            # If no improvements from radius increases, try position adjustments
            if not improved:
                # Process only a subset of circles for efficiency
                circle_indices = list(range(n))  
                random.shuffle(circle_indices)
                
                for i in circle_indices[:max(n//4, 1)]:  # Process about 25% randomly
                    x, y, r = improved_circles[i]
                    
                    # Try small movements in 4 directions for speed
                    movements = [(-0.005, 0), (0.005, 0), (0, -0.005), (0, 0.005)]
                    
                    best_x, best_y = x, y
                    best_score = -float('inf')
                    best_radius = r
                    
                    for dx, dy in movements:
                        new_x, new_y = x + dx, y + dy
                        
                        # Check bounds
                        if new_x - r < 0 or new_x + r > 1 or new_y - r < 0 or new_y + r > 1:
                            continue
                            
                        # Quick overlap check with fewest neighbors
                        overlap_penalty = 0
                        valid = True
                        
                        # Only check with a few closest neighbors for efficiency
                        points = improved_circles[:, :2]
                        distances = np.sqrt(np.sum((points - [new_x, new_y])**2, axis=1))
                        closest_indices = np.argsort(distances)[:min(5, len(distances))]
                        
                        for j in closest_indices:
                            if i != j:
                                x2, y2, r2 = improved_circles[j]
                                dist_sq = (new_x - x2)**2 + (new_y - y2)**2
                                min_dist_sq = (r + r2)**2
                                if dist_sq < min_dist_sq:
                                    overlap_penalty += (min_dist_sq - dist_sq) * 1000
                                    valid = False
                        
                        if valid:
                            # Simple scoring based on radius
                            score = r - overlap_penalty * 0.0001
                            if score > best_score:
                                best_score = score
                                best_x, best_y = new_x, new_y
                    
                    # Apply the best movement if it helps
                    if best_x != x or best_y != y:
                        improved_circles[i, 0] = best_x
                        improved_circles[i, 1] = best_y
                        improved = True
            
            # Exit early if no improvements
            if not improved:
                break
        
        return improved_circles

    def _improve_with_gradient_descent_optimized(self, circles: np.ndarray, 
                                               max_iter: int = 50) -> np.ndarray:
        """
        Optimized gradient-based improvement with smarter radius adjustments.
        """
        improved_circles = circles.copy()
        n = len(improved_circles)
        
        for iteration in range(max_iter):
            updated = False
            
            # Process circles in batches for parallel-like efficiency
            batch_size = max(1, n // 4)
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices[:batch_size]:
                x, y, r = improved_circles[i]
                max_radius = min(x, 1-x, y, 1-y)
                
                # Quick check - only proceed if there's potential for improvement
                if max_radius <= r + 0.001:
                    continue
                
                # Find closest neighbors for constraint checking
                points = improved_circles[:, :2]
                distances = np.sqrt(np.sum((points - [x, y])**2, axis=1))
                closest_indices = np.argsort(distances)[1:min(6, len(distances))]  # Up to 5 neighbors
                
                # Check overlap constraints efficiently
                valid = True
                current_max_radius = max_radius
                
                for j in closest_indices:
                    x2, y2, r2 = improved_circles[j]
                    dist_sq = (x - x2)**2 + (y - y2)**2
                    if dist_sq < (r + r2)**2:
                        valid = False
                        break
                
                if valid:
                    # Try to increase radius
                    if r < max_radius - 0.001:
                        test_radius = min(max_radius, r + 0.005)
                        # Quick verification
                        valid_test = True
                        for j in closest_indices:
                            x2, y2, r2 = improved_circles[j]
                            dist_sq = (x - x2)**2 + (y - y2)**2
                            if dist_sq < (test_radius + r2)**2:
                                valid_test = False
                                break
                        
                        if valid_test:
                            improved_circles[i, 2] = test_radius
                            updated = True
                else:
                    # Reduce radius to resolve overlap
                    safe_radius = max(0.001, min(r, max_radius))
                    if safe_radius < r - 1e-6:
                        improved_circles[i, 2] = safe_radius
                        updated = True
            
            if not updated:
                break
        
        return improved_circles

    def optimize(self) -> np.ndarray:
        """Main optimization pipeline with optimizations."""
        # Step 1: Generate initial configuration using enhanced method
        circles = self.generate_voronoi_initialization()
        
        # Step 2: Multi-scale refinement with optimized steps
        circles = self._multi_scale_refinement_optimized(circles, max_iterations=150)
        
        # Step 3: Apply optimized constraint-aware local search
        circles = self._constraint_aware_local_search_optimized(circles, max_iterations=100)
        
        # Step 4: Additional optimized gradient-based refinement
        circles = self._improve_with_gradient_descent_optimized(circles, max_iter=50)
        
        # Step 5: Final validation and fallback if needed
        if not self.validate_placement(circles):
            # Try improved fallback
            circles = self.generate_greedy_fallback()
            
            # If still invalid, use Voronoi init
            if not self.validate_placement(circles):
                circles = self.generate_voronoi_initialization()
        
        return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses an optimized physics-inspired approach with enhanced computational efficiency.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer(n_circles=26, seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END