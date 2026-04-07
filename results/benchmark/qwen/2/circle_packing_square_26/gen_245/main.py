# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import random
import math
from typing import Tuple, List, Optional
import time

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

class HexagonEvolve:
    def __init__(self, n_circles: int = 26):
        self.n_circles = n_circles
        self.max_iterations = 1000
        
    def create_hexagonal_grid(self, spacing_factor: float = 1.0) -> np.ndarray:
        """
        Create initial configuration using optimized hexagonal packing pattern.
        This method creates a more sophisticated hexagonal arrangement than simple grids.
        """
        circles = np.zeros((self.n_circles, 3))
        
        # Calculate optimal grid size for hexagonal packing
        rows = int(math.ceil(math.sqrt(self.n_circles)))
        cols = int(math.ceil(self.n_circles / rows))
        
        # Optimal hexagonal spacing (based on circle packing theory)
        hex_spacing = 1.0 / max(rows, cols)
        radius = hex_spacing * 0.35 * spacing_factor
        
        # Create hexagonal pattern with proper offsetting
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= self.n_circles:
                    break
                    
                # Hexagonal offset pattern
                offset = 0 if i % 2 == 0 else 0.5
                x = (j + offset) * hex_spacing
                y = i * hex_spacing * math.sqrt(3)/2
                
                # Ensure within bounds with safety margin
                x = max(radius, min(1-radius, x))
                y = max(radius, min(1-radius, y))
                
                circles[count] = [x, y, radius]
                count += 1
                
            if count >= self.n_circles:
                break
                
        return circles[:count]
    
    def initialize_with_multiple_strategies(self) -> np.ndarray:
        """
        Initialize with multiple strategies to capture diverse good starting points.
        """
        strategies = []
        
        # Strategy 1: Standard hexagonal grid
        s1 = self.create_hexagonal_grid(0.85)
        strategies.append(s1)
        
        # Strategy 2: Slighly denser hexagonal grid
        s2 = self.create_hexagonal_grid(0.9)
        strategies.append(s2)
        
        # Strategy 3: Sparse hexagonal grid with large radii
        s3 = self.create_hexagonal_grid(0.7)
        strategies.append(s3)
        
        # Strategy 4: Randomized hexagonal approach
        s4 = self.create_hexagonal_grid(0.95)
        # Add some randomness to positions
        for i in range(len(s4)):
            s4[i, 0] += np.random.uniform(-0.01, 0.01)
            s4[i, 1] += np.random.uniform(-0.01, 0.01)
            s4[i, 0] = np.clip(s4[i, 0], s4[i, 2], 1 - s4[i, 2])
            s4[i, 1] = np.clip(s4[i, 1], s4[i, 2], 1 - s4[i, 2])
        strategies.append(s4)
        
        # Select the best initialized configuration
        best_fitness = -1
        best_config = None
        
        for strategy in strategies:
            if len(strategy) < self.n_circles:
                # Pad with random circles
                extended = np.zeros((self.n_circles, 3))
                extended[:len(strategy)] = strategy
                for i in range(len(strategy), self.n_circles):
                    extended[i] = [np.random.uniform(0.05, 0.95), 
                                  np.random.uniform(0.05, 0.95), 
                                  np.random.uniform(0.005, 0.05)]
                strategy = extended
                
            # Calculate fitness (sum of radii) for this strategy
            fitness = np.sum(strategy[:, 2])
            if fitness > best_fitness:
                best_fitness = fitness
                best_config = strategy.copy()
                
        return best_config
    
    def validate_circles(self, circles: np.ndarray) -> bool:
        """Validate that circles are within bounds and non-overlapping."""
        if len(circles) != self.n_circles:
            return False
            
        # Check containment constraints
        for i in range(self.n_circles):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1 - r or y < r or y > 1 - r:
                return False
        
        # Check overlap constraints using KDTree for efficiency
        points = circles[:, :2]
        if len(points) > 1:
            tree = KDTree(points)
            
            for i in range(self.n_circles):
                x, y, r = circles[i]
                # Find nearby circles (within 2*r distance)
                nearby = tree.query_ball_point([x, y], 2 * r)
                for j in nearby:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if distance < r + r2:
                            return False
        
        return True
    
    def calculate_fitness(self, circles: np.ndarray) -> float:
        """Calculate total radius sum as fitness."""
        return np.sum(circles[:, 2])
    
    def compute_forces(self, circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attractive and repulsive forces for all circles.
        Attractive to center, repulsive from other circles.
        """
        forces = np.zeros((len(circles), 2))
        center = np.array([0.5, 0.5])
        
        # Attractive force towards center (stronger for outer circles)
        for i in range(len(circles)):
            x, y, r = circles[i]
            vec_to_center = center - np.array([x, y])
            distance_to_center = np.linalg.norm(vec_to_center)
            
            if distance_to_center > 0.01:  # Avoid division by zero
                # Stronger attraction for circles farther from center
                attraction_strength = 0.005 * (distance_to_center / 0.5)
                forces[i] += attraction_strength * vec_to_center / distance_to_center
        
        # Repulsive forces from other circles
        points = circles[:, :2]
        if len(points) > 1:
            tree = KDTree(points)
            
            for i in range(len(circles)):
                x, y, r = circles[i]
                # Query nearby circles within repulsion distance
                nearby = tree.query_ball_point([x, y], 2 * (r + 0.001))
                
                for j in nearby:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        
                        if distance < r + r2 + 0.001:  # Overlapping or nearly overlapping
                            # Repulsive force
                            repulsion_vec = np.array([x, y]) - np.array([x2, y2])
                            if distance > 0.001:
                                force_magnitude = 0.01 * (r + r2 - distance) / distance
                                forces[i] += force_magnitude * repulsion_vec / distance
        
        return forces
    
    def apply_forces(self, circles: np.ndarray, forces: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Apply forces to update positions."""
        updated = circles.copy()
        
        for i in range(len(updated)):
            x, y, r = updated[i]
            fx, fy = forces[i]
            
            # Apply forces with damping
            new_x = x + fx * dt
            new_y = y + fy * dt
            
            # Constrain to boundaries with safety margin
            new_x = np.clip(new_x, r, 1 - r)
            new_y = np.clip(new_y, r, 1 - r)
            
            updated[i] = [new_x, new_y, r]
            
        return updated
    
    def expand_radii(self, circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """
        Increase radii systematically while maintaining validity.
        This is the key insight - instead of random mutations, we greedily expand.
        """
        expanded = circles.copy()
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try to expand each circle's radius
            for i in range(len(expanded)):
                original_r = expanded[i, 2]
                x, y, _ = expanded[i]
                
                # Calculate maximum possible radius
                max_radius = min(x, 1-x, y, 1-y) * 0.95  # Leave some margin
                if max_radius <= original_r:
                    continue
                    
                # Binary search for largest possible radius
                low = original_r
                high = max_radius
                best_radius = original_r
                
                # Check if we can increase radius
                while high - low > 1e-6:
                    mid = (low + high) / 2
                    test_circles = expanded.copy()
                    test_circles[i, 2] = mid
                    
                    # Test validity
                    if self.validate_circles(test_circles):
                        best_radius = mid
                        low = mid
                    else:
                        high = mid
                
                if best_radius > original_r + 1e-6:
                    expanded[i, 2] = best_radius
                    improved = True
        
        return expanded
    
    def resolve_overlaps(self, circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """
        Resolve overlaps by adjusting positions and radii systematically.
        """
        resolved = circles.copy()
        
        # Iteratively resolve overlaps
        for _ in range(max_iterations):
            any_changes = False
            
            # Try to reduce radii of overlapping circles
            for i in range(len(resolved)):
                x1, y1, r1 = resolved[i]
                
                # Check overlap with all other circles
                for j in range(len(resolved)):
                    if i != j:
                        x2, y2, r2 = resolved[j]
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        min_distance = r1 + r2
                        
                        if distance < min_distance:
                            # We need to reduce one of the radii
                            reduction = (min_distance - distance) * 0.3
                            
                            # Prefer reducing the smaller circle's radius
                            if r1 <= r2:
                                new_r1 = max(0.001, r1 - reduction)
                                if new_r1 < r1:
                                    resolved[i, 2] = new_r1
                                    any_changes = True
                            else:
                                new_r2 = max(0.001, r2 - reduction)
                                if new_r2 < r2:
                                    resolved[j, 2] = new_r2
                                    any_changes = True
            
            if not any_changes:
                break
        
        return resolved
    
    def optimize_step(self, circles: np.ndarray) -> np.ndarray:
        """Perform a single optimization step combining multiple techniques."""
        # Phase 1: Apply physical forces
        forces = self.compute_forces(circles)
        moved = self.apply_forces(circles, forces)
        
        # Phase 2: Expand radii
        expanded = self.expand_radii(moved, max_iterations=30)
        
        # Phase 3: Resolve any remaining overlaps
        resolved = self.resolve_overlaps(expanded)
        
        return resolved
    
    def optimize_circles(self) -> np.ndarray:
        """Main optimization routine using hybrid approach."""
        # Step 1: Initialize with multiple strategies
        circles = self.initialize_with_multiple_strategies()
        
        # Step 2: Apply iterative optimization
        best_fitness = self.calculate_fitness(circles)
        best_circles = circles.copy()
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Apply optimization step
            circles = self.optimize_step(circles)
            
            # Check if this is better
            current_fitness = self.calculate_fitness(circles)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_circles = circles.copy()
            
            # Early termination if no significant improvement
            if iteration > 10 and abs(current_fitness - best_fitness) < 1e-6:
                break
        
        # Final refinement
        final_circles = self.resolve_overlaps(best_circles)
        final_circles = self.expand_radii(final_circles, max_iterations=50)
        
        # Ensure final validation
        if not self.validate_circles(final_circles):
            # Fallback to simple valid configuration
            final_circles = np.zeros((self.n_circles, 3))
            # Use a simple but effective grid arrangement
            grid_size = int(np.ceil(np.sqrt(self.n_circles)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            radius = spacing_x * 0.3
            
            count = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if count >= self.n_circles:
                        break
                    x = (i + 1) * spacing_x
                    y = (j + 1) * spacing_y
                    # Add some small variation
                    x += np.random.uniform(-spacing_x/10, spacing_x/10)
                    y += np.random.uniform(-spacing_y/10, spacing_y/10)
                    final_circles[count] = [x, y, radius]
                    count += 1
                if count >= self.n_circles:
                    break
        
        return final_circles
    
    def optimize_with_local_search(self) -> np.ndarray:
        """Perform local search refinement to further improve solution."""
        # Get initial solution
        circles = self.optimize_circles()
        
        # Local refinement with hill climbing
        current_fitness = self.calculate_fitness(circles)
        improved = True
        max_no_improvement = 50
        no_improvement_count = 0
        
        while improved and no_improvement_count < max_no_improvement:
            improved = False
            
            # Try small position tweaks
            for i in range(len(circles)):
                if np.random.random() < 0.3:  # 30% chance to tweak each circle
                    original_x, original_y, original_r = circles[i]
                    
                    # Try small movement in random direction
                    delta_x = np.random.uniform(-0.005, 0.005)
                    delta_y = np.random.uniform(-0.005, 0.005)
                    
                    new_x = original_x + delta_x
                    new_y = original_y + delta_y
                    
                    # Clip to valid bounds
                    new_x = np.clip(new_x, original_r, 1 - original_r)
                    new_y = np.clip(new_y, original_r, 1 - original_r)
                    
                    # Test this change
                    test_circles = circles.copy()
                    test_circles[i, 0] = new_x
                    test_circles[i, 1] = new_y
                    
                    if self.validate_circles(test_circles):
                        test_fitness = self.calculate_fitness(test_circles)
                        if test_fitness > current_fitness:
                            circles = test_circles
                            current_fitness = test_fitness
                            improved = True
                            no_improvement_count = 0
                            break
            
            if not improved:
                no_improvement_count += 1
                
        return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        # Use the HexagonEvolve optimizer
        optimizer = HexagonEvolve(n_circles=26)
        circles = optimizer.optimize_with_local_search()
        
        # Final validation
        if not optimizer.validate_circles(circles):
            # Create a valid fallback
            circles = np.zeros((26, 3))
            grid_size = int(np.ceil(np.sqrt(26)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            radius = spacing_x * 0.3
            
            count = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if count >= 26:
                        break
                    x = (i + 1) * spacing_x
                    y = (j + 1) * spacing_y
                    x += np.random.uniform(-spacing_x/10, spacing_x/10)
                    y += np.random.uniform(-spacing_y/10, spacing_y/10)
                    circles[count] = [x, y, radius]
                    count += 1
                if count >= 26:
                    break
                    
        return circles
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Fallback to simple initialization
        circles = np.zeros((26, 3))
        grid_size = int(np.ceil(np.sqrt(26)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        radius = spacing_x * 0.3
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                x += np.random.uniform(-spacing_x/10, spacing_x/10)
                y += np.random.uniform(-spacing_y/10, spacing_y/10)
                circles[count] = [x, y, radius]
                count += 1
            if count >= 26:
                break
                
        return circles

# EVOLVE-BLOCK-END