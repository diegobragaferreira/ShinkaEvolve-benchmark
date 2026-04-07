# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

class CirclePackState:
    """Encapsulates the complete state of a circle packing configuration."""
    def __init__(self, circles=None):
        self.circles = circles if circles is not None else np.zeros((0, 3))
        self.tree = None
        self.max_radius = 0.0
        
    def update_tree(self):
        """Update the spatial index tree for efficient neighbor queries."""
        if len(self.circles) > 0:
            circle_points = np.array([[c[0], c[1]] for c in self.circles])
            self.tree = cKDTree(circle_points)
            self.max_radius = np.max(self.circles[:, 2]) if len(self.circles) > 0 else 0.1
        else:
            self.tree = None
            self.max_radius = 0.1

class CircleValidator:
    """Handles all circle placement validation logic."""
    
    @staticmethod
    def is_valid_placement(state, x, y, r):
        """Check if placing a circle at (x,y) with radius r is valid."""
        # Check boundary constraints
        if r > x or r > y or r > (1-x) or r > (1-y):
            return False

        # Check overlap with existing circles using spatial index if available
        if state.tree is not None and state.max_radius > 0:
            # Find potential overlapping circles within distance 2*(r+max_radius)
            candidates = state.tree.query_ball_point([x, y], 2*(r + state.max_radius))
            for i in candidates:
                cx, cy, cr = state.circles[i]
                distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                if distance < (r + cr):
                    return False
        else:
            # Fallback to brute force for small number of circles
            for i in range(len(state.circles)):
                cx, cy, cr = state.circles[i]
                distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                if distance < (r + cr):
                    return False

        return True

class DensityAnalyzer:
    """Computes local density information for strategic placement."""
    
    @staticmethod
    def compute_local_density(circles, point, k=5):
        """Compute local density around a point using k nearest neighbors."""
        if len(circles) < 2:
            return 0

        # Convert circles to array for KDTree query
        circle_points = np.array([[c[0], c[1]] for c in circles])
        tree = cKDTree(circle_points)

        # Query k nearest neighbors
        dists, _ = tree.query(point, k=min(k, len(circle_points)))

        # Return average distance to nearest neighbors
        return np.mean(dists[dists > 0]) if np.any(dists > 0) else 1.0

class CircleInitializer:
    """Handles the initialization of circle positions."""
    
    @staticmethod
    def create_hexagonal_grid(n, spacing_factor=1.5):
        """Create an initial hexagonal grid pattern."""
        grid_size = int(np.ceil(np.sqrt(n * spacing_factor)))
        spacing = 1.0 / (grid_size + 1)
        circles = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(circles) >= n:
                    break
                # Create a hexagonal offset pattern
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                if i % 2 == 1:  # Offset every other row
                    x += spacing * 0.5
                    
                # Conservative initial radius
                r_min = min(x, y, 1-x, 1-y)
                r = min(r_min * 0.4, 0.15)
                
                circles.append([x, y, r])
                
        return circles[:n]
    
    @staticmethod
    def fill_remaining_spots(state, n):
        """Fill remaining spots with additional circles using density-aware sampling."""
        circles = state.circles.tolist()
        
        while len(circles) < n:
            best_r = 0
            best_x, best_y = 0, 0
            
            # Sample potential positions with better distribution
            for _ in range(2000):
                # Adaptive sampling: prioritize corners and center areas
                if np.random.random() < 0.7:  # 70% regular sampling
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.uniform(0.05, 0.95)
                else:  # 30% concentrated sampling
                    corner_probs = [0.25, 0.25, 0.25, 0.25]
                    corner_idx = np.random.choice(4, p=corner_probs)
                    if corner_idx == 0:
                        x = np.random.uniform(0.05, 0.3)
                        y = np.random.uniform(0.05, 0.3)
                    elif corner_idx == 1:
                        x = np.random.uniform(0.7, 0.95)
                        y = np.random.uniform(0.05, 0.3)
                    elif corner_idx == 2:
                        x = np.random.uniform(0.05, 0.3)
                        y = np.random.uniform(0.7, 0.95)
                    else:
                        x = np.random.uniform(0.7, 0.95)
                        y = np.random.uniform(0.7, 0.95)

                # Estimate max radius at this location
                r_max = min(x, y, 1-x, 1-y)
                if r_max <= 0:
                    continue

                # Better density estimation with multiple samples
                sample_density = 0
                sample_count = 10
                for _ in range(sample_count):
                    px = np.random.uniform(max(0.05, x-0.1), min(0.95, x+0.1))
                    py = np.random.uniform(max(0.05, y-0.1), min(0.95, y+0.1))
                    point = np.array([px, py])
                    density = DensityAnalyzer.compute_local_density(circles, point)
                    sample_density += density
                avg_density = sample_density / sample_count if sample_count > 0 else 0

                # Adjust radius based on density
                r_factor = max(0.1, 1.0 / (1.0 + avg_density * 10))
                adjusted_r_max = r_max * r_factor

                # Try different radii with better spacing
                test_radii = np.linspace(0.005, adjusted_r_max * 0.7, 15)
                for r in test_radii:
                    if CircleValidator.is_valid_placement(state, x, y, r):
                        if r > best_r:
                            best_r = r
                            best_x, best_y = x, y
                            break

            if best_r > 0:
                circles.append([best_x, best_y, best_r])
            else:
                # If we couldn't place another circle, try reducing the circle count
                if len(circles) > 0:
                    circles = circles[:-1]
                else:
                    break
                    
        return circles[:n]

class CirclePacker:
    """Main circle packing orchestrator."""
    
    def __init__(self):
        self.validator = CircleValidator()
        self.density_analyzer = DensityAnalyzer()
        self.initializer = CircleInitializer()
        
    def create_initial_pack(self, n=32):
        """Create initial circle packing configuration."""
        # Initialize with hexagonal grid
        grid_circles = self.initializer.create_hexagonal_grid(n)
        state = CirclePackState(np.array(grid_circles))
        state.update_tree()
        
        # Fill remaining spots
        final_circles = self.initializer.fill_remaining_spots(state, n)
        state.circles = np.array(final_circles)
        state.update_tree()
        
        return state
    
    def optimize_pack(self, state, max_iter=1000):
        """Apply local optimization to improve the configuration."""
        circles = state.circles.copy()
        
        # Store the best solution found so far
        best_circles = circles.copy()
        best_sum = np.sum(best_circles[:, 2])
        
        for iteration in range(max_iter):
            # Try to improve by adjusting one circle at a time
            for i in range(len(circles)):
                cx, cy, cr = circles[i]
                
                # Save original values
                orig_x, orig_y, orig_r = cx, cy, cr
                
                # Try small perturbations
                for _ in range(100):
                    # Generate random perturbation
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    dr = np.random.uniform(-0.005, 0.005)
                    
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    new_r = orig_r + dr
                    
                    # Ensure new_r is positive
                    if new_r <= 0:
                        continue
                        
                    # Ensure new position is inside the square
                    if new_r > new_x or new_r > new_y or new_r > (1-new_x) or new_r > (1-new_y):
                        continue
                        
                    # Check if new placement is valid
                    if self.validator.is_valid_placement(state, new_x, new_y, new_r):
                        # Temporarily update this circle
                        circles[i] = [new_x, new_y, new_r]
                        
                        # Check if this gives a better total radius
                        new_sum = np.sum(circles[:, 2])
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_circles = circles.copy()
                        else:
                            # Revert the change
                            circles[i] = [orig_x, orig_y, orig_r]
            
            # Update circles to best found so far
            circles[:] = best_circles[:]
            state.circles = circles.copy()
            state.update_tree()
            
        return state

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    packer = CirclePacker()
    
    # Create initial pack
    state = packer.create_initial_pack(32)
    
    # Apply optimization
    optimized_state = packer.optimize_pack(state)
    
    return optimized_state.circles

# EVOLVE-BLOCK-END