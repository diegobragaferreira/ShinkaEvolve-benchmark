# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform


class PointDispersionOptimizer:
    """Optimizes point placement to maximize min/max distance ratio using simulated annealing."""
    
    def __init__(self, num_points=16, dimension=2):
        self.num_points = num_points
        self.dimension = dimension
        self.best_points = None
        self.best_ratio = 0.0
        self.iteration_count = 0
        
    def compute_min_max_ratio(self, points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def compute_min_max_ratio_with_boundary_penalty(self, points, penalty_factor=1e6):
        """Compute ratio with penalty for points near boundaries."""
        # Check if any point is too close to boundary
        boundary_penalties = []
        for point in points:
            min_dist_to_boundaries = min(point[0], 1-point[0], point[1], 1-point[1])
            if min_dist_to_boundaries < 0.01:
                boundary_penalties.append(min_dist_to_boundaries)

        # Compute base ratio
        ratio = self.compute_min_max_ratio(points)

        # Apply penalty if needed
        if boundary_penalties:
            penalty = penalty_factor * sum(boundary_penalties)
            return ratio - penalty

        return ratio
    
    def initialize_hexagonal_points(self):
        """Initialize points using a controlled hexagonal arrangement with deliberate asymmetry."""
        points = np.zeros((self.num_points, self.dimension))
        
        # Create a hexagonal-like grid with irregularities
        rows, cols = 4, 4
        row_spacing = 1.0 / (rows - 1) if rows > 1 else 0.25
        col_spacing = np.sqrt(3) / 2.0 * row_spacing
        
        # Generate base hexagonal pattern
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= self.num_points:
                    break
                    
                # Base position with alternating offset
                x = col * row_spacing + (row % 2) * row_spacing * 0.5
                y = row * col_spacing
                
                # Add controlled noise to break symmetry
                noise_strength = 0.01
                x += np.random.normal(0, noise_strength * 0.5)
                y += np.random.normal(0, noise_strength * 0.5)
                
                points[idx, 0] = x
                points[idx, 1] = y
                idx += 1
                
        # Normalize to [0,1] x [0,1] while preserving relative positions
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
            
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        
        # Apply additional asymmetric perturbations to key points
        # These ensure breaking of any remaining symmetries
        asymmetry_indices = [0, 3, 5, 9, 12, 15]  # Corner and center points
        for idx in asymmetry_indices:
            if idx < len(points):
                points[idx] += np.random.uniform(-0.015, 0.015, 2)
                points[idx] = np.clip(points[idx], 0, 1)
        
        return points
    
    def perturb_points(self, points, magnitude=0.01, method='individual'):
        """Apply small perturbations to points with boundary checking."""
        new_points = points.copy()

        if method == 'individual':
            # Select random subset of points to perturb
            num_perturb = max(1, len(points) // 4)
            indices = np.random.choice(len(points), size=num_perturb, replace=False)

            for idx in indices:
                # Apply perturbation
                delta = np.random.uniform(-magnitude, magnitude, 2)
                new_points[idx] += delta

                # Keep within bounds
                new_points[idx] = np.clip(new_points[idx], 0, 1)
        elif method == 'cluster':
            # Perturb small clusters together to maintain local structure
            cluster_size = min(3, len(points) // 4)
            num_clusters = len(points) // cluster_size

            for i in range(num_clusters):
                start_idx = i * cluster_size
                end_idx = min(start_idx + cluster_size, len(points))

                # Find centroid of cluster
                cluster_center = np.mean(new_points[start_idx:end_idx], axis=0)

                # Apply same perturbation to whole cluster
                perturbation = np.random.uniform(-magnitude, magnitude, 2)
                for idx in range(start_idx, end_idx):
                    new_points[idx] += perturbation

                    # Keep within bounds
                    new_points[idx] = np.clip(new_points[idx], 0, 1)

        return new_points
    
    def adaptive_cooling_schedule(self, iteration, max_iterations, base_cooling_rate=0.9995):
        """Provide adaptive cooling based on progress."""
        # Start with a high temperature and cool based on iteration
        temperature = 0.1 * (base_cooling_rate ** iteration)
        
        # Accelerate cooling if we've made significant progress recently
        if iteration > max_iterations * 0.7:
            temperature *= (0.999 ** (iteration - max_iterations * 0.7))
            
        return max(temperature, 1e-6)
    
    def optimize(self, max_iterations=5000):
        """Run the simulated annealing optimization."""
        # Initialize points
        current_points = self.initialize_hexagonal_points()
        current_ratio = self.compute_min_max_ratio_with_boundary_penalty(current_points)
        
        self.best_points = current_points.copy()
        self.best_ratio = current_ratio
        
        # Tracking variables for adaptive cooling
        recent_improvements = []
        max_recent = 50
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Alternate perturbation methods
            perturbation_method = 'cluster' if iteration % 3 == 0 else 'individual'
            
            # Perturb points
            new_points = self.perturb_points(
                current_points, 
                magnitude=self.adaptive_cooling_schedule(iteration, max_iterations) * 0.5,
                method=perturbation_method
            )
            
            # Evaluate new configuration
            new_ratio = self.compute_min_max_ratio_with_boundary_penalty(new_points)
            
            # Accept or reject based on Metropolis criterion
            temperature = self.adaptive_cooling_schedule(iteration, max_iterations)
            
            if new_ratio > current_ratio or np.random.random() < np.exp((new_ratio - current_ratio) / temperature):
                current_points = new_points
                current_ratio = new_ratio
                
                # Update best solution
                if current_ratio > self.best_ratio:
                    self.best_points = current_points.copy()
                    self.best_ratio = current_ratio
                    recent_improvements = []  # Reset improvement tracking
                else:
                    # Track recent improvements
                    if len(recent_improvements) < max_recent:
                        recent_improvements.append(current_ratio)
                    else:
                        recent_improvements.pop(0)
                        recent_improvements.append(current_ratio)
            
            # Adaptive cooling based on recent improvement variance
            if len(recent_improvements) >= 10:
                recent_std = np.std(recent_improvements[-10:])
                if recent_std < 0.001 * self.best_ratio:  # Very little variation
                    # Slow down cooling to allow more exploration
                    pass  # Already handled by cooling schedule
            
            self.iteration_count = iteration
            
            # Early stopping if we're getting very good results
            if self.best_ratio > 0.3:  # Early exit threshold
                break
                
        return self.best_points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    # Create optimizer instance
    optimizer = PointDispersionOptimizer(num_points=16, dimension=2)
    
    # Run optimization
    result = optimizer.optimize(max_iterations=5000)
    
    return result


# EVOLVE-BLOCK-END