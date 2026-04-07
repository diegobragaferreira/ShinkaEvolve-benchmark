# EVOLVE-BLOCK-START
import numpy as np
import torch
import time
from scipy.spatial.distance import pdist

class SphereTorchOptimizer:
    """Physics-inspired optimizer using PyTorch for 3D point dispersion optimization."""
    
    def __init__(self, num_points=14, device='cpu', max_iterations=5000):
        self.num_points = num_points
        self.device = device
        self.max_iterations = max_iterations
        self.learning_rate = 0.01
        self.momentum = 0.9
        
    def fibonacci_sphere(self, samples=14):
        """Generate points on sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    def initialize_points(self):
        """Initialize points using spherical Fibonacci with perturbations."""
        # Generate initial points on sphere
        initial_points = self.fibonacci_sphere(self.num_points)
        
        # Scale to unit cube [0,1]^3
        initial_points = initial_points - np.mean(initial_points, axis=0)
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / (2 * max_coord) + 0.5
            
        # Add small random perturbations
        initial_points += np.random.normal(0, 0.01, initial_points.shape)
        initial_points = np.clip(initial_points, 0, 1)
        
        return initial_points
    
    def calculate_ratio(self, points):
        """Calculate min/max distance ratio with robust error handling."""
        if len(points) < 2:
            return 0.0
        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0
            # Filter out invalid distances
            finite_distances = distances[np.isfinite(distances)]
            if len(finite_distances) == 0:
                return 0.0
            d_min = np.min(finite_distances)
            d_max = np.max(finite_distances)
            if d_max <= 0:
                return 0.0
            return d_min / d_max
        except:
            return 0.0
    
    def run_physics_simulation(self, initial_points):
        """Run particle-based physics simulation with PyTorch."""
        # Convert to PyTorch tensor
        points_tensor = torch.tensor(initial_points, dtype=torch.float32, requires_grad=True, device=self.device)
        
        # Initialize velocity (momentum)
        velocity = torch.zeros_like(points_tensor, device=self.device)
        
        # Optimizer parameters
        base_lr = self.learning_rate
        momentum = self.momentum
        best_points = points_tensor.clone().detach().cpu().numpy()
        best_ratio = self.calculate_ratio(best_points)
        
        optimizer = torch.optim.SGD([points_tensor], lr=base_lr, momentum=momentum)
        
        # Physics simulation loop
        for iteration in range(self.max_iterations):
            # Clear gradients
            optimizer.zero_grad()
            
            # Calculate pairwise distances using PyTorch
            # Expand dimensions for broadcasting
            expanded_i = points_tensor.unsqueeze(1)  # (n, 1, 3)
            expanded_j = points_tensor.unsqueeze(0)  # (1, n, 3)
            
            # Compute squared differences
            diff = expanded_i - expanded_j  # (n, n, 3)
            
            # Compute squared distances
            sq_distances = torch.sum(diff**2, dim=2)  # (n, n)
            
            # Avoid self-interactions (diagonal should be zero)
            diag_mask = torch.eye(self.num_points, dtype=torch.bool, device=self.device)
            sq_distances = sq_distances.masked_fill(diag_mask, float('inf'))
            
            # Compute actual distances
            distances = torch.sqrt(sq_distances)
            
            # Calculate min and max distances
            d_min = torch.min(distances)
            d_max = torch.max(distances)
            
            # Avoid division by zero
            if d_max.item() <= 0:
                loss = torch.tensor(float('inf'), device=self.device)
            else:
                # Minimize negative ratio (maximize ratio)
                ratio = d_min / d_max
                loss = -ratio  # We want to maximize ratio, so minimize negative ratio
            
            # Add boundary constraints as penalty terms
            penalty = torch.tensor(0.0, device=self.device)
            
            # Penalty for points going below 0
            below_zero = torch.relu(-points_tensor)
            penalty += torch.sum(below_zero**2)
            
            # Penalty for points going above 1
            above_one = torch.relu(points_tensor - 1)
            penalty += torch.sum(above_one**2)
            
            # Total loss with penalty
            total_loss = loss + 100.0 * penalty
            
            # Backward pass
            total_loss.backward()
            
            # Update points
            optimizer.step()
            
            # Project points back to valid domain [0,1]^3
            with torch.no_grad():
                points_tensor.clamp_(0, 1)
            
            # Periodic best solution saving
            if iteration % 100 == 0:
                current_points = points_tensor.detach().cpu().numpy()
                current_ratio = self.calculate_ratio(current_points)
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
            
            # Early stopping based on improvement
            if iteration > 1000 and iteration % 500 == 0:
                # Check recent improvement
                recent_points = points_tensor.detach().cpu().numpy()
                recent_ratio = self.calculate_ratio(recent_points)
                if abs(recent_ratio - best_ratio) < 1e-8:
                    break
        
        return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize optimizer
    optimizer = SphereTorchOptimizer(num_points=14, device='cpu', max_iterations=3000)
    
    # Set time limit
    start_time = time.time()
    time_limit = 340  # seconds
    
    # Generate initial configuration
    initial_points = optimizer.initialize_points()
    
    # Run physics-based optimization
    optimized_points = optimizer.run_physics_simulation(initial_points)
    
    # Additional refinement with local optimization
    if time.time() - start_time < time_limit - 10:
        # Simple gradient-based refinement
        try:
            points_tensor = torch.tensor(optimized_points, dtype=torch.float32, requires_grad=True, device='cpu')
            optimizer_refine = torch.optim.Adam([points_tensor], lr=0.001)
            
            for _ in range(1000):
                optimizer_refine.zero_grad()
                
                # Distance calculations
                expanded_i = points_tensor.unsqueeze(1)
                expanded_j = points_tensor.unsqueeze(0)
                diff = expanded_i - expanded_j
                sq_distances = torch.sum(diff**2, dim=2)
                diag_mask = torch.eye(14, dtype=torch.bool, device='cpu')
                sq_distances = sq_distances.masked_fill(diag_mask, float('inf'))
                distances = torch.sqrt(sq_distances)
                
                d_min = torch.min(distances)
                d_max = torch.max(distances)
                
                if d_max.item() <= 0:
                    loss = torch.tensor(float('inf'), device='cpu')
                else:
                    ratio = d_min / d_max
                    loss = -ratio
                
                # Boundary penalties
                penalty = torch.sum(torch.relu(-points_tensor)**2) + torch.sum(torch.relu(points_tensor - 1)**2)
                total_loss = loss + 1000.0 * penalty
                
                total_loss.backward()
                optimizer_refine.step()
                
                with torch.no_grad():
                    points_tensor.clamp_(0, 1)
            
            optimized_points = points_tensor.detach().cpu().numpy()
            
        except Exception as e:
            pass  # Continue with original result if refinement fails
    
    # Ensure all points are within [0,1]^3
    optimized_points = np.clip(optimized_points, 0, 1)
    
    # Final validation
    if optimizer.calculate_ratio(optimized_points) <= 0:
        # Return original initialization if something failed
        return initial_points
    
    return optimized_points

# EVOLVE-BLOCK-END