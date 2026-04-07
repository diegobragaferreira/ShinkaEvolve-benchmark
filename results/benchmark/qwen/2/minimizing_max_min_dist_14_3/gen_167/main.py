# EVOLVE-BLOCK-START
import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Novel Voronoi-based neural network optimization using PyTorch for gradient computation.
    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def initialize_fibonacci_sphere(n):
        """Better fibonacci-based sphere initialization"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def voronoi_entropy_score(points):
        """
        Calculate entropy-based score of Voronoi cell distribution.
        High entropy indicates more uniform cell distribution.
        """
        try:
            sv = SphericalVoronoi(points)
            areas = sv.calculate_areas()
            # Normalize areas
            areas = areas / np.sum(areas)
            # Entropy calculation
            entropy = -np.sum(areas * np.log(areas + 1e-10))
            return entropy
        except:
            # Fallback for cases where SphericalVoronoi fails
            return 0.0

    def project_to_unit_cube(points):
        """Project points to unit cube [0,1]^3"""
        # Find min/max along each axis
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Handle case where there's no variation
        ranges = max_coords - min_coords
        if np.any(ranges == 0):
            # If any dimension has no variation, return points centered at 0.5
            return np.full_like(points, 0.5)

        # Scale to [0,1] range
        normalized = (points - min_coords) / ranges

        # Ensure they're clipped to [0,1]
        return np.clip(normalized, 0, 1)

    def torch_project_to_sphere(points_tensor):
        """Project points to unit sphere using PyTorch"""
        norms = torch.norm(points_tensor, dim=1, keepdim=True)
        # Avoid division by zero
        norms = torch.where(norms > 1e-10, norms, torch.ones_like(norms))
        return points_tensor / norms

    def torch_voronoi_energy_ratio(points_tensor):
        """Compute the Voronoi energy ratio using PyTorch for gradient computation"""
        # Ensure points are on unit sphere
        points_normalized = torch_project_to_sphere(points_tensor)
        
        # Compute pairwise distances efficiently using PyTorch
        # Reshape for broadcasting: (n, 1, 3) - (1, n, 3) = (n, n, 3)
        diff = points_normalized.unsqueeze(1) - points_normalized.unsqueeze(0)
        distances_squared = torch.sum(diff**2, dim=2)
        
        # Avoid self-distances
        mask = ~torch.eye(points_tensor.shape[0], dtype=torch.bool, device=points_tensor.device)
        valid_distances = distances_squared[mask].view(points_tensor.shape[0], -1)
        
        # Get min and max distances
        d_min = torch.min(valid_distances)
        d_max = torch.max(valid_distances)
        
        # Return the ratio (sqrt for actual distance)
        if d_max.item() > 0:
            ratio = d_min / d_max
        else:
            ratio = torch.tensor(0.0, device=points_tensor.device)
            
        return ratio

    def torch_smooth_ratio(points_tensor):
        """Compute a smooth approximation of the ratio that's differentiable"""
        points_normalized = torch_project_to_sphere(points_tensor)
        
        # Compute pairwise distances
        diff = points_normalized.unsqueeze(1) - points_normalized.unsqueeze(0)
        distances_squared = torch.sum(diff**2, dim=2)
        
        # Avoid self-distances
        mask = ~torch.eye(points_tensor.shape[0], dtype=torch.bool, device=points_tensor.device)
        valid_distances = distances_squared[mask].view(points_tensor.shape[0], -1)
        
        # Smooth minimum and maximum using soft operations
        # Using log-sum-exp trick for smooth approximations
        eps = 1e-8
        # Soft minimum using log-sum-exp (approximate)
        soft_min = torch.logsumexp(-valid_distances / 0.1, dim=1) * 0.1  # Scale back
        soft_min = torch.min(soft_min)
        
        # Soft maximum using log-sum-exp
        soft_max = torch.logsumexp(valid_distances / 0.1, dim=1) * 0.1
        soft_max = torch.max(soft_max)
        
        # Return ratio
        if soft_max.item() > eps:
            ratio = soft_min / soft_max
        else:
            ratio = torch.tensor(0.0, device=points_tensor.device)
            
        return ratio

    def torch_voronoi_entropy_score(points_tensor):
        """Compute Voronoi entropy score using PyTorch"""
        try:
            # Convert to numpy for SphericalVoronoi
            points_np = points_tensor.detach().cpu().numpy()
            sv = SphericalVoronoi(points_np)
            areas = sv.calculate_areas()
            # Normalize areas
            areas = areas / np.sum(areas)
            # Entropy calculation
            entropy = -np.sum(areas * np.log(areas + 1e-10))
            return torch.tensor(entropy, device=points_tensor.device)
        except:
            return torch.tensor(0.0, device=points_tensor.device)

    def torch_comprehensive_fitness(points_tensor):
        """Combine ratio and entropy into a comprehensive fitness"""
        ratio = torch_smooth_ratio(points_tensor)
        entropy = torch_voronoi_entropy_score(points_tensor)
        # Combine with weighting
        fitness = ratio * (1.0 + 0.1 * entropy)
        return fitness

    def optimize_with_pytorch(initial_points, max_iter=5000, lr=0.01):
        """Optimize using PyTorch's automatic differentiation"""
        # Convert to PyTorch tensor with requires_grad=True
        points_tensor = torch.tensor(initial_points, dtype=torch.float32, requires_grad=True)
        
        # Optimizer
        optimizer = torch.optim.Adam([points_tensor], lr=lr)
        
        best_points = initial_points.copy()
        best_ratio = calculate_min_max_ratio(initial_points)
        best_fitness = torch_comprehensive_fitness(points_tensor).item()
        
        # Tracking for early stopping
        prev_fitness = float('-inf')
        patience_counter = 0
        patience_limit = 50
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # Forward pass
            fitness = torch_comprehensive_fitness(points_tensor)
            
            # Backward pass (negative because we want to maximize)
            (-fitness).backward()
            
            # Update
            optimizer.step()
            
            # Project back to sphere after update
            with torch.no_grad():
                points_tensor.data = torch_project_to_sphere(points_tensor)
            
            # Compute current ratio for checking
            current_points = points_tensor.detach().cpu().numpy()
            current_ratio = calculate_min_max_ratio(current_points)
            current_fitness = fitness.item()
            
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_ratio = current_ratio
                best_points = current_points.copy()
                
            # Early stopping logic
            if abs(current_fitness - prev_fitness) < 1e-8:
                patience_counter += 1
                if patience_counter > patience_limit:
                    break
            else:
                patience_counter = 0
            
            prev_fitness = current_fitness
            
        return best_points, best_ratio

    def torch_gradient_refinement(initial_points, max_iter=1000):
        """Refine solution using pure gradient descent on PyTorch tensors"""
        # Convert to PyTorch tensor
        points_tensor = torch.tensor(initial_points, dtype=torch.float32, requires_grad=True)
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam([points_tensor], lr=0.005)
        
        best_ratio = calculate_min_max_ratio(initial_points)
        best_points = initial_points.copy()
        
        # Gradient descent loop
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # Compute fitness - we want to maximize, so minimize negative
            fitness = torch_comprehensive_fitness(points_tensor)
            loss = -fitness
            
            # Backpropagate
            loss.backward()
            
            # Step
            optimizer.step()
            
            # Project points back to sphere
            with torch.no_grad():
                points_tensor.data = torch_project_to_sphere(points_tensor)
            
            # Check if this is better
            current_points = points_tensor.detach().cpu().numpy()
            current_ratio = calculate_min_max_ratio(current_points)
            
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points.copy()
        
        return best_points, best_ratio

    # Main execution flow
    np.random.seed(42)
    
    # Initialize with Fibonacci sphere (good starting point)
    initial_points = initialize_fibonacci_sphere(14)
    
    # First phase: PyTorch gradient optimization
    optimized_points, _ = optimize_with_pytorch(initial_points, max_iter=3000, lr=0.01)
    
    # Second phase: Refinement with more precise gradient approach
    refined_points, _ = torch_gradient_refinement(optimized_points, max_iter=2000)
    
    # Third phase: Final local improvement with numpy-based search
    def local_search_improvement(points):
        """Simple local search to fine-tune the result"""
        current_points = points.copy()
        current_ratio = calculate_min_max_ratio(current_points)
        
        # Try small perturbations to improve the solution
        for _ in range(5000):
            neighbor_points = current_points.copy()
            point_idx = np.random.randint(len(neighbor_points))
            
            # Small random perturbation
            perturbation = np.random.normal(0, 0.001, 3)
            neighbor_points[point_idx] += perturbation
            
            # Project back to sphere
            norm = np.linalg.norm(neighbor_points[point_idx])
            if norm > 0:
                neighbor_points[point_idx] = neighbor_points[point_idx] / norm
            
            new_ratio = calculate_min_max_ratio(neighbor_points)
            
            if new_ratio > current_ratio:
                current_ratio = new_ratio
                current_points = neighbor_points.copy()
        
        return current_points, current_ratio
    
    final_points, _ = local_search_improvement(refined_points)
    
    # Normalize to unit cube [0,1]^3
    points_in_cube = project_to_unit_cube(final_points)
    
    return points_in_cube

# EVOLVE-BLOCK-END