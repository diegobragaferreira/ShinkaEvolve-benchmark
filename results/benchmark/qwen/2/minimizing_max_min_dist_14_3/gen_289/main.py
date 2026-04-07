# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Uses a novel spherical tiling evolution approach with hierarchical optimization.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
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

    def project_to_sphere(points):
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]

    def fibonacci_sphere(n):
        """Generate n points distributed approximately uniformly on a sphere."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)
            theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def create_hexagonal_tiling_initial(n):
        """Create initial configuration using hexagonal-like tiling on sphere."""
        # Start with a base configuration
        points = fibonacci_sphere(n)
        
        # Apply a transformation to promote better spacing
        # Use a variant of the icosahedral construction for better uniformity
        try:
            # Create tetrahedral-like structure and perturb
            base_points = np.array([
                [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
            ])
            base_points = base_points / np.linalg.norm(base_points, axis=1, keepdims=True)
            
            # For 14 points, we'll use a hybrid approach
            if n == 14:
                # Start with icosahedron vertices
                phi = (1 + np.sqrt(5)) / 2
                vertices = [
                    (-1, 0, phi), (1, 0, phi), (-1, 0, -phi), (1, 0, -phi),
                    (0, phi, 1), (0, phi, -1), (0, -phi, 1), (0, -phi, -1),
                    (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0)
                ]
                vertices = np.array(vertices)
                # Normalize to unit sphere
                vertices = vertices / np.linalg.norm(vertices[0])
                
                # Add additional points to reach 14
                # Take midpoints of some edges and add to vertices
                edges = []
                for i in range(len(vertices)):
                    for j in range(i+1, len(vertices)):
                        dist = np.linalg.norm(vertices[i] - vertices[j])
                        if abs(dist - 2) < 0.2:
                            edges.append((i, j))
                
                # Add midpoints of first few edges
                additional_points = []
                for i, j in edges[:2]:
                    midpoint = (vertices[i] + vertices[j]) / 2
                    midpoint = midpoint / np.linalg.norm(midpoint)
                    additional_points.append(midpoint)
                
                all_points = np.vstack([vertices, additional_points])
                points = all_points[:14]
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
        except:
            # Fallback to fibonacci if the above fails
            pass
            
        # Add small noise to break symmetries
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        points = project_to_sphere(points)
        
        return points

    def adaptive_tangent_perturbation(points, point_idx, current_ratio, iteration=0):
        """Apply perturbation in tangent plane direction based on local geometry."""
        # Compute distances to neighbors
        distances = pdist(points)
        dist_matrix = squareform(distances)
        
        # Get distances from selected point to others
        dist_from_selected = dist_matrix[point_idx]
        
        # Identify closest and furthest neighbors
        valid_dists = np.copy(dist_from_selected)
        valid_dists[point_idx] = np.inf  # exclude self
        nearest_idx = np.argmin(valid_dists)
        furthest_idx = np.argmax(valid_dists)
        
        # Calculate adaptive perturbation magnitude
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        avg_dist = np.mean(distances)
        
        # Base perturbation based on current configuration state
        base_perturbation = 0.003 * (1.0 + 0.2 * min_dist)
        
        # Adjust based on whether point is clustered or scattered
        if min_dist < 0.2:
            # Very tight - large perturbations
            perturbation_magnitude = base_perturbation * 4.0
        elif min_dist < 0.4:
            # Moderately tight - medium perturbations  
            perturbation_magnitude = base_perturbation * 2.0
        else:
            # Well-separated - small perturbations
            perturbation_magnitude = base_perturbation * 0.5
            
        # Further adjust based on local density
        # If point is surrounded by close neighbors, push away
        close_neighbors = np.where(valid_dists < avg_dist * 0.8)[0]
        if len(close_neighbors) > 3:
            # Point is in a cluster, push away from cluster center
            cluster_center = np.mean(points[close_neighbors], axis=0)
            direction = points[point_idx] - cluster_center
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            perturbation = direction * perturbation_magnitude * 0.8
        else:
            # Point has relatively sparse neighbors, explore in tangent plane  
            perturbation = np.random.normal(0, perturbation_magnitude, 3)
        
        # Project perturbation into tangent plane of sphere
        normal_vec = points[point_idx]
        tangent_pert = perturbation - np.dot(perturbation, normal_vec) * normal_vec
        tangent_norm = np.linalg.norm(tangent_pert)
        if tangent_norm > 0:
            tangent_pert = tangent_pert / tangent_norm * perturbation_magnitude
            
        return tangent_pert

    def hierarchical_optimization(initial_points, max_iterations=5000):
        """Optimize using a hierarchical approach with progressive refinement."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        # Stage 1: Coarse optimization (larger steps)
        for iteration in range(1000):
            test_points = current_points.copy()
            point_idx = np.random.randint(len(test_points))
            
            # Apply adaptive perturbation
            perturbation = adaptive_tangent_perturbation(test_points, point_idx, best_ratio, iteration)
            test_points[point_idx] += perturbation
            
            # Project back to sphere
            test_points = project_to_sphere(test_points)
            
            # Check improvement
            test_ratio = compute_min_max_ratio(test_points)
            if test_ratio > best_ratio:
                current_points = test_points.copy()
                best_ratio = test_ratio
                best_points = current_points.copy()
            elif np.random.rand() < 0.1:  # Occasionally accept poor moves for escape
                current_points = test_points.copy()
        
        # Stage 2: Medium optimization (medium steps)  
        for iteration in range(1500):
            test_points = current_points.copy()
            point_idx = np.random.randint(len(test_points))
            
            # Smaller perturbation
            perturbation = adaptive_tangent_perturbation(test_points, point_idx, best_ratio, iteration)
            perturbation *= 0.3
            test_points[point_idx] += perturbation
            
            # Project back to sphere
            test_points = project_to_sphere(test_points)
            
            # Check improvement
            test_ratio = compute_min_max_ratio(test_points)
            if test_ratio > best_ratio:
                current_points = test_points.copy()
                best_ratio = test_ratio
                best_points = current_points.copy()
            elif np.random.rand() < 0.05:  # Occasionally accept poor moves for escape
                current_points = test_points.copy()
        
        # Stage 3: Fine optimization (small steps)
        for iteration in range(2500):
            test_points = current_points.copy()
            point_idx = np.random.randint(len(test_points))
            
            # Even smaller perturbation
            perturbation = adaptive_tangent_perturbation(test_points, point_idx, best_ratio, iteration)
            perturbation *= 0.1
            test_points[point_idx] += perturbation
            
            # Project back to sphere
            test_points = project_to_sphere(test_points)
            
            # Check improvement  
            test_ratio = compute_min_max_ratio(test_points)
            if test_ratio > best_ratio:
                current_points = test_points.copy()
                best_ratio = test_ratio
                best_points = current_points.copy()
        
        return best_points, best_ratio

    def energy_based_local_refinement(points, iterations=500):
        """Refine using an energy-based approach that minimizes local distortion."""
        current_points = points.copy()
        
        # Create a simple energy model that penalizes small distances
        def energy_function(current_points):
            distances = pdist(current_points)
            if len(distances) == 0:
                return 0
                
            # Penalize small distances and reward large ones  
            # Using inverse square to emphasize differences
            energy = 0
            for dist in distances:
                if dist > 0:
                    energy += 1.0 / (dist * dist)
            return energy
        
        # Simple gradient descent approach with sphere constraint
        for _ in range(iterations):
            # Compute approximate gradient via finite differences
            energies = []
            for i in range(len(current_points)):
                old_point = current_points[i].copy()
                h = 1e-6
                
                # Test perturbations in three coordinate directions
                grad = np.zeros(3)
                for dim in range(3):
                    # Forward difference
                    test_points_plus = current_points.copy()
                    test_points_minus = current_points.copy()
                    
                    test_points_plus[i, dim] += h
                    test_points_minus[i, dim] -= h
                    
                    # Project back to sphere
                    test_points_plus = project_to_sphere(test_points_plus)
                    test_points_minus = project_to_sphere(test_points_minus)
                    
                    # Compute gradient approximation
                    energy_plus = energy_function(test_points_plus)
                    energy_minus = energy_function(test_points_minus) 
                    grad[dim] = (energy_plus - energy_minus) / (2 * h)
                
                # Project gradient onto tangent plane
                normal = current_points[i]
                tangent_grad = grad - np.dot(grad, normal) * normal
                
                # Apply update (in direction of steepest descent to minimize energy)
                update = -0.001 * tangent_grad
                current_points[i] += update
                
            # Project back to sphere after all updates
            current_points = project_to_sphere(current_points)
            
        return current_points, compute_min_max_ratio(current_points)

    def project_to_unit_cube(points):
        """Project points to unit cube [0,1]^3"""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        ranges = max_coords - min_coords
        if np.any(ranges == 0):
            return np.full_like(points, 0.5)
        normalized = (points - min_coords) / ranges
        return np.clip(normalized, 0, 1)

    # Main optimization process
    np.random.seed(42)
    
    # Strategy 1: Hexagonal tiling initialization
    initial_points = create_hexagonal_tiling_initial(14)
    
    # Strategy 2: Hierarchical optimization
    optimized_points, _ = hierarchical_optimization(initial_points, max_iterations=5000)
    
    # Strategy 3: Energy-based local refinement  
    final_points, final_ratio = energy_based_local_refinement(optimized_points, iterations=300)
    
    # Ensure we have the best result
    if final_ratio > compute_min_max_ratio(initial_points):
        points = final_points
    else:
        points = initial_points
    
    # Final projection to unit cube
    points_in_cube = project_to_unit_cube(points)
    
    return points_in_cube

# EVOLVE-BLOCK-END