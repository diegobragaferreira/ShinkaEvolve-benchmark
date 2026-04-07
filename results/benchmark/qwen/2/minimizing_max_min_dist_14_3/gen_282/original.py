# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a Voronoi lattice optimization approach with multi-scale refinement.
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
            return 0.0

    def lattice_point_generator(n_points, grid_size=4):
        """
        Generate points on a 3D lattice and project them onto the unit sphere.
        This provides a better initial distribution than random points.
        """
        # Create a cubic lattice with specified grid size
        grid_points = []
        spacing = 2.0 / grid_size
        
        # Generate points in a cube, then project to sphere
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    x = i * spacing - 1.0
                    y = j * spacing - 1.0
                    z = k * spacing - 1.0
                    # Only include points that aren't too close to center
                    if np.linalg.norm([x, y, z]) > 0.1:
                        grid_points.append([x, y, z])
        
        # Convert to numpy array
        points = np.array(grid_points)
        
        # Normalize to unit sphere
        for i in range(len(points)):
            norm = np.linalg.norm(points[i])
            if norm > 0:
                points[i] = points[i] / norm
                
        # If we have more points than needed, select randomly
        if len(points) >= n_points:
            selected_indices = np.random.choice(len(points), n_points, replace=False)
            return points[selected_indices]
        else:
            # If we don't have enough points, fill with random points on sphere
            extra_points = n_points - len(points)
            random_points = np.random.uniform(-1, 1, (extra_points, 3))
            for i in range(extra_points):
                norm = np.linalg.norm(random_points[i])
                if norm > 0:
                    random_points[i] = random_points[i] / norm
            return np.vstack([points, random_points])

    def adaptive_fitness_function(points):
        """Enhanced fitness function that balances distance ratio and Voronoi uniformity."""
        ratio = calculate_min_max_ratio(points)
        entropy = voronoi_entropy_score(points)
        # Enhanced fitness combining both metrics with dynamic weights
        # Weight entropy more heavily when ratio is high to promote uniformity
        weight = 1.0 + 0.2 * (ratio ** 2)  # Higher ratio = more emphasis on uniformity
        return ratio * (1.0 + weight * entropy)

    def tangent_space_perturbation(current_points, perturbation_magnitude=0.05):
        """Apply perturbations that respect the spherical constraint using tangent space."""
        neighbor_points = current_points.copy()
        
        # Select multiple points to perturb for better exploration
        num_modify = max(2, min(len(current_points) // 3, 5))
        indices_to_modify = np.random.choice(len(current_points), num_modify, replace=False)
        
        for idx in indices_to_modify:
            # Generate perturbation in tangent space
            random_vec = np.random.randn(3)
            normal_vec = current_points[idx]
            # Tangent vector (orthogonal to normal)
            tangent_vec = random_vec - np.dot(random_vec, normal_vec) * normal_vec
            # Normalize tangent vector
            tangent_norm = np.linalg.norm(tangent_vec)
            if tangent_norm > 1e-10:
                tangent_vec = tangent_vec / tangent_norm
            # Apply perturbation scaled by magnitude
            perturbation = tangent_vec * np.random.normal(0, perturbation_magnitude)
            neighbor_points[idx] += perturbation
            # Project back to sphere ensuring numerical stability
            norm = np.linalg.norm(neighbor_points[idx])
            if norm > 1e-10:
                neighbor_points[idx] = neighbor_points[idx] / norm
                
        return neighbor_points

    def multi_scale_optimizer(initial_points, max_iterations=3000):
        """
        Multi-scale optimization that applies different strategies at different scales.
        """
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_fitness = adaptive_fitness_function(current_points)
        
        # Phase 1: Global exploration (large steps)
        exploration_phase_iterations = max_iterations // 3
        step_size = 0.1
        
        for iter_num in range(exploration_phase_iterations):
            neighbor_points = tangent_space_perturbation(current_points, step_size)
            
            # Calculate fitness
            current_fitness = adaptive_fitness_function(current_points)
            neighbor_fitness = adaptive_fitness_function(neighbor_points)
            
            # Accept or reject
            if neighbor_fitness > current_fitness or np.random.rand() < 0.1:
                current_points = neighbor_points
                if neighbor_fitness > best_fitness:
                    best_fitness = neighbor_fitness
                    best_points = neighbor_points.copy()
                    
            # Gradually decrease step size
            step_size = 0.1 * (1.0 - iter_num / exploration_phase_iterations)
        
        # Phase 2: Medium-scale refinement (medium steps)
        refinement_phase_iterations = max_iterations // 3
        step_size = 0.03
        
        for iter_num in range(refinement_phase_iterations):
            neighbor_points = tangent_space_perturbation(current_points, step_size)
            
            # Calculate fitness
            current_fitness = adaptive_fitness_function(current_points)
            neighbor_fitness = adaptive_fitness_function(neighbor_points)
            
            # Accept or reject
            if neighbor_fitness > current_fitness or np.random.rand() < 0.3:
                current_points = neighbor_points
                if neighbor_fitness > best_fitness:
                    best_fitness = neighbor_fitness
                    best_points = neighbor_points.copy()
                    
            # Gradually decrease step size
            step_size = 0.03 * (1.0 - iter_num / refinement_phase_iterations)
        
        # Phase 3: Local fine-tuning (small steps)
        fine_tuning_iterations = max_iterations - exploration_phase_iterations - refinement_phase_iterations
        step_size = 0.005
        
        for iter_num in range(fine_tuning_iterations):
            neighbor_points = tangent_space_perturbation(current_points, step_size)
            
            # Calculate fitness
            current_fitness = adaptive_fitness_function(current_points)
            neighbor_fitness = adaptive_fitness_function(neighbor_points)
            
            # Accept or reject
            if neighbor_fitness > current_fitness or np.random.rand() < 0.7:
                current_points = neighbor_points
                if neighbor_fitness > best_fitness:
                    best_fitness = neighbor_fitness
                    best_points = neighbor_points.copy()
                    
            # Gradually decrease step size
            step_size = 0.005 * (1.0 - iter_num / fine_tuning_iterations)
        
        return best_points, best_fitness

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

    def enhanced_gradient_refinement(points):
        """Apply gradient-based refinement to fine-tune the solution."""
        def objective(x_flat):
            points_local = x_flat.reshape(-1, 3)
            # Keep points on unit sphere constraint
            for i in range(len(points_local)):
                norm = np.linalg.norm(points_local[i])
                if norm > 1e-10:
                    points_local[i] = points_local[i] / norm
            return -calculate_min_max_ratio(points_local)

        try:
            # Use L-BFGS-B for fine tuning with strict tolerances
            result = minimize(objective, points.flatten(), method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}, 
                            tol=1e-8)
            refined_points = result.x.reshape(-1, 3)
            # Project back to sphere
            for i in range(len(refined_points)):
                norm = np.linalg.norm(refined_points[i])
                if norm > 1e-10:
                    refined_points[i] = refined_points[i] / norm
            return refined_points, -result.fun
        except:
            # Fallback to iterative refinement
            current_points = points.copy()
            best_ratio = calculate_min_max_ratio(current_points)
            best_points = current_points.copy()

            for iteration in range(1000):
                neighbor_points = current_points.copy()
                point_idx = np.random.randint(len(neighbor_points))
                
                # Small, controlled perturbation
                perturbation = np.random.normal(0, 0.0005, 3)
                neighbor_points[point_idx] += perturbation
                
                # Project back to sphere
                norm = np.linalg.norm(neighbor_points[point_idx])
                if norm > 1e-10:
                    neighbor_points[point_idx] = neighbor_points[point_idx] / norm
                
                new_ratio = calculate_min_max_ratio(neighbor_points)
                
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = neighbor_points.copy()
                    current_points = neighbor_points.copy()

            return best_points, best_ratio

    # Main execution flow
    np.random.seed(42)

    # Strategy 1: Lattice-based initialization
    initial_points = lattice_point_generator(14, grid_size=4)
    
    # Strategy 2: Multiple lattice initializations with different seeds
    best_points = None
    best_fitness = -np.inf
    
    # Try multiple lattice configurations with different random elements
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        # Generate a new lattice-based configuration
        lattice_points = lattice_point_generator(14, grid_size=4)
        # Optimize this configuration
        optimized_points, fitness = multi_scale_optimizer(lattice_points, max_iterations=2000)
        if fitness > best_fitness:
            best_fitness = fitness
            best_points = optimized_points.copy()
    
    # Final refinement with gradient-based approach
    final_points, _ = enhanced_gradient_refinement(best_points)

    # Normalize to unit cube [0,1]^3
    points_in_cube = project_to_unit_cube(final_points)

    return points_in_cube

# EVOLVE-BLOCK-END