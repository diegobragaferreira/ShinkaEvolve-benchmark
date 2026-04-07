# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import time
from scipy.spatial.transform import Rotation as R

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return 0.0

        return d_min / d_max
    
    def spherical_voronoi_sampling(n_points):
        """Generate points using spherical Voronoi sampling for better uniformity"""
        # Generate random points on sphere
        points = np.random.randn(n_points, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1e-10)
        
        # Iteratively improve using spherical Voronoi
        for _ in range(10):
            sv = SphericalVoronoi(points, radius=1.0)
            # Move points to centroid of their Voronoi cells
            new_points = []
            for cell in sv.spherical_voronoi_cells:
                # Simple centroid calculation for Voronoi cell
                cell_points = cell.vertices
                if len(cell_points) > 0:
                    centroid = np.mean(cell_points, axis=0)
                    # Project back to sphere
                    norm = np.linalg.norm(centroid)
                    if norm > 1e-10:
                        centroid = centroid / norm
                else:
                    # If no vertices, keep original point
                    centroid = points[sv._vertex_to_index.get(id(cell.vertices[0]), 0)] if len(cell.vertices) > 0 else points[0]
                new_points.append(centroid)
            points = np.array(new_points)
        
        return points
    
    def quaternion_rotation_operator(points, mutation_strength=0.1):
        """Apply quaternion-based rotations to points while maintaining spherical constraint"""
        # Create a random quaternion
        quat = np.random.randn(4)
        quat = quat / np.linalg.norm(quat)
        
        # Convert to rotation matrix
        rot_mat = R.from_quat(quat).as_matrix()
        
        # Apply to all points
        rotated_points = points @ rot_mat.T
        
        # Add small random perturbations
        perturbations = np.random.normal(0, mutation_strength, points.shape)
        mutated_points = rotated_points + perturbations
        
        # Project back to sphere
        norms = np.linalg.norm(mutated_points, axis=1, keepdims=True)
        mutated_points = mutated_points / np.maximum(norms, 1e-10)
        
        return mutated_points
    
    def sphere_evolutionary_operator(parents, population_size=20, mutation_rate=0.1):
        """Custom evolutionary operator designed for sphere geometry"""
        offspring = []
        
        # Elitism - keep best parent
        parents_sorted = sorted(parents, key=compute_min_max_ratio, reverse=True)
        best_parent = parents_sorted[0]
        offspring.append(best_parent)
        
        # Generate offspring through combinations and mutations
        for _ in range(population_size - 1):
            # Select two parents randomly (tournament selection)
            parent1_idx = np.random.randint(0, len(parents))
            parent2_idx = np.random.randint(0, len(parents))
            
            parent1 = parents[parent1_idx]
            parent2 = parents[parent2_idx]
            
            # Blend the points (simple average)
            blend_factor = np.random.random()
            child = parent1 * blend_factor + parent2 * (1 - blend_factor)
            
            # Normalize to sphere
            norms = np.linalg.norm(child, axis=1, keepdims=True)
            child = child / np.maximum(norms, 1e-10)
            
            # Apply mutation
            if np.random.random() < mutation_rate:
                child = quaternion_rotation_operator(child, 0.1)
            
            offspring.append(child)
        
        return offspring
    
    def progressive_refinement_optimization(initial_points, max_evaluations=5000):
        """Progressive refinement that starts coarse and becomes precise"""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        # Phase 1: Coarse optimization (fast, broad search)
        for i in range(500):
            # Generate candidate through evolutionary operator
            candidates = sphere_evolutionary_operator([current_points], population_size=5)
            candidate = candidates[0]
            
            ratio = compute_min_max_ratio(candidate)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = candidate.copy()
            
            # Accept with some probability based on improvement
            if ratio > compute_min_max_ratio(current_points):
                current_points = candidate.copy()
        
        # Phase 2: Fine optimization (precise, local search)
        for i in range(1000):
            # Try small local perturbations around current best
            perturbed = current_points.copy()
            noise = np.random.normal(0, 0.01, perturbed.shape)
            perturbed = perturbed + noise
            
            # Project back to sphere
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            perturbed = perturbed / np.maximum(norms, 1e-10)
            
            ratio = compute_min_max_ratio(perturbed)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = perturbed.copy()
                current_points = perturbed.copy()
        
        # Phase 3: Final gradient-based refinement (L-BFGS-B)
        def objective(x_flat):
            points = x_flat.reshape(-1, 3)
            # Ensure points remain on sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / np.maximum(norms, 1e-10)
            ratio = compute_min_max_ratio(points)
            return -ratio  # Negative for minimization
        
        try:
            x0 = best_points.flatten()
            bounds = [(-1, 1)] * (14 * 3)
            
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
                refined_points = refined_points / np.maximum(norms, 1e-10)
                
                refined_ratio = compute_min_max_ratio(refined_points)
                if refined_ratio > best_ratio:
                    best_points = refined_points
                    best_ratio = refined_ratio
        except Exception:
            pass
        
        return best_points
    
    def generate_sphere_initializations():
        """Generate diverse initial point sets specifically designed for spherical optimization"""
        initial_sets = []
        
        # Strategy 1: Spherical Voronoi sampling (high uniformity)
        sv_points = spherical_voronoi_sampling(14)
        initial_sets.append(sv_points)
        
        # Strategy 2: Fibonacci sphere with systematic perturbation
        def fibonacci_sphere(n):
            points = []
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(n):
                y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y
                theta = golden_angle * i  # Golden angle increment
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                points.append([x, y, z])
            return np.array(points)
        
        fib_points = fibonacci_sphere(14)
        # Add structured perturbation
        perturbed = fib_points + np.random.normal(0, 0.01, fib_points.shape) * 0.5
        initial_sets.append(perturbed)
        
        # Strategy 3: Regular polyhedron-based distribution
        # Octahedron vertices
        octahedron_points = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ])
        
        # Add more points through subdivision or perturbation
        additional_points = np.random.randn(8, 3)
        # Normalize to sphere
        norms = np.linalg.norm(additional_points, axis=1, keepdims=True)
        additional_points = additional_points / np.maximum(norms, 1e-10)
        
        combined_points = np.vstack([octahedron_points, additional_points])
        # Ensure exactly 14 points
        if len(combined_points) > 14:
            combined_points = combined_points[:14]
        elif len(combined_points) < 14:
            # Fill with random points on sphere
            fill_points = np.random.randn(14 - len(combined_points), 3)
            norms = np.linalg.norm(fill_points, axis=1, keepdims=True)
            fill_points = fill_points / np.maximum(norms, 1e-10)
            combined_points = np.vstack([combined_points, fill_points])
        initial_sets.append(combined_points)
        
        # Strategy 4: Rotated Fibonacci with different angles
        rot_fib = fibonacci_sphere(14)
        # Rotate by fixed matrix
        rotation_matrix = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        rotated = rot_fib @ rotation_matrix.T
        initial_sets.append(rotated)
        
        # Strategy 5: Clustering-aware initialization
        # Start with a random uniform distribution and apply some structure
        rand_points = np.random.randn(14, 3)
        norms = np.linalg.norm(rand_points, axis=1, keepdims=True)
        rand_points = rand_points / np.maximum(norms, 1e-10)
        initial_sets.append(rand_points)
        
        return initial_sets
    
    # Main optimization loop
    best_solution = None
    best_ratio = 0.0
    
    # Generate diverse initial configurations
    initial_sets = generate_sphere_initializations()
    
    # Try multiple optimizations with different initializations
    for i, initial_points in enumerate(initial_sets):
        # Apply progressive refinement to each initial point set
        optimized_points = progressive_refinement_optimization(
            initial_points, 
            max_evaluations=1000
        )
        
        ratio = compute_min_max_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()
    
    # Additional meta-optimization
    if best_solution is not None:
        # Try one more round of progressive optimization from the best solution
        refined_solution = progressive_refinement_optimization(
            best_solution,
            max_evaluations=2000
        )
        
        refined_ratio = compute_min_max_ratio(refined_solution)
        if refined_ratio > best_ratio:
            best_ratio = refined_ratio
            best_solution = refined_solution
    
    # Fallback to spherical Voronoi if nothing worked
    if best_solution is None:
        best_solution = spherical_voronoi_sampling(14)
    
    return best_solution

# EVOLVE-BLOCK-END