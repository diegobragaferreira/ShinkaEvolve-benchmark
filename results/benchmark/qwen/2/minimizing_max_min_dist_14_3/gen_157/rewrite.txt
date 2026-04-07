# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a Voronoi-based clustering optimization approach.
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

    def initialize_points_on_sphere(n):
        """Initialize points on a unit sphere using fibonacci spiral method."""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - 2 * (i / (n - 1)))
            phi = np.arctan2(np.sin(i * 2 * np.pi / golden_ratio), np.cos(i * 2 * np.pi / golden_ratio))
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)

    def voronoi_uniformity_score(points):
        """
        Calculate uniformity score based on Voronoi cell areas.
        """
        try:
            sv = SphericalVoronoi(points)
            areas = sv.calculate_areas()
            # Normalize areas
            areas = areas / np.sum(areas)
            # Variance of areas indicates uniformity (lower variance = more uniform)
            return 1.0 / (1.0 + np.var(areas) * 100.0)
        except:
            return 0.0

    def clustering_based_fitness(points):
        """Fitness function combining distance ratio and clustering uniformity."""
        ratio = calculate_min_max_ratio(points)
        uniformity = voronoi_uniformity_score(points)
        # Combine fitness: prioritize both uniformity and distance ratio
        return ratio * (1.0 + 0.3 * uniformity)

    def compute_voronoi_cell_stats(points):
        """Compute statistics about Voronoi cells for clustering analysis."""
        try:
            sv = SphericalVoronoi(points)
            areas = sv.calculate_areas()
            return {
                'mean_area': np.mean(areas),
                'std_area': np.std(areas),
                'min_area': np.min(areas),
                'max_area': np.max(areas)
            }
        except:
            return {'mean_area': 1.0, 'std_area': 0.0, 'min_area': 1.0, 'max_area': 1.0}

    def smart_neighbor_generation(current_points, generation=0, max_generations=2000):
        """Generate neighbor configuration with intelligent clustering-aware perturbations."""
        neighbor_points = current_points.copy()

        # Determine perturbation intensity based on optimization phase
        phase_factor = generation / max_generations
        if phase_factor < 0.3:
            # Exploration phase - larger perturbations
            base_magnitude = 0.1
        elif phase_factor < 0.7:
            # Exploitation phase - medium perturbations  
            base_magnitude = 0.05
        else:
            # Fine-tuning phase - small perturbations
            base_magnitude = 0.01
            
        # Analyze current structure for adaptive perturbation
        cell_stats = compute_voronoi_cell_stats(current_points)
        area_variance = cell_stats['std_area']
        
        # Adjust perturbation based on clustering quality
        if area_variance > 0.3:  # Highly uneven cells
            magnitude_factor = 2.0
        elif area_variance > 0.1:  # Moderately uneven
            magnitude_factor = 1.5
        else:  # Well-balanced cells
            magnitude_factor = 1.0
            
        perturbation_magnitude = base_magnitude * magnitude_factor
        
        # Select points to modify - focus on those in high-curvature regions
        # Compute pairwise distances to identify clustering patterns
        distances = pdist(current_points)
        if len(distances) > 0:
            # Find points that are unusually close or far
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Identify points that are part of tight clusters or isolated
            point_cluster_distances = []
            for i in range(len(current_points)):
                # Compute average distance to neighbors
                point_dists = []
                for j in range(len(current_points)):
                    if i != j:
                        point_dists.append(distance.euclidean(current_points[i], current_points[j]))
                avg_dist = np.mean(point_dists) if point_dists else mean_dist
                point_cluster_distances.append(avg_dist)
            
            # Select points to modify based on deviation from mean
            deviations = np.abs(np.array(point_cluster_distances) - mean_dist)
            threshold = mean_dist + 0.5 * std_dist if std_dist > 0 else mean_dist * 0.5
            candidates = [i for i, dev in enumerate(deviations) if dev > threshold]
            
            # If no strong candidates, just select randomly
            if len(candidates) == 0:
                candidates = list(range(len(current_points)))
                
            num_modify = min(max(2, len(candidates) // 4), 5)
            indices_to_modify = np.random.choice(candidates, num_modify, replace=False)
        else:
            # Default: modify random points
            num_modify = max(2, len(current_points) // 3)
            indices_to_modify = np.random.choice(len(current_points), num_modify, replace=False)

        for idx in indices_to_modify:
            # Generate perturbation that preserves spherical nature
            random_vec = np.random.randn(3)
            normal_vec = current_points[idx]
            tangent_vec = random_vec - np.dot(random_vec, normal_vec) * normal_vec
            tangent_norm = np.linalg.norm(tangent_vec)
            if tangent_norm > 1e-10:
                tangent_vec = tangent_vec / tangent_norm
            perturbation = tangent_vec * np.random.normal(0, perturbation_magnitude)
            neighbor_points[idx] += perturbation
            
            # Project back to sphere
            norm = np.linalg.norm(neighbor_points[idx])
            if norm > 0:
                neighbor_points[idx] = neighbor_points[idx] / norm

        return neighbor_points

    def hierarchical_clustering_optimize(initial_points, max_generations=2000):
        """
        Hierarchical clustering-based optimization approach.
        """
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_fitness = clustering_based_fitness(current_points)

        # Phase 1: Global structure optimization
        for generation in range(max_generations // 2):
            neighbor_points = smart_neighbor_generation(current_points, generation, max_generations)
            
            new_fitness = clustering_based_fitness(neighbor_points)
            
            # Accept or reject based on fitness
            if new_fitness > best_fitness:
                current_points = neighbor_points
                best_fitness = new_fitness
                best_points = neighbor_points.copy()
            elif np.random.rand() < 0.1:  # Allow some bad moves for escape
                current_points = neighbor_points

        # Phase 2: Local refinement with multi-level analysis
        for generation in range(max_generations // 2, max_generations):
            # Multi-scale clustering analysis
            # First, perform clustering to identify regions
            try:
                # Use hierarchical clustering to find natural groupings
                if len(current_points) > 3:
                    linkage_matrix = linkage(current_points, method='ward')
                    clusters = fcluster(linkage_matrix, t=0.3, criterion='distance')
                    
                    # Optimize each cluster separately to promote uniformity
                    unique_clusters = np.unique(clusters)
                    for cluster_id in unique_clusters:
                        cluster_indices = np.where(clusters == cluster_id)[0]
                        if len(cluster_indices) >= 2:
                            # Create subset for this cluster
                            cluster_points = current_points[cluster_indices]
                            
                            # Apply local optimization to cluster
                            cluster_center = np.mean(cluster_points, axis=0)
                            cluster_center = cluster_center / np.linalg.norm(cluster_center)  # Keep on sphere
                            
                            # Move cluster points closer to center
                            for j, idx in enumerate(cluster_indices):
                                if len(cluster_points) > 1:
                                    direction = cluster_center - current_points[idx]
                                    step_size = 0.02 * (1.0 - generation / max_generations)
                                    current_points[idx] += direction * step_size
                                    norm = np.linalg.norm(current_points[idx])
                                    if norm > 0:
                                        current_points[idx] = current_points[idx] / norm
            except:
                pass  # Skip clustering if it fails

            neighbor_points = smart_neighbor_generation(current_points, generation, max_generations)
            
            new_fitness = clustering_based_fitness(neighbor_points)
            
            if new_fitness > best_fitness:
                current_points = neighbor_points
                best_fitness = new_fitness
                best_points = neighbor_points.copy()
            elif np.random.rand() < 0.05:  # Even more rare acceptance
                current_points = neighbor_points

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

    def advanced_gradient_refinement(points):
        """Advanced gradient-based refinement with multi-step approach."""
        def objective(x_flat):
            points_local = x_flat.reshape(-1, 3)
            # Keep points on unit sphere constraint
            for i in range(len(points_local)):
                norm = np.linalg.norm(points_local[i])
                if norm > 1e-10:
                    points_local[i] = points_local[i] / norm
            return -calculate_min_max_ratio(points_local)

        def objective_with_gradients(x_flat):
            points_local = x_flat.reshape(-1, 3)
            # Keep points on unit sphere constraint
            for i in range(len(points_local)):
                norm = np.linalg.norm(points_local[i])
                if norm > 1e-10:
                    points_local[i] = points_local[i] / norm
            value = -calculate_min_max_ratio(points_local)
            # Simple finite difference gradients
            grad = np.zeros_like(x_flat)
            eps = 1e-6
            for i in range(len(x_flat)):
                x_plus = x_flat.copy()
                x_minus = x_flat.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            return value, grad

        refined_points = points.copy()

        # Stage 1: Coarse refinement
        try:
            from scipy.optimize import minimize
            result1 = minimize(objective, refined_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 150, 'ftol': 1e-5, 'gtol': 1e-5})
            refined_points = result1.x.reshape(-1, 3)
            
            # Project back to sphere
            for i in range(len(refined_points)):
                norm = np.linalg.norm(refined_points[i])
                if norm > 1e-10:
                    refined_points[i] = refined_points[i] / norm
                    
        except:
            pass

        # Stage 2: Fine refinement with stricter tolerances
        try:
            result2 = minimize(objective, refined_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            refined_points = result2.x.reshape(-1, 3)
            
            # Project back to sphere
            for i in range(len(refined_points)):
                norm = np.linalg.norm(refined_points[i])
                if norm > 1e-10:
                    refined_points[i] = refined_points[i] / norm
                    
        except:
            pass

        # Stage 3: Iterative local search for final improvements
        current_points = refined_points.copy()
        best_ratio = calculate_min_max_ratio(current_points)
        best_points = current_points.copy()

        for iteration in range(500):
            neighbor_points = current_points.copy()
            point_idx = np.random.randint(len(neighbor_points))
            
            # Adaptive perturbation based on current configuration
            distances = pdist(current_points)
            if len(distances) > 0:
                d_min = np.min(distances)
                # Larger perturbations when points are tightly clustered
                perturbation_magnitude = 0.002 * (1.0 - d_min/2.0) if d_min < 1.0 else 0.001
            else:
                perturbation_magnitude = 0.001
                
            perturbation = np.random.normal(0, perturbation_magnitude, 3)
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

    # Strategy 1: Multi-start with different initialization approaches
    initial_strategies = []
    
    # Multiple Fibonacci sphere initializations
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        init1 = initialize_points_on_sphere(14)
        initial_strategies.append(("fibonacci", init1))
    
    # Strategy 2: Random initialization on sphere
    np.random.seed(999)
    init2 = np.random.uniform(-1, 1, (14, 3))
    for i in range(len(init2)):
        norm = np.linalg.norm(init2[i])
        if norm > 1e-10:
            init2[i] = init2[i] / norm
    initial_strategies.append(("random_sphere", init2))
    
    # Strategy 3: Systematic initialization from icosahedral vertices
    try:
        # Create vertices of regular icosahedron and sample from them
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            (-1, 0, phi), (1, 0, phi), (-1, 0, -phi), (1, 0, -phi),
            (0, phi, 1), (0, phi, -1), (0, -phi, 1), (0, -phi, -1),
            (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0)
        ])
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        # Sample 14 points with good distribution
        sample_indices = np.random.choice(len(vertices), 14, replace=True)
        init3 = vertices[sample_indices]
        initial_strategies.append(("icosahedral", init3))
    except:
        pass

    # Run hierarchical clustering optimization on all strategies
    best_points = None
    best_fitness = -np.inf
    
    for i, (strategy_name, initial_points) in enumerate(initial_strategies):
        print(f"Starting optimization run {i+1}/{len(initial_strategies)}")
        optimized_points, fitness = hierarchical_clustering_optimize(initial_points, max_generations=1000)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_points = optimized_points.copy()

    # Final advanced refinement
    final_points, _ = advanced_gradient_refinement(best_points)

    # Normalize to unit cube [0,1]^3
    points_in_cube = project_to_unit_cube(final_points)

    return points_in_cube

# EVOLVE-BLOCK-END