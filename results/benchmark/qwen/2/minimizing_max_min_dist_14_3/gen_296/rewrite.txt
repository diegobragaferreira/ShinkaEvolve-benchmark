# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import math
import random
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist

    def compute_voronoi_uniformity(points):
        """Compute Voronoi cell area uniformity score."""
        try:
            sv = SphericalVoronoi(points)
            areas = sv.calculate_areas()
            # Return inverse of coefficient of variation (higher is better)
            if np.std(areas) == 0:
                return 1.0
            return 1.0 / (np.std(areas) / np.mean(areas) + 1e-10)
        except:
            return 0.0

    def fibonacci_sphere(n: int, seed: int = 42) -> np.ndarray:
        """Generate n points on a sphere using improved Fibonacci spiral method."""
        np.random.seed(seed)
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            theta = phi * i + np.random.normal(0, 0.1)  # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append([x, y, z])

        return np.array(points)

    def icosahedron_points(n: int, seed: int = 42) -> np.ndarray:
        """Generate points based on icosahedron vertices with perturbations."""
        np.random.seed(seed)
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        vertices = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        
        vertices = np.array(vertices)
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        if n <= 12:
            points = vertices[:n].copy()
        else:
            points = vertices.copy()
            for i in range(n - 12):
                idx = i % 12
                perturbation = np.random.normal(0, 0.1, 3)
                new_point = points[idx] + perturbation
                new_point = new_point / np.linalg.norm(new_point)
                points = np.vstack([points, new_point])
        
        return points[:n]

    def initialize_points(n: int, method: str = 'hybrid') -> np.ndarray:
        """Initialize points using hybrid approach."""
        if method == 'fibonacci':
            return fibonacci_sphere(n, 42)
        elif method == 'icosahedron':
            return icosahedron_points(n, 42)
        elif method == 'uniform':
            # Generate random uniform points
            points = np.random.uniform(-1, 1, (n, 3))
            for i in range(n):
                norm = np.linalg.norm(points[i])
                if norm > 0:
                    points[i] = points[i] / norm
            return points
        else:  # hybrid approach
            # Mix of fibonacci and icosahedron
            points1 = fibonacci_sphere(8, 42) 
            points2 = icosahedron_points(6, 42)
            return np.vstack([points1, points2])

    def project_to_sphere(points):
        """Project points to unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]

    def energy_based_perturb(points, index, temperature, current_ratio):
        """Perturb point using energy-based approach that considers both distance and Voronoi uniformity."""
        # Compute pairwise distances and Voronoi information for this point
        current_point = points[index]
        
        # Build influence map - how other points affect this one
        influence = np.zeros(3)
        total_influence = 0
        
        # Attract repulsive force from nearby points
        distances = pdist(points)
        if len(distances) > 0:
            dist_matrix = squareform(distances)
            # Get distances for this point
            point_dists = dist_matrix[index]
            # Consider nearby points (within some threshold)
            nearby_mask = point_dists < np.percentile(point_dists, 70)  # Consider top 30%
            nearby_indices = np.where(nearby_mask)[0]
            
            for i in nearby_indices:
                if i != index:
                    diff = points[i] - current_point
                    dist = np.linalg.norm(diff)
                    if dist > 1e-10:
                        # Repulsive force (inverse square law)
                        force_magnitude = 1.0 / (dist * dist + 1e-8)
                        force_direction = diff / dist
                        influence += force_magnitude * force_direction
                        total_influence += force_magnitude
        
        # Add some random component
        random_component = np.random.randn(3) * temperature * 0.05
        
        # Combined perturbation
        if total_influence > 0:
            # Weight by influence and add randomness
            perturbation = (influence / total_influence) * 0.02 + random_component
        else:
            perturbation = random_component
            
        # Ensure it's perpendicular to the point (tangent plane)
        tangent_component = perturbation - np.dot(perturbation, current_point) * current_point
        tangent_norm = np.linalg.norm(tangent_component)
        if tangent_norm > 1e-10:
            tangent_component = tangent_component / tangent_norm
            
        new_point = current_point + perturbation
        new_point = new_point / np.linalg.norm(new_point)
        return new_point

    def phase_optimization_phase1(points, max_iterations=50000):
        """Coarse exploration phase with large step sizes."""
        current_points = points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        best_uniformity = compute_voronoi_uniformity(current_points)
        
        # High temperature for exploration
        T = 1.0
        alpha = 0.999  # Cooling rate
        Tmin = 1e-6
        
        for iteration in range(max_iterations):
            T = max(Tmin, T * alpha)
            
            # Choose point to perturb
            point_idx = np.random.randint(0, 14)
            
            # Large perturbation in early stages
            step_size = T * 0.05
            new_point = energy_based_perturb(current_points, point_idx, T, best_ratio)
            current_points[point_idx] = new_point
            
            # Evaluate
            new_ratio = compute_min_max_ratio(current_points)
            new_uniformity = compute_voronoi_uniformity(current_points)
            
            # Accept or reject
            if new_ratio > best_ratio or np.random.random() < math.exp((new_ratio - best_ratio) / T):
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = current_points.copy()
                    best_uniformity = new_uniformity
                elif new_uniformity > best_uniformity:
                    # Even if ratio didn't improve, accept if uniformity did
                    best_uniformity = new_uniformity
                    best_points = current_points.copy()
                    best_ratio = new_ratio
            else:
                # Revert
                current_points[point_idx] = points[point_idx]
                
        return best_points, best_ratio

    def phase_optimization_phase2(points, max_iterations=30000):
        """Fine tuning phase with moderate step sizes."""
        current_points = points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        T = 0.1
        alpha = 0.9995
        Tmin = 1e-8
        
        for iteration in range(max_iterations):
            T = max(Tmin, T * alpha)
            
            # Choose point to perturb
            point_idx = np.random.randint(0, 14)
            
            # Moderate perturbation
            step_size = T * 0.02
            new_point = energy_based_perturb(current_points, point_idx, T, best_ratio)
            current_points[point_idx] = new_point
            
            # Evaluate
            new_ratio = compute_min_max_ratio(current_points)
            
            # Accept or reject
            if new_ratio > best_ratio or np.random.random() < math.exp((new_ratio - best_ratio) / T):
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = current_points.copy()
            else:
                # Revert
                current_points[point_idx] = points[point_idx]
                
        return best_points, best_ratio

    def phase_optimization_phase3(points, max_iterations=20000):
        """Local refinement phase with small step sizes."""
        current_points = points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        T = 0.01
        alpha = 0.9999
        Tmin = 1e-10
        
        for iteration in range(max_iterations):
            T = max(Tmin, T * alpha)
            
            # Choose point to perturb
            point_idx = np.random.randint(0, 14)
            
            # Small perturbation
            step_size = T * 0.01
            new_point = energy_based_perturb(current_points, point_idx, T, best_ratio)
            current_points[point_idx] = new_point
            
            # Evaluate
            new_ratio = compute_min_max_ratio(current_points)
            
            # Accept or reject
            if new_ratio > best_ratio or np.random.random() < math.exp((new_ratio - best_ratio) / T):
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = current_points.copy()
            else:
                # Revert
                current_points[point_idx] = points[point_idx]
                
        return best_points, best_ratio

    def multi_phase_optimization(starting_points):
        """Run multi-phase optimization with structured approach."""
        # Phase 1: Coarse exploration
        phase1_points, phase1_ratio = phase_optimization_phase1(starting_points, 50000)
        
        # Phase 2: Fine tuning
        phase2_points, phase2_ratio = phase_optimization_phase2(phase1_points, 30000)
        
        # Phase 3: Local refinement
        phase3_points, phase3_ratio = phase_optimization_phase3(phase2_points, 20000)
        
        # Return best result from all phases
        if phase3_ratio > phase2_ratio:
            return phase3_points, phase3_ratio
        elif phase2_ratio > phase1_ratio:
            return phase2_points, phase2_ratio
        else:
            return phase1_points, phase1_ratio

    def adaptive_multi_start_optimization():
        """Run optimization from multiple starting configurations."""
        best_points = None
        best_ratio = 0.0
        
        # Try different initialization methods
        methods = ['hybrid', 'fibonacci', 'icosahedron', 'uniform']
        seeds = [42, 123, 456, 789, 999, 111, 222, 333]
        
        start_time = time.time()
        for seed in seeds:
            if time.time() - start_time > 350:
                break
                
            for method in methods:
                try:
                    np.random.seed(seed)
                    initial_points = initialize_points(14, method)
                    optimized_points, ratio = multi_phase_optimization(initial_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                except Exception as e:
                    continue
                    
        # Final fallback
        if best_points is None:
            initial_points = initialize_points(14, 'hybrid')
            best_points, best_ratio = multi_phase_optimization(initial_points)
            
        return best_points

    # Run the optimization
    return adaptive_multi_start_optimization()

# EVOLVE-BLOCK-END