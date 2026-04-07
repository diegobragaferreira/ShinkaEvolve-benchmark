# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a spherical Voronoi-based evolutionary approach for optimal point distribution.
    """
    
    def compute_voronoi_uniformity(points):
        """Compute coefficient of variation of Voronoi cell areas"""
        try:
            if len(points) < 4:
                return 1.0
                
            sv = SphericalVoronoi(points)
            areas = sv.voronoi_cell_areas()
            
            if len(areas) == 0:
                return 1.0
                
            mean_area = np.mean(areas)
            if mean_area <= 0:
                return 1.0
                
            cv = np.std(areas) / mean_area
            return cv
        except:
            return 1.0
    
    def compute_min_max_ratio(points):
        """Compute the actual min/max distance ratio"""
        if len(points) < 2:
            return 0.0
            
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return 0.0
            
        return min_dist / max_dist
    
    def project_to_sphere(points):
        """Project points to unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def create_spherical_grid(n_points):
        """Create initial point configuration using refined spherical grid"""
        # Generate points using a modified spherical grid approach
        points = []
        
        # Use a combination of Fibonacci-like distribution with refinement
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Generate points with careful spacing
        for i in range(n_points):
            # Modified Fibonacci spiral with better distribution
            if i == 0:
                theta = 0
                phi = 0
            elif i == n_points - 1:
                theta = np.pi
                phi = 0
            else:
                theta = np.arccos(1 - 2 * (i / (n_points - 1)))
                phi = i * 2 * np.pi / golden_ratio
                
            # Add slight perturbation to avoid symmetries
            perturbation = 0.02 * np.sin(i * 0.7) * np.cos(i * 0.3)
            theta += perturbation * 0.1
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
            
        return np.array(points)
    
    def spherical_coordinate_gradient(points, idx, step_size=1e-3):
        """Compute gradient approximation in spherical coordinates"""
        # Simple finite difference approach
        grad = np.zeros(3)
        base_point = points[idx].copy()
        
        # Convert cartesian to spherical coordinates
        r = np.linalg.norm(base_point)
        if r > 1e-10:
            theta = np.arccos(base_point[2] / r)
            phi = np.arctan2(base_point[1], base_point[0])
        else:
            theta, phi = 0, 0
            
        # Estimate gradient by perturbing each coordinate
        for dim in range(3):
            delta = np.zeros(3)
            delta[dim] = step_size
            
            # Forward difference
            p_plus = points[idx] + delta
            p_plus = project_to_sphere(p_plus.reshape(1, 3))[0]
            
            # Backward difference  
            p_minus = points[idx] - delta
            p_minus = project_to_sphere(p_minus.reshape(1, 3))[0]
            
            # Approximate gradient component
            grad[dim] = (compute_voronoi_uniformity(np.vstack([points[:idx], p_plus, points[idx+1:]])) - 
                        compute_voronoi_uniformity(np.vstack([points[:idx], p_minus, points[idx+1:]]))) / (2 * step_size)
            
        return grad
    
    def evolve_points_spherical_voronoi(points, iterations=500):
        """Evolve points using spherical Voronoi uniformity maximization"""
        best_points = points.copy()
        best_cv = compute_voronoi_uniformity(best_points)
        best_ratio = compute_min_max_ratio(best_points)
        
        # Evolution parameters
        learning_rate = 0.01
        decay_factor = 0.995
        min_lr = 1e-6
        
        for iteration in range(iterations):
            # Apply small random perturbations to explore neighborhood
            perturbation_magnitude = max(learning_rate * (decay_factor ** iteration), min_lr)
            
            # Create candidate points with perturbations
            candidate_points = best_points.copy()
            
            # Perturb each point individually (not too aggressively)
            for i in range(len(candidate_points)):
                # Random direction in tangent plane
                tangent_dir = np.random.randn(3)
                # Project to tangent plane (orthogonal to normal vector)
                normal_vec = candidate_points[i]
                tangent_dir = tangent_dir - np.dot(tangent_dir, normal_vec) * normal_vec
                tangent_dir = tangent_dir / (np.linalg.norm(tangent_dir) + 1e-10)
                
                # Apply perturbation along tangent
                candidate_points[i] += perturbation_magnitude * tangent_dir
                
                # Project back to sphere
                candidate_points[i] = project_to_sphere(candidate_points[i].reshape(1, 3))[0]
            
            # Evaluate candidate
            candidate_cv = compute_voronoi_uniformity(candidate_points)
            candidate_ratio = compute_min_max_ratio(candidate_points)
            
            # Accept if improvement or with some probability
            if candidate_cv < best_cv or (np.random.random() < 0.01 and candidate_cv < best_cv * 1.01):
                best_points = candidate_points.copy()
                best_cv = candidate_cv
                best_ratio = candidate_ratio
                
                # Early stopping when CV gets very low
                if best_cv < 0.05:
                    break
                    
            # Additional local refinement step
            if iteration % 50 == 0:
                # Try local search around current best
                local_candidates = []
                for _ in range(10):
                    test_points = best_points.copy()
                    idx = np.random.randint(len(test_points))
                    # Small random movement
                    move = np.random.normal(0, perturbation_magnitude * 0.1, 3)
                    test_points[idx] += move
                    test_points[idx] = project_to_sphere(test_points[idx].reshape(1, 3))[0]
                    
                    local_candidates.append(test_points)
                
                # Select best local candidate
                for candidate in local_candidates:
                    candidate_cv = compute_voronoi_uniformity(candidate)
                    if candidate_cv < best_cv:
                        best_points = candidate.copy()
                        best_cv = candidate_cv
                        best_ratio = compute_min_max_ratio(best_points)
        
        return best_points
    
    # Create initial configuration
    np.random.seed(42)
    initial_points = create_spherical_grid(14)
    
    # Further refine using Voronoi uniformity approach
    optimized_points = evolve_points_spherical_voronoi(initial_points, iterations=300)
    
    # Final polishing with local refinement
    final_points = optimized_points.copy()
    
    # Try several local optimizations
    for _ in range(10):
        # Random perturbations with small magnitude
        local_points = final_points.copy()
        for i in range(len(local_points)):
            if np.random.random() < 0.3:  # Only modify some points
                # Perturb in tangent direction
                tangent_dir = np.random.randn(3)
                normal_vec = local_points[i]
                tangent_dir = tangent_dir - np.dot(tangent_dir, normal_vec) * normal_vec
                tangent_dir = tangent_dir / (np.linalg.norm(tangent_dir) + 1e-10)
                local_points[i] += 0.001 * tangent_dir
                local_points[i] = project_to_sphere(local_points[i].reshape(1, 3))[0]
        
        # Check if this improved the ratio
        current_ratio = compute_min_max_ratio(final_points)
        candidate_ratio = compute_min_max_ratio(local_points)
        
        if candidate_ratio > current_ratio:
            final_points = local_points.copy()
    
    # Final validation and normalization
    final_points = project_to_sphere(final_points)
    
    return final_points

# EVOLVE-BLOCK-END