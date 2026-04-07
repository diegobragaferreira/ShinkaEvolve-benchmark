# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
from scipy.linalg import eig
import time
import warnings
from sklearn.cluster import SpectralClustering

class SpectralPointEvolver:
    """Spectral clustering based point dispersion optimizer using geometric force fields."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        
    def calculate_ratio(self, points: np.ndarray) -> tuple[float, float, float]:
        """Calculate min/max distance ratio with proper error handling."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
        
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0, min_dist, max_dist
            
        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist
    
    def compute_spectral_clustering(self, points: np.ndarray, n_clusters: int = 4) -> np.ndarray:
        """Use spectral clustering to identify natural groupings in the point distribution."""
        # Convert to distance matrix
        distances = squareform(pdist(points))
        
        # Create similarity matrix (inverse of distances with small epsilon)
        eps = 1e-10
        similarity = 1.0 / (distances + eps)
        
        # Set diagonal to zero to remove self-connections
        np.fill_diagonal(similarity, 0)
        
        # Use spectral clustering to identify natural groupings
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            n_init=10
        )
        
        labels = clustering.fit_predict(similarity)
        
        # Create clustered points with slight perturbation around cluster centers
        clustered_points = points.copy()
        cluster_centers = []
        
        for i in range(n_clusters):
            cluster_mask = labels == i
            if np.sum(cluster_mask) > 0:
                center = np.mean(points[cluster_mask], axis=0)
                cluster_centers.append(center)
            else:
                cluster_centers.append(np.mean(points, axis=0))
        
        # Distribute points around cluster centers with radial perturbation
        for i, label in enumerate(labels):
            center = cluster_centers[label]
            # Radial displacement based on cluster size and point index
            cluster_size = np.sum(labels == label)
            if cluster_size > 1:
                # Add small random displacement from cluster center
                displacement = np.random.normal(0, 0.05 / np.sqrt(cluster_size), self.dimension)
                clustered_points[i] = center + displacement
            else:
                clustered_points[i] = center
        
        # Ensure bounds
        clustered_points = np.clip(clustered_points, 0.001, 0.999)
        return clustered_points
    
    def compute_geometric_force_field(self, points: np.ndarray, 
                                   k_repulsion: float = 1.0, 
                                   k_attraction: float = 0.1,
                                   k_cluster: float = 0.5) -> np.ndarray:
        """Compute forces using geometric and spectral insights."""
        n = points.shape[0]
        forces = np.zeros_like(points)
        
        # Compute distance matrix
        distances = squareform(pdist(points))
        
        # Repulsive forces (Coulomb-like)
        for i in range(n):
            for j in range(i+1, n):
                diff = points[i] - points[j]
                distance = np.linalg.norm(diff)
                
                if distance > 1e-8:  # Avoid division by zero
                    # Repulsive force (inverse square law, with distance threshold)
                    force_magnitude = k_repulsion / (distance * distance + 0.01)
                    force_vector = force_magnitude * diff / distance
                    forces[i] += force_vector
                    forces[j] -= force_vector
        
        # Attractive forces towards cluster centers (based on spectral clustering)
        # This helps maintain global structure while allowing local optimization
        try:
            clustered_points = self.compute_spectral_clustering(points, n_clusters=4)
            for i in range(n):
                # Attract to the centroid of the cluster that point belongs to
                # Using a simple heuristic for simplicity
                center = np.mean(points, axis=0)  # Simple overall center
                diff = center - points[i]
                force_magnitude = k_attraction * np.linalg.norm(diff)
                force_vector = force_magnitude * diff / (np.linalg.norm(diff) + 1e-8)
                forces[i] += force_vector
        except:
            # Fallback to simple center attraction
            center = np.mean(points, axis=0)
            for i in range(n):
                diff = center - points[i]
                force_magnitude = k_attraction * np.linalg.norm(diff)
                force_vector = force_magnitude * diff / (np.linalg.norm(diff) + 1e-8)
                forces[i] += force_vector
        
        # Cluster cohesion forces
        if n >= 4:
            # Create pseudo-cluster forces for better distribution
            for i in range(0, n, 4):  # Every 4th point
                if i + 3 < n:
                    cluster_center = np.mean(points[i:i+4], axis=0)
                    for j in range(i, i+4):
                        diff = cluster_center - points[j]
                        force_magnitude = k_cluster * np.linalg.norm(diff)
                        force_vector = force_magnitude * diff / (np.linalg.norm(diff) + 1e-8)
                        forces[j] += force_vector
                        
        return forces
    
    def force_relaxation(self, points: np.ndarray, iterations: int = 200, 
                        initial_step_size: float = 0.02) -> np.ndarray:
        """Perform force relaxation using computed geometric forces."""
        current_points = points.copy()
        step_size = initial_step_size
        
        # Progressive step size reduction
        for iteration in range(iterations):
            if iteration > 0 and iteration % 40 == 0:
                step_size *= 0.8
            
            # Compute forces
            forces = self.compute_geometric_force_field(current_points)
            
            # Apply forces with adaptive step size
            current_points += forces * step_size
            
            # Enforce bounds
            current_points = np.clip(current_points, 0.001, 0.999)
            
        return current_points
    
    def generate_spectral_initial(self) -> np.ndarray:
        """Generate initial configuration using spectral clustering insights."""
        # Start with hexagonal grid
        points = []
        rows = cols = 4
        
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                points.append([x, y])
        
        points = np.array(points[:self.num_points])
        
        # Apply spectral clustering to create better structured distribution
        points = self.compute_spectral_clustering(points, n_clusters=4)
        
        return points
    
    def generate_multi_scale_pattern(self) -> np.ndarray:
        """Generate multiple configurations using different geometric patterns."""
        configs = []
        
        # 1. Spectral initial configuration
        configs.append(self.generate_spectral_initial())
        
        # 2. Hexagonal pattern with better spacing
        points = []
        rows = cols = 4
        spacing_x = 0.9 / (cols - 1) if cols > 1 else 0.9
        spacing_y = 0.9 / (rows - 1) if rows > 1 else 0.9
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = 0.05 + (j * spacing_x) + x_offset
                y = 0.05 + i * spacing_y
                points.append([x, y])
        
        configs.append(np.array(points[:self.num_points]))
        
        # 3. Polar pattern with spectral insights
        points = []
        radii = [0.15, 0.3, 0.45, 0.6]
        angles_per_ring = [4, 6, 8, 10]
        
        # Center point
        points.append([0.5, 0.5])
        
        # Ring points
        for i, (radius, num_angles) in enumerate(zip(radii, angles_per_ring)):
            for j in range(num_angles):
                if len(points) >= self.num_points:
                    break
                angle = (j * 2 * np.pi) / num_angles
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
            if len(points) >= self.num_points:
                break
        
        # Fill remaining spots with random distribution
        remaining = self.num_points - len(points)
        for _ in range(remaining):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            points.append([x, y])
        
        configs.append(np.array(points[:self.num_points]))
        
        # Add perturbed versions
        np.random.seed(42)
        for config in configs:
            # Add multiple perturbations
            for perturbation_magnitude in [0.01, 0.015, 0.02]:
                perturbed = config + np.random.normal(0, perturbation_magnitude, config.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                configs.append(perturbed)
        
        return configs
    
    def spectral_optimize(self, configs: list[np.ndarray]) -> np.ndarray:
        """Main optimization using spectral clustering and force relaxation."""
        best_ratio = -np.inf
        best_points = None
        
        for i, config in enumerate(configs):
            # Apply force relaxation for global optimization
            relaxed_points = self.force_relaxation(config, iterations=150, initial_step_size=0.015)
            
            # Perform final optimization with gradient-based refinement
            try:
                def objective(x):
                    points_temp = x.reshape(-1, self.dimension)
                    ratio, _, _ = self.calculate_ratio(points_temp)
                    return -ratio
                
                result = minimize(
                    objective,
                    relaxed_points.flatten(),
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, self.dimension)
                    ratio, _, _ = self.calculate_ratio(optimized_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
            except Exception:
                # Fallback to relaxed points
                ratio, _, _ = self.calculate_ratio(relaxed_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = relaxed_points.copy()
        
        # If still no good solution, return the best from initial configurations
        if best_points is None:
            return configs[0] if configs else np.random.rand(self.num_points, self.dimension)
        
        # Additional refinement with spectral clustering guided optimization
        try:
            # Re-optimize using final configuration
            refined_points = self.force_relaxation(best_points, iterations=75, initial_step_size=0.005)
            
            # Final gradient-based optimization
            def objective(x):
                points_temp = x.reshape(-1, self.dimension)
                ratio, _, _ = self.calculate_ratio(points_temp)
                return -ratio
            
            result = minimize(
                objective,
                refined_points.flatten(),
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, self.dimension)
                ratio, _, _ = self.calculate_ratio(final_points)
                if ratio > best_ratio:
                    best_points = final_points.copy()
                    
        except Exception:
            pass
            
        return best_points if best_points is not None else configs[0]

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize evolver
    evolver = SpectralPointEvolver(16, 2)
    
    # Generate diverse initial configurations
    initial_configs = evolver.generate_multi_scale_pattern()
    
    # Optimize using spectral clustering approach
    try:
        best_points = evolver.spectral_optimize(initial_configs)
    except Exception as e:
        warnings.warn(f"Spectral optimization failed: {e}")
        # Fallback to first configuration
        best_points = initial_configs[0] if initial_configs else np.random.rand(16, 2)
    
    # Ensure bounds
    best_points = np.clip(best_points, 0.001, 0.999)
    
    return best_points

# EVOLVE-BLOCK-END