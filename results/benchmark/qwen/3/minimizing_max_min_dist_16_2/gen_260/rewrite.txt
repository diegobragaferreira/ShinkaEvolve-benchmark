# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import time
from scipy.optimize import minimize
import warnings

class VoronoiBasedOptimizer:
    """Voronoi-based optimization approach for point dispersion problems."""
    
    def __init__(self, num_points=16, seed=42):
        self.num_points = num_points
        self.seed = seed
        np.random.seed(seed)
        
    def _create_initial_voronoi_layout(self):
        """Create initial point layout based on Voronoi cell uniformity principles."""
        # Start with a hexagonal grid and adjust for better Voronoi uniformity
        points = []
        
        # Create hexagonal pattern with optimized spacing
        rows, cols = 4, 4
        hex_spacing = 1.0
        vert_spacing = np.sqrt(3) / 2 * hex_spacing
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                # Hexagonal offset pattern
                x = j * hex_spacing + (i % 2) * hex_spacing * 0.5
                y = i * vert_spacing
                
                # Add systematic perturbations
                perturbation = 0.02 * np.sin(i * 0.7) * np.cos(j * 0.5)
                x += perturbation * 0.3
                y += perturbation * 0.3
                
                points.append([x, y])
        
        points = np.array(points[:self.num_points])
        
        # Normalize to [0,1] square while maintaining proportions
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        if x_max > x_min and y_max > y_min:
            scale = min(1.0/(x_max - x_min), 1.0/(y_max - y_min)) * 0.9
            points[:, 0] = (points[:, 0] - x_min) * scale + 0.05
            points[:, 1] = (points[:, 1] - y_min) * scale + 0.05
        elif x_max > x_min:
            scale = 1.0 / (x_max - x_min) * 0.9
            points[:, 0] = (points[:, 0] - x_min) * scale + 0.05
        elif y_max > y_min:
            scale = 1.0 / (y_max - y_min) * 0.9
            points[:, 1] = (points[:, 1] - y_min) * scale + 0.05
            
        return points
        
    def _compute_voronoi_uniformity_metric(self, points):
        """Compute a metric based on Voronoi cell uniformity."""
        try:
            # Compute Voronoi diagram
            vor = Voronoi(points)
            
            # Calculate areas of finite Voronoi cells
            areas = []
            for i, region in enumerate(vor.regions):
                if not any(v == -1 for v in region) and len(region) >= 3:
                    # Calculate area using shoelace formula
                    polygon = [vor.vertices[j] for j in region]
                    if len(polygon) >= 3:
                        area = 0.5 * abs(sum(polygon[j][0] * polygon[(j+1)%len(polygon)][1] - 
                                           polygon[(j+1)%len(polygon)][0] * polygon[j][1] 
                                           for j in range(len(polygon))))
                        areas.append(area)
            
            if len(areas) < 2:
                return 0.0
                
            # Return coefficient of variation of areas (lower is better)
            mean_area = np.mean(areas)
            if mean_area == 0:
                return 0.0
            std_area = np.std(areas)
            return std_area / mean_area if mean_area > 0 else 0.0
            
        except Exception:
            return 1.0  # Penalize bad Voronoi computations
            
    def _compute_distance_ratio(self, points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0

        # Use cdist for efficient pairwise distance computation
        distances = cdist(points, points)
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        d_min = np.min(distances[np.isfinite(distances)])
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max <= 0:
            return 0.0

        return d_min / d_max
        
    def _compute_combined_objective(self, points, uniformity_weight=0.3):
        """Compute combined objective function that balances distance ratio and uniformity."""
        ratio = self._compute_distance_ratio(points)
        uniformity = self._compute_voronoi_uniformity_metric(points)
        
        # Combine objectives: higher ratio is better, lower uniformity metric is better
        # We want to maximize (ratio - uniformity_weight * uniformity)
        return ratio - uniformity_weight * uniformity
        
    def _compute_voronoi_gradients(self, points):
        """Compute approximate gradients using Voronoi-based information."""
        # This is a simplified gradient approach based on Voronoi structure
        # In practice, this could be more sophisticated
        return np.random.normal(0, 0.001, points.shape)
        
    def _local_voronoi_refinement(self, points, max_iterations=100):
        """Perform local refinement using Voronoi-based geometric considerations."""
        current_points = points.copy()
        current_obj = self._compute_combined_objective(current_points)
        
        best_points = current_points.copy()
        best_obj = current_obj
        
        for iteration in range(max_iterations):
            # Select points to perturb based on their influence
            # Points in regions with highly variable Voronoi cells get more attention
            try:
                # Get Voronoi structure for current points
                vor = Voronoi(current_points)
                
                # Calculate influence weights based on cell sizes
                weights = np.ones(len(current_points))
                try:
                    areas = []
                    for i, region in enumerate(vor.regions):
                        if not any(v == -1 for v in region) and len(region) >= 3:
                            polygon = [vor.vertices[j] for j in region]
                            if len(polygon) >= 3:
                                area = 0.5 * abs(sum(polygon[j][0] * polygon[(j+1)%len(polygon)][1] - 
                                                   polygon[(j+1)%len(polygon)][0] * polygon[j][1] 
                                                   for j in range(len(polygon))))
                                areas.append(area)
                    
                    if len(areas) > 0:
                        mean_area = np.mean(areas)
                        # Weight points based on deviation from mean area
                        for i in range(len(current_points)):
                            # Estimate area for this point's cell (simplified)
                            weights[i] = 1.0 / (1.0 + np.abs(mean_area - 0.01) * 10)  # Approximation
                except:
                    pass  # Fall back to uniform weights
                
                # Perturb points with weights
                for i in range(len(current_points)):
                    if np.random.random() < 0.3:  # 30% chance to perturb each point
                        # Adjust perturbation magnitude based on weights
                        magnitude = 0.005 * (1.0 + weights[i] * 0.5)
                        
                        # Apply perturbation
                        delta = np.random.normal(0, magnitude, 2)
                        new_points = current_points.copy()
                        new_points[i] += delta
                        
                        # Keep within bounds
                        new_points[i] = np.clip(new_points[i], 0, 1)
                        
                        # Evaluate
                        new_obj = self._compute_combined_objective(new_points)
                        
                        if new_obj > current_obj:
                            current_points = new_points
                            current_obj = new_obj
                            if new_obj > best_obj:
                                best_points = current_points.copy()
                                best_obj = new_obj
                                
            except Exception:
                # Fallback to simple random perturbations
                for i in range(len(current_points)):
                    if np.random.random() < 0.2:
                        delta = np.random.normal(0, 0.003, 2)
                        new_points = current_points.copy()
                        new_points[i] += delta
                        new_points[i] = np.clip(new_points[i], 0, 1)
                        
                        new_obj = self._compute_combined_objective(new_points)
                        
                        if new_obj > current_obj:
                            current_points = new_points
                            current_obj = new_obj
                            if new_obj > best_obj:
                                best_points = current_points.copy()
                                best_obj = new_obj
                                
        return best_points
        
    def _global_voronoi_optimization(self, points, max_iterations=1000):
        """Perform global optimization using Voronoi-based insights."""
        current_points = points.copy()
        current_ratio = self._compute_distance_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Store recent improvements for adaptive behavior
        recent_improvements = []
        patience = 50
        
        for iteration in range(max_iterations):
            # Create candidate point configuration
            candidate_points = current_points.copy()
            
            # Select multiple points to perturb (adaptive selection)
            num_perturb = max(2, len(current_points) // 3)
            perturb_indices = np.random.choice(len(current_points), num_perturb, replace=False)
            
            # Apply coordinated perturbations based on Voronoi structure
            for idx in perturb_indices:
                # Perturb based on whether point is in dense or sparse region
                try:
                    vor = Voronoi(current_points)
                    # Simplified approach: perturb based on iteration and index
                    magnitude = 0.01 * (1.0 + 0.1 * np.sin(iteration * 0.1 + idx * 0.5))
                    delta = np.random.normal(0, magnitude, 2)
                    candidate_points[idx] += delta
                except:
                    # Fallback
                    magnitude = 0.005
                    delta = np.random.normal(0, magnitude, 2)
                    candidate_points[idx] += delta
                
                # Keep within bounds
                candidate_points[idx] = np.clip(candidate_points[idx], 0, 1)
            
            # Evaluate candidate
            candidate_ratio = self._compute_distance_ratio(candidate_points)
            
            # Accept or reject (simple acceptance rule)
            if candidate_ratio > current_ratio or np.random.random() < 0.05:
                current_points = candidate_points
                current_ratio = candidate_ratio
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
                    recent_improvements = []  # Reset patience counter
                else:
                    recent_improvements.append(current_ratio)
                    if len(recent_improvements) > patience:
                        recent_improvements.pop(0)
            
            # Early stopping based on improvement rate
            if len(recent_improvements) >= 10:
                improvement_rate = np.mean(recent_improvements[-10:]) - np.mean(recent_improvements[:10])
                if improvement_rate < 1e-6:
                    break  # Stop if not improving significantly
                    
        return best_points
    
    def optimize(self, max_iterations=2000):
        """Main optimization routine using Voronoi-based approach."""
        # Step 1: Generate initial configuration using Voronoi principles
        points = self._create_initial_voronoi_layout()
        
        # Step 2: Local refinement to improve Voronoi uniformity
        points = self._local_voronoi_refinement(points, max_iterations=500)
        
        # Step 3: Global optimization with Voronoi-aware perturbations  
        points = self._global_voronoi_optimization(points, max_iterations=1000)
        
        # Step 4: Final local refinement
        points = self._local_voronoi_refinement(points, max_iterations=300)
        
        # Step 5: Final improvement using distance-based optimization
        final_ratio = self._compute_distance_ratio(points)
        best_points = points.copy()
        
        # Try a few more iterations of focused improvement
        for _ in range(200):
            candidate_points = points.copy()
            # Focus on points that are likely to help minimum distance
            idx = np.random.randint(len(points))
            magnitude = 0.001
            candidate_points[idx] += np.random.normal(0, magnitude, 2)
            candidate_points[idx] = np.clip(candidate_points[idx], 0, 1)
            
            candidate_ratio = self._compute_distance_ratio(candidate_points)
            
            if candidate_ratio > final_ratio:
                points = candidate_points
                final_ratio = candidate_ratio
                best_points = points.copy()
        
        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    # Create optimizer and run optimization
    optimizer = VoronoiBasedOptimizer(num_points=16, seed=42)
    result = optimizer.optimize(max_iterations=2000)
    
    return result

# EVOLVE-BLOCK-END