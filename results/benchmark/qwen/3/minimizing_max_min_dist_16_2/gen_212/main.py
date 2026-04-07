# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist
import warnings
import math
from typing import List, Tuple, Optional
import time
from sklearn.cluster import KMeans

class VoronoiGuidedOptimizer:
    """Voronoi-guided optimization for point dispersion."""
    
    def __init__(self, n_points: int = 16, dimensions: int = 2, seed: int = 42, max_time_seconds: int = 180):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        self.max_time_seconds = max_time_seconds
        np.random.seed(seed)
    
    def _compute_voronoi_metrics(self, points: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Compute Voronoi-based metrics for point distribution."""
        try:
            vor = Voronoi(points)
            # Get Voronoi cell areas
            cell_areas = []
            for region in vor.filtered_regions:
                if len(region) > 0:
                    # Approximate area using polygon area formula
                    vertices = np.array([vor.vertices[i] for i in region])
                    if len(vertices) >= 3:
                        # Use shoelace formula for polygon area
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        cell_areas.append(area)
            
            # If no valid regions, compute naive minimum distance
            if not cell_areas:
                distances = pdist(points)
                if len(distances) > 0:
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    return d_min / d_max if d_max > 0 else 0, d_min, np.array([0.0] * self.n_points)
                else:
                    return 0, 0, np.array([0.0] * self.n_points)
            
            # Compute average cell area and std deviation
            avg_area = np.mean(cell_areas) if cell_areas else 0
            std_area = np.std(cell_areas) if len(cell_areas) > 1 else 0
            
            return avg_area / (std_area + 1e-10), np.min(cell_areas), np.array(cell_areas)
        except:
            # Fallback to simple distance computation
            distances = pdist(points)
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                return d_min / d_max if d_max > 0 else 0, d_min, np.array([0.0] * self.n_points)
            else:
                return 0, 0, np.array([0.0] * self.n_points)

    def _generate_voronoi_based_points(self) -> np.ndarray:
        """Generate points using Voronoi-based initialization."""
        # Generate points via k-means clustering of random points to ensure good spread
        # But also incorporate Voronoi-like principles
        
        # Start with random points
        points = np.random.rand(self.n_points, self.dimensions)
        
        # Use a modified approach inspired by Voronoi tessellation
        # Sample points from a grid with some randomness to avoid regular patterns
        grid_size = max(3, int(np.ceil(np.sqrt(self.n_points))))
        spacing = 1.0 / (grid_size - 1) if grid_size > 1 else 1.0
        
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < self.n_points:
                    # Add slight perturbation around grid points
                    x = max(0, min(1, i * spacing + np.random.normal(0, spacing * 0.15)))
                    y = max(0, min(1, j * spacing + np.random.normal(0, spacing * 0.15)))
                    points.append([x, y])
        
        points = np.array(points[:self.n_points])
        
        # Improve distribution using Lloyd relaxation (iterative centroid optimization)
        for _ in range(3):  # Few iterations to improve quality
            # Compute Voronoi diagram
            try:
                vor = Voronoi(points)
                new_points = []
                for i in range(len(points)):
                    # Find the Voronoi region for this point
                    region_idx = None
                    for j, region in enumerate(vor.filtered_regions):
                        if i in region:
                            region_idx = j
                            break
                    
                    if region_idx is not None and len(vor.filtered_regions[region_idx]) > 0:
                        # Compute centroid of Voronoi cell
                        vertices = np.array([vor.vertices[k] for k in vor.filtered_regions[region_idx]])
                        if len(vertices) > 0:
                            centroid = np.mean(vertices, axis=0)
                            # Ensure centroid stays within bounds
                            centroid = np.clip(centroid, 0, 1)
                            new_points.append(centroid)
                        else:
                            new_points.append(points[i])
                    else:
                        new_points.append(points[i])
                
                points = np.array(new_points)
            except:
                # If Voronoi computation fails, keep original points
                break
        
        return points

    def _generate_kmeans_based_points(self) -> np.ndarray:
        """Generate points using k-means clustering for better spread."""
        # Generate initial random points
        initial_points = np.random.rand(100, self.dimensions)
        
        # Cluster into n_points clusters
        kmeans = KMeans(n_clusters=self.n_points, random_state=self.seed, n_init=10)
        kmeans.fit(initial_points)
        
        # Use cluster centers as initial points
        points = kmeans.cluster_centers_
        
        # Clip to bounds
        points = np.clip(points, 0, 1)
        
        # Add some noise to break perfect symmetry
        noise_magnitude = 0.02
        points += np.random.normal(0, noise_magnitude, points.shape)
        points = np.clip(points, 0, 1)
        
        return points

    def _generate_fibonacci_sphere_points(self) -> np.ndarray:
        """Generate points using Fibonacci sphere-like distribution."""
        # For 2D, we'll use a similar principle with spiral pattern
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        
        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Map to 2D unit square
            x_mapped = (x + 1) / 2
            y_mapped = (z + 1) / 2
            
            points.append([np.clip(x_mapped, 0, 1), np.clip(y_mapped, 0, 1)])
        
        return np.array(points)

    def _compute_ratio(self, points: np.ndarray) -> float:
        """Compute the min/max distance ratio for given points."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return d_min / d_max

    def _voronoi_guided_local_search(self, points: np.ndarray, max_iter: int = 1000) -> Tuple[np.ndarray, float]:
        """Local search guided by Voronoi properties."""
        current_points = points.copy()
        current_ratio = self._compute_ratio(current_points)
        
        # Store best solution so far
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        for iteration in range(max_iter):
            # Create a neighbor by perturbing points based on Voronoi cell analysis
            neighbor_points = current_points.copy()
            
            # Select a subset of points to perturb (based on Voronoi cell sizes)
            # Points in smaller cells get more aggressive perturbation to expand their regions
            try:
                vor = Voronoi(current_points)
                cell_sizes = []
                for region in vor.filtered_regions:
                    if len(region) > 0:
                        vertices = np.array([vor.vertices[i] for i in region])
                        if len(vertices) >= 3:
                            x = vertices[:, 0]
                            y = vertices[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            cell_sizes.append(area)
                        else:
                            cell_sizes.append(0)
                    else:
                        cell_sizes.append(0)
                
                # Convert to array for easier handling
                cell_sizes = np.array(cell_sizes)
                
                # Determine which points to perturb (those with smaller cells)
                # Use percentile threshold
                if len(cell_sizes) > 0:
                    threshold = np.percentile(cell_sizes, 30)  # Top 30% smallest cells
                    indices_to_perturb = np.where(cell_sizes <= threshold)[0]
                else:
                    indices_to_perturb = []
                
                # If no small cells, just pick a random few
                if len(indices_to_perturb) == 0:
                    indices_to_perturb = np.random.choice(len(current_points), 
                                                        max(1, len(current_points) // 4), 
                                                        replace=False)
                
                # Perturb selected points with adaptive step sizes based on cell size
                for idx in indices_to_perturb:
                    # Smaller cell = larger stepsize to encourage expansion
                    step_size = 0.01 + (threshold - cell_sizes[idx]) * 0.01 if len(cell_sizes) > 0 else 0.01
                    
                    # Add perturbation
                    neighbor_points[idx, 0] += np.random.normal(0, step_size)
                    neighbor_points[idx, 1] += np.random.normal(0, step_size)
                    
                    # Keep within bounds
                    neighbor_points[idx, 0] = np.clip(neighbor_points[idx, 0], 0, 1)
                    neighbor_points[idx, 1] = np.clip(neighbor_points[idx, 1], 0, 1)
                    
            except:
                # Fallback to simple random perturbation
                idx = np.random.randint(0, len(current_points))
                step_size = 0.01
                neighbor_points[idx, 0] += np.random.normal(0, step_size)
                neighbor_points[idx, 1] += np.random.normal(0, step_size)
                neighbor_points[idx, 0] = np.clip(neighbor_points[idx, 0], 0, 1)
                neighbor_points[idx, 1] = np.clip(neighbor_points[idx, 1], 0, 1)
            
            # Calculate neighbor ratio
            neighbor_ratio = self._compute_ratio(neighbor_points)
            
            # Accept or reject the neighbor
            if neighbor_ratio > current_ratio:
                current_points = neighbor_points
                current_ratio = neighbor_ratio
                if neighbor_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = neighbor_ratio
            else:
                # Accept with probability based on difference
                delta = neighbor_ratio - current_ratio
                if delta < 0:  # Only accept worse solutions with probability
                    acceptance_prob = math.exp(delta * 100)  # Higher acceptance probability for large differences
                    if np.random.random() < acceptance_prob:
                        current_points = neighbor_points
                        current_ratio = neighbor_ratio
            
            # Early stopping if improvement is minimal
            if iteration > 100 and abs(best_ratio - current_ratio) < 1e-8:
                break
        
        return best_points, best_ratio

    def optimize(self) -> np.ndarray:
        """Main optimization routine using Voronoi-guided evolutionary approach."""
        start_time = time.time()
        best_points = None
        best_ratio = -np.inf
        
        # Generate diverse initial configurations using Voronoi-inspired methods
        initial_configs = []
        
        # 1. Voronoi-based initialization
        initial_configs.append(self._generate_voronoi_based_points())
        
        # 2. KMeans-based initialization
        initial_configs.append(self._generate_kmeans_based_points())
        
        # 3. Fibonacci-inspired initialization
        initial_configs.append(self._generate_fibonacci_sphere_points())
        
        # 4. Random initialization with symmetry breaking
        random_points = np.random.rand(self.n_points, self.dimensions)
        # Add some structured perturbation to break perfect symmetry
        for i in range(self.n_points):
            random_points[i, 0] += np.sin(i * 0.7) * 0.01
            random_points[i, 1] += np.cos(i * 0.7) * 0.01
        random_points = np.clip(random_points, 0, 1)
        initial_configs.append(random_points)
        
        # Try each configuration with Voronoi-guided optimization
        for i, initial_config in enumerate(initial_configs):
            if time.time() - start_time > self.max_time_seconds - 5:
                break
                
            try:
                # Initial evaluation
                initial_ratio = self._compute_ratio(initial_config)
                current_points = initial_config.copy()
                current_ratio = initial_ratio
                
                # Multi-phase optimization
                phase1_points = current_points.copy()
                phase1_ratio = current_ratio
                
                # Phase 1: Voronoi-guided local search
                if time.time() - start_time < self.max_time_seconds - 10:
                    phase1_points, phase1_ratio = self._voronoi_guided_local_search(
                        phase1_points, max_iter=500
                    )
                
                # Phase 2: Simulated annealing refinement with Voronoi awareness
                if time.time() - start_time < self.max_time_seconds - 10 and phase1_ratio > 0.1:
                    # Simple simulated annealing variant focused on Voronoi-enhanced moves
                    current_points = phase1_points.copy()
                    current_ratio = phase1_ratio
                    
                    temp = 1.0
                    for iteration in range(500):
                        if time.time() - start_time > self.max_time_seconds - 5:
                            break
                            
                        # Create neighbor by Voronoi-aware perturbation
                        neighbor_points = current_points.copy()
                        idx = np.random.randint(0, len(current_points))
                        
                        # Adjust step size based on Voronoi cell information
                        try:
                            vor = Voronoi(current_points)
                            if len(vor.filtered_regions) > idx and len(vor.filtered_regions[idx]) > 0:
                                # Get average cell size for adjustment
                                cell_vertices = [vor.vertices[i] for i in vor.filtered_regions[idx]]
                                if len(cell_vertices) >= 3:
                                    cell_area = self._compute_polygon_area(cell_vertices)
                                    # Larger area = smaller stepsize (points don't need to move much)
                                    step_size = max(0.001, 0.03 * (1 / (cell_area + 1e-6)))
                                else:
                                    step_size = 0.02
                            else:
                                step_size = 0.02
                        except:
                            step_size = 0.02
                        
                        # Perturb point
                        neighbor_points[idx, 0] += np.random.normal(0, step_size)
                        neighbor_points[idx, 1] += np.random.normal(0, step_size)
                        
                        # Keep within bounds
                        neighbor_points[idx, 0] = np.clip(neighbor_points[idx, 0], 0, 1)
                        neighbor_points[idx, 1] = np.clip(neighbor_points[idx, 1], 0, 1)
                        
                        # Calculate neighbor ratio
                        neighbor_ratio = self._compute_ratio(neighbor_points)
                        
                        # Accept or reject
                        if neighbor_ratio > current_ratio:
                            current_points = neighbor_points
                            current_ratio = neighbor_ratio
                        else:
                            delta = neighbor_ratio - current_ratio
                            if delta < 0:
                                acceptance_prob = math.exp(delta / temp)
                                if np.random.random() < acceptance_prob:
                                    current_points = neighbor_points
                                    current_ratio = neighbor_ratio
                        
                        temp *= 0.999  # Cooling schedule
                        
                        # Update best if needed
                        if current_ratio > best_ratio:
                            best_ratio = current_ratio
                            best_points = current_points.copy()
                        
                        # Early stopping
                        if temp < 1e-8:
                            break
                else:
                    # Use phase 1 result
                    if phase1_ratio > best_ratio:
                        best_ratio = phase1_ratio
                        best_points = phase1_points.copy()
                
                # Update overall best
                if phase1_ratio > best_ratio:
                    best_ratio = phase1_ratio
                    best_points = phase1_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Error in optimization round {i}: {str(e)}")
                continue
        
        # Final validation
        if best_points is not None:
            final_ratio = self._compute_ratio(best_points)
            print(f"Final optimized ratio: {final_ratio:.6f}")
            return best_points
        else:
            # Fallback to first configuration
            return initial_configs[0] if initial_configs else np.random.rand(self.n_points, self.dimensions)

    def _compute_polygon_area(self, vertices: List[List[float]]) -> float:
        """Compute area of polygon given vertices using shoelace formula."""
        if len(vertices) < 3:
            return 0.0
        
        x = np.array([v[0] for v in vertices])
        y = np.array([v[1] for v in vertices])
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    optimizer = VoronoiGuidedOptimizer(n_points=16, dimensions=2, seed=42, max_time_seconds=180)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END