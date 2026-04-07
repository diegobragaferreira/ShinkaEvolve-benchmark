# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time

class PointConfiguration:
    """Manages point configurations and boundary handling."""
    
    def __init__(self, points):
        self.points = points.astype(np.float64)
        self.n = len(points)
        
    def reflect_boundaries(self):
        """Reflect points that go out of bounds."""
        for i in range(self.n):
            if self.points[i, 0] < 0:
                self.points[i, 0] = -self.points[i, 0]
            elif self.points[i, 0] > 1:
                self.points[i, 0] = 2 - self.points[i, 0]
            if self.points[i, 1] < 0:
                self.points[i, 1] = -self.points[i, 1]
            elif self.points[i, 1] > 1:
                self.points[i, 1] = 2 - self.points[i, 1]
                
    def clip_boundaries(self):
        """Clip points to stay within bounds."""
        self.points[:, 0] = np.clip(self.points[:, 0], 0, 1)
        self.points[:, 1] = np.clip(self.points[:, 1], 0, 1)
        
    def copy(self):
        """Return a copy of the configuration."""
        return PointConfiguration(self.points.copy())

class DistanceCalculator:
    """Handles distance computations for point configurations."""
    
    @staticmethod
    def calculate_ratio(points):
        """Calculate min/max distance ratio."""
        if len(points) < 2:
            return 0
            
        # Compute pairwise distances efficiently
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        if distances.size == 0:
            return 0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 0
            
        return d_min / d_max

class VoronoiAnalyzer:
    """Analyzes Voronoi diagrams for optimization guidance."""
    
    @staticmethod
    def analyze_voronoi_cell_areas(points):
        """Analyze Voronoi cell areas to identify problematic regions."""
        try:
            vor = Voronoi(points)
            # Measure variance of cell areas - more uniform cells are better
            areas = []
            for i in range(len(points)):
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 0:
                    # Simple area calculation for convex polygons
                    vertices = np.array([vor.vertices[j] for j in region if j >= 0])
                    if len(vertices) >= 3:
                        # Calculate area using shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        areas.append(area)
            
            if areas:
                return np.var(areas)
            return 0
        except:
            return 0

class CoolingScheduler:
    """Manages adaptive cooling schedules for simulated annealing."""
    
    def __init__(self, initial_temp=0.3, cooling_rate=0.9996, min_temp=1e-6):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.current_temp = initial_temp
        self.iteration = 0
        self.last_improvement = 0
        self.improvement_count = 0
        
    def update_temperature(self, has_improved, max_iterations):
        """Update temperature based on optimization progress."""
        self.iteration += 1
        
        if has_improved:
            self.last_improvement = self.iteration
            self.improvement_count += 1
            
        # Accelerate cooling if no improvement for a while
        if self.iteration - self.last_improvement > 300:
            self.cooling_rate = min(self.cooling_rate * 0.999, 0.9999)
            
        self.current_temp *= self.cooling_rate
        
        # Reset cooling rate after significant improvement
        if has_improved and self.iteration - self.last_improvement < 100:
            self.cooling_rate = 0.9996
            
        return self.current_temp < self.min_temp

class OptimizationEngine:
    """Main optimization engine implementing simulated annealing."""
    
    def __init__(self, max_iterations=8000):
        self.max_iterations = max_iterations
        self.scheduler = CoolingScheduler()
        
    def optimize(self, initial_points):
        """Run optimization using simulated annealing with Voronoi guidance."""
        current_config = PointConfiguration(initial_points)
        current_ratio = DistanceCalculator.calculate_ratio(current_config.points)
        
        best_config = current_config.copy()
        best_ratio = current_ratio
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            # Update cooling schedule
            should_stop = self.scheduler.update_temperature(
                current_ratio > best_ratio, 
                self.max_iterations
            )
            
            if should_stop:
                break
                
            # Get targeted perturbation
            new_config, new_ratio = self._get_targeted_perturbation(
                current_config, 
                self.scheduler.current_temp, 
                iteration
            )
            
            # Accept or reject using Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / self.scheduler.current_temp):
                current_config = new_config
                current_ratio = new_ratio
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_config = current_config.copy()
        
        return best_config.points, best_ratio
    
    def _get_targeted_perturbation(self, points_config, temperature, iteration):
        """Generate a targeted perturbation based on Voronoi analysis."""
        points = points_config.points
        vor = Voronoi(points)
        
        # Identify the point with the smallest Voronoi cell area (likely in a dense region)
        cell_areas = []
        for i in range(len(points)):
            try:
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 0:
                    vertices = np.array([vor.vertices[j] for j in region if j >= 0])
                    if len(vertices) >= 3:
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        cell_areas.append(area)
                    else:
                        cell_areas.append(0)
                else:
                    cell_areas.append(0)
            except:
                cell_areas.append(0)
        
        # Find the most problematic point (smallest cell area)
        if len(cell_areas) > 0:
            min_area_idx = np.argmin(cell_areas)
        else:
            min_area_idx = np.random.randint(len(points))
        
        # Create perturbation focused on this point
        new_points = points.copy()
        perturbation_magnitude = temperature * 0.07
        
        # Sample several candidate positions and choose the best one
        best_candidate_pos = None
        best_candidate_ratio = current_ratio
        
        # Sample more directions for better exploration
        num_samples = 20
        for _ in range(num_samples):
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.exponential(perturbation_magnitude)
            dx = distance * np.cos(angle)
            dy = distance * np.sin(angle)
            
            candidate_point = points[min_area_idx].copy()
            candidate_point[0] += dx
            candidate_point[1] += dy
            
            # Check boundaries and reflect
            if candidate_point[0] < 0:
                candidate_point[0] = -candidate_point[0]
            elif candidate_point[0] > 1:
                candidate_point[0] = 2 - candidate_point[0]
            if candidate_point[1] < 0:
                candidate_point[1] = -candidate_point[1]
            elif candidate_point[1] > 1:
                candidate_point[1] = 2 - candidate_point[1]
            
            # Test this move
            test_points = new_points.copy()
            test_points[min_area_idx] = candidate_point
            
            test_ratio = DistanceCalculator.calculate_ratio(test_points)
            if test_ratio > best_candidate_ratio:
                best_candidate_ratio = test_ratio
                best_candidate_pos = candidate_point.copy()
        
        if best_candidate_pos is not None:
            new_points[min_area_idx] = best_candidate_pos
        else:
            # Fallback to simple perturbation
            new_points[min_area_idx, 0] += np.random.normal(0, perturbation_magnitude)
            new_points[min_area_idx, 1] += np.random.normal(0, perturbation_magnitude)
            # Reflect if out of bounds
            if new_points[min_area_idx, 0] < 0:
                new_points[min_area_idx, 0] = -new_points[min_area_idx, 0]
            elif new_points[min_area_idx, 0] > 1:
                new_points[min_area_idx, 0] = 2 - new_points[min_area_idx, 0]
            if new_points[min_area_idx, 1] < 0:
                new_points[min_area_idx, 1] = -new_points[min_area_idx, 1]
            elif new_points[min_area_idx, 1] > 1:
                new_points[min_area_idx, 1] = 2 - new_points[min_area_idx, 1]
        
        return PointConfiguration(new_points), best_candidate_ratio

def create_lattice_initialization():
    """Create initial structured lattice configuration."""
    # Create a 4x4 grid with slight perturbations
    rows, cols = 4, 4
    points = []
    
    for i in range(rows):
        for j in range(cols):
            x = j + 0.5 * (i % 2)
            y = i * np.sqrt(3)/2
            points.append([x, y])
    
    points = np.array(points)
    
    # Normalize to [0,1] x [0,1]
    max_x = cols - 0.5
    max_y = (rows - 1) * np.sqrt(3)/2
    
    points[:, 0] = points[:, 0] / max_x
    points[:, 1] = points[:, 1] / max_y
    
    # Add structured perturbations to break symmetry with higher variance
    noise_scale = 0.02
    points += np.random.normal(0, noise_scale, points.shape)
    
    # Ensure points stay within bounds using reflection instead of clipping
    for i in range(len(points)):
        if points[i, 0] < 0:
            points[i, 0] = -points[i, 0]
        elif points[i, 0] > 1:
            points[i, 0] = 2 - points[i, 0]
        if points[i, 1] < 0:
            points[i, 1] = -points[i, 1]
        elif points[i, 1] > 1:
            points[i, 1] = 2 - points[i, 1]
    
    return points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    np.random.seed(42)
    
    # Create initial configuration
    initial_points = create_lattice_initialization()
    
    # Run optimization
    optimizer = OptimizationEngine(max_iterations=8000)
    optimized_points, _ = optimizer.optimize(initial_points)
    
    return optimized_points

# EVOLVE-BLOCK-END
