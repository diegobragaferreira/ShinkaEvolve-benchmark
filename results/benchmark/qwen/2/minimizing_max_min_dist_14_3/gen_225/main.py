# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time

class SphereSimulatedAnnealingOptimizer:
    def __init__(self, n_points=14, seed=42):
        self.n_points = n_points
        self.seed = seed
        np.random.seed(seed)
        
    def fibonacci_sphere(self, n):
        """Generate n points on a sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    def project_to_sphere(self, points):
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms
    
    def compute_min_max_ratio(self, points):
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
    
    def compute_local_density_score(self, points, point_idx):
        """Compute a score indicating how locally dense a point is."""
        # Get distances to all other points
        distances = cdist([points[point_idx]], points)[0]
        distances = distances[distances > 0]  # Remove self-distance
        
        if len(distances) == 0:
            return 0.0
            
        # Calculate various density measures
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)
        
        # Score based on how close neighbors are compared to average
        # Lower values indicate denser regions
        density_score = min_dist / (avg_dist + 1e-12)
        
        return density_score
    
    def select_point_for_perturbation(self, points, recent_improvements):
        """Select point to perturb based on local density analysis."""
        # Select point based on density - prefer points in less dense regions
        scores = []
        for i in range(len(points)):
            score = self.compute_local_density_score(points, i)
            scores.append(score)
        
        scores = np.array(scores)
        
        # If we haven't improved recently, prefer less dense points
        # Otherwise, select randomly with some preference for sparse points
        if len(recent_improvements) > 10 and np.sum(recent_improvements[-10:]) < 3:
            # Recent stagnation - focus on dense regions that might need expansion
            probabilities = 1.0 / (scores + 1e-8)  # Prefer less dense points
        else:
            # Otherwise, favor more uniform distribution
            probabilities = np.exp(-scores * 2)  # Exponentially decay with density
        
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Select point using weighted probability
        idx = np.random.choice(len(points), p=probabilities)
        return idx
    
    def adaptive_perturbation(self, points, point_idx, current_temp, current_ratio):
        """Generate adaptive perturbation based on temperature and current solution quality."""
        # Base perturbation size
        base_size = 0.01
        
        # Temperature-dependent scaling
        perturbation_scale = base_size * current_temp
        
        # Ratio-quality dependent scaling - if ratio is poor, allow larger moves
        if current_ratio < 0.2:
            perturbation_scale *= 3.0
        elif current_ratio < 0.3:
            perturbation_scale *= 2.0
        elif current_ratio < 0.4:
            perturbation_scale *= 1.5
        
        # Generate perturbation vector
        perturbation = np.random.randn(3) * perturbation_scale
        
        # Project to tangent plane of sphere at point_idx
        point = points[point_idx]
        # Perturbation in tangent plane: subtract component in normal direction
        normal_component = np.dot(perturbation, point)
        tangent_perturbation = perturbation - normal_component * point
        
        return tangent_perturbation
    
    def optimize_with_simulated_annealing(self, initial_points, max_time=350):
        """Optimize using simulated annealing with adaptive parameters."""
        points = initial_points.copy()
        current_ratio = self.compute_min_max_ratio(points)
        
        # Track best solution
        best_points = points.copy()
        best_ratio = current_ratio
        
        # Adaptive cooling schedule
        temperature = 0.1
        min_temperature = 1e-8
        base_cooling_rate = 0.9995
        
        # Track performance for adaptive cooling
        recent_improvements = []
        max_recent = 50
        
        # Iteration counters
        iteration = 0
        last_improvement = 0
        max_no_improvement = 2000
        
        start_time = time.time()
        
        while (temperature > min_temperature and 
               iteration < 500000 and 
               time.time() - start_time < max_time):
            
            # Select point to perturb based on density
            point_idx = self.select_point_for_perturbation(points, recent_improvements)
            
            # Generate perturbation
            perturbation = self.adaptive_perturbation(points, point_idx, temperature, current_ratio)
            
            # Apply perturbation
            new_points = points.copy()
            new_points[point_idx] += perturbation
            
            # Project back to sphere
            new_points[point_idx] = self.project_to_sphere(new_points[point_idx:point_idx+1])[0]
            
            # Compute new ratio
            new_ratio = self.compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                points = new_points
                current_ratio = new_ratio
                
                # Update best if improved
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()
                    last_improvement = iteration
                    
                # Record improvement
                recent_improvements.append(True)
            else:
                # Accept with probability based on temperature
                if np.random.rand() < np.exp((new_ratio - current_ratio) / temperature):
                    points = new_points
                    current_ratio = new_ratio
                    recent_improvements.append(True)
                else:
                    recent_improvements.append(False)
            
            # Trim recent improvements list
            if len(recent_improvements) > max_recent:
                recent_improvements.pop(0)
            
            # Adaptive cooling based on recent performance
            if len(recent_improvements) >= 10:
                recent_improvement_rate = np.mean(recent_improvements[-10:])
                
                # Adjust cooling rate based on improvement rate
                if recent_improvement_rate < 0.1:  # Very slow improvement
                    cooling_rate = base_cooling_rate * 1.1
                elif recent_improvement_rate > 0.4:  # Fast improvement
                    cooling_rate = base_cooling_rate * 0.99
                else:  # Moderate improvement
                    cooling_rate = base_cooling_rate
            
                # Apply cooling
                temperature = max(temperature * cooling_rate, min_temperature)
            else:
                # Default cooling
                temperature = max(temperature * base_cooling_rate, min_temperature)
            
            # Early stopping conditions
            iteration += 1
            
            # Stop if no improvement for a while
            if iteration - last_improvement > max_no_improvement:
                break
                
        return best_points, best_ratio
    
    def run_multi_start_optimization(self):
        """Run multiple optimizations with different initializations."""
        best_points = None
        best_ratio = 0.0
        
        # Strategy 1: Fibonacci sphere with slight random noise
        init1 = self.fibonacci_sphere(self.n_points)
        noise = np.random.normal(0, 0.005, init1.shape)
        init1 = init1 + noise
        init1 = self.project_to_sphere(init1)
        
        # Strategy 2: Icosahedron-based initialization
        try:
            init2 = self._icosahedron_initialization()
            init2 = self.project_to_sphere(init2)
        except:
            init2 = init1.copy()  # fallback
        
        # Strategy 3: Random points with small perturbations from fibonacci
        init3 = np.random.randn(self.n_points, 3)
        init3 = self.project_to_sphere(init3)
        
        # Strategy 4: Perturbed Fibonacci with noise
        init4 = self.fibonacci_sphere(self.n_points)
        noise = np.random.normal(0, 0.01, init4.shape)
        init4 = init4 + noise
        init4 = self.project_to_sphere(init4)
        
        initial_strategies = [init1, init2, init3, init4]
        
        for i, initial_points in enumerate(initial_strategies):
            try:
                optimized_points, ratio = self.optimize_with_simulated_annealing(initial_points, max_time=80)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
                print(f"Strategy {i+1}: ratio = {ratio:.6f}")
            except Exception as e:
                print(f"Strategy {i+1} failed: {str(e)}")
                continue
        
        # Final refinement with longer optimization if we found a good solution
        if best_points is not None:
            try:
                refined_points, refined_ratio = self.optimize_with_simulated_annealing(best_points, max_time=200)
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
            except:
                pass
        
        if best_points is None:
            # Fallback to basic fibonacci
            best_points = self.fibonacci_sphere(self.n_points)
            best_points = self.project_to_sphere(best_points)
            
        return best_points, best_ratio
    
    def _icosahedron_initialization(self):
        """Initialize using icosahedron vertices."""
        # Regular icosahedron vertices (normalized)
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            (-1, 0, phi), (1, 0, phi), (-1, 0, -phi), (1, 0, -phi),
            (0, phi, 1), (0, phi, -1), (0, -phi, 1), (0, -phi, -1),
            (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0)
        ])
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # For 14 points, we'll use 12 vertices plus 2 additional points
        # Add two points at extreme positions
        additional_points = np.array([[0, 0, 1], [0, 0, -1]])
        all_points = np.vstack([vertices, additional_points])
        
        # Ensure we have exactly n_points
        if len(all_points) > self.n_points:
            # Select the most spread out points
            return all_points[:self.n_points]
        else:
            # Add more points via fibonacci for remainder
            remaining = self.n_points - len(all_points)
            fib_points = self.fibonacci_sphere(remaining)
            return np.vstack([all_points, fib_points])

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = SphereSimulatedAnnealingOptimizer(n_points=14, seed=42)
    points, ratio = optimizer.run_multi_start_optimization()
    return points

# EVOLVE-BLOCK-END