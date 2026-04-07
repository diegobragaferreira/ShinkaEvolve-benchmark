# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

class PointDistributionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio."""
    
    def __init__(self, n_points=14, seed=42):
        self.n_points = n_points
        self.seed = seed
        np.random.seed(seed)
        
    def calculate_min_max_ratio(self, points):
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
    
    def initialize_regular_polyhedron(self):
        """Initialize points based on regular icosahedron vertices."""
        # Regular icosahedron vertices (normalized)
        phi = (1 + np.sqrt(5)) / 2
        vertices = [
            (-1, 0, phi), (1, 0, phi), (-1, 0, -phi), (1, 0, -phi),
            (0, phi, 1), (0, phi, -1), (0, -phi, 1), (0, -phi, -1),
            (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0)
        ]
        vertices = np.array(vertices)
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        # Add more points by taking edge midpoints for better distribution
        edges = []
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                if abs(dist - 2) < 0.1:  # approximately the edge length
                    edges.append((i, j))
        
        # Add midpoints of edges
        additional_points = []
        for i, j in edges[:2]:  # Take first 2 edges for simplicity
            midpoint = (vertices[i] + vertices[j]) / 2
            midpoint = midpoint / np.linalg.norm(midpoint)  # normalize
            additional_points.append(midpoint)
        
        # Combine and ensure we have proper number of points
        all_points = np.vstack([vertices, additional_points])
        if len(all_points) > self.n_points:
            # Select points that are well spread
            return all_points[:self.n_points]
        elif len(all_points) < self.n_points:
            # Fill with fibonacci points
            fib_points = self.fibonacci_sphere(self.n_points - len(all_points))
            return np.vstack([all_points, fib_points])
        else:
            return all_points
    
    def project_to_unit_cube(self, points):
        """Project points to unit cube [0,1]^3."""
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
    
    def optimize_with_differential_evolution(self, initial_points, max_iter=500):
        """Use differential evolution for global optimization."""
        bounds = [(-1, 1), (-1, 1), (-1, 1)] * len(initial_points)
        
        def objective(x_flat):
            points = x_flat.reshape(-1, 3)
            # Keep points on unit sphere constraint
            for i in range(len(points)):
                norm = np.linalg.norm(points[i])
                if norm > 0:
                    points[i] = points[i] / norm
            
            ratio = self.calculate_min_max_ratio(points)
            return -ratio  # negative because we want to maximize
        
        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iter,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=self.seed,
            disp=False
        )
        
        optimized_points = result.x.reshape(-1, 3)
        # Project back to unit sphere
        for i in range(len(optimized_points)):
            norm = np.linalg.norm(optimized_points[i])
            if norm > 0:
                optimized_points[i] = optimized_points[i] / norm
        
        return optimized_points, -result.fun
    
    def local_refinement(self, points, iterations=200):
        """Local refinement using gradient-based approach."""
        # Convert points to flat array for optimization
        initial_flat = points.flatten()
        
        def objective(flat_points):
            points_local = flat_points.reshape(-1, 3)
            # Keep points on unit sphere
            for i in range(len(points_local)):
                norm = np.linalg.norm(points_local[i])
                if norm > 0:
                    points_local[i] = points_local[i] / norm
            return -self.calculate_min_max_ratio(points_local)  # negative for maximization
        
        # Use L-BFGS-B for local refinement
        try:
            result = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                options={'maxiter': iterations},
                tol=1e-6
            )
            refined_points = result.x.reshape(-1, 3)
            # Project back to sphere
            for i in range(len(refined_points)):
                norm = np.linalg.norm(refined_points[i])
                if norm > 0:
                    refined_points[i] = refined_points[i] / norm
            return refined_points, -result.fun
        except:
            return points, self.calculate_min_max_ratio(points)
    
    def run_optimization(self):
        """Main optimization workflow."""
        # Try multiple initialization strategies and pick the best
        best_ratio = 0.0
        best_points = None
        
        # Strategy 1: Fibonacci sphere initialization
        init1 = self.fibonacci_sphere(self.n_points)
        ratio1 = self.calculate_min_max_ratio(init1)
        print(f"Fibonacci initialization ratio: {ratio1}")
        
        if ratio1 > best_ratio:
            best_ratio = ratio1
            best_points = init1.copy()
        
        # Strategy 2: Icosahedron-based initialization  
        try:
            init2 = self.initialize_regular_polyhedron()
            ratio2 = self.calculate_min_max_ratio(init2)
            print(f"Icosahedron initialization ratio: {ratio2}")
            
            if ratio2 > best_ratio:
                best_ratio = ratio2
                best_points = init2.copy()
        except:
            pass
        
        # Strategy 3: Random initialization
        init3 = np.random.uniform(-1, 1, (self.n_points, 3))
        # Normalize to unit sphere
        for i in range(len(init3)):
            norm = np.linalg.norm(init3[i])
            if norm > 0:
                init3[i] = init3[i] / norm
        ratio3 = self.calculate_min_max_ratio(init3)
        print(f"Random initialization ratio: {ratio3}")
        
        if ratio3 > best_ratio:
            best_ratio = ratio3
            best_points = init3.copy()
        
        # Perform global optimization on best initialization
        if best_points is not None:
            optimized_points, optimized_ratio = self.optimize_with_differential_evolution(best_points)
            
            # Do local refinement
            final_points, final_ratio = self.local_refinement(optimized_points)
            
            # Final check and normalization
            if final_ratio > best_ratio:
                points_to_return = final_points
            else:
                points_to_return = best_points
        else:
            # Fallback to simple initialization
            points_to_return = self.fibonacci_sphere(self.n_points)
        
        # Normalize to unit cube [0,1]^3
        points_in_cube = self.project_to_unit_cube(points_to_return)
        
        return points_in_cube

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    optimizer = PointDistributionOptimizer(n_points=14, seed=42)
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END