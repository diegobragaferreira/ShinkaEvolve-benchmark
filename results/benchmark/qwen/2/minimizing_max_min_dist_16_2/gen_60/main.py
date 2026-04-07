# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
from typing import Tuple, List, Optional

class SpectralVoronoiEvolver:
    """Novel hybrid optimizer combining spectral clustering and voronoi-based evolution."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        
    def calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min/max distance ratio along with actual values."""
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
    
    def objective_function(self, x: np.ndarray) -> float:
        """Objective function to minimize (negative ratio)."""
        points = x.reshape(-1, self.dimension)
        ratio, _, _ = self.calculate_ratio(points)
        return -ratio
    
    def generate_clustered_initial(self) -> np.ndarray:
        """Generate initial configuration using cluster-aware pattern."""
        # Create base hexagonal pattern
        points = []
        rows = cols = 4
        
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Adjust spacing for better distribution
        spacing_x *= 0.85
        spacing_y *= 0.85
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Add small perturbations for non-uniformity
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                
                points.append([x, y])
        
        initial_points = np.array(points[:self.num_points])
        # Ensure all points are within bounds
        initial_points = np.clip(initial_points, 0.001, 0.999)
        return initial_points
    
    def generate_voronoi_initial(self) -> np.ndarray:
        """Generate initial configuration using voronoi-inspired pattern."""
        # Start with fibonacci-like arrangement but add voronoi influence
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        # Generate points with some voronoi-like distribution
        for i in range(self.num_points):
            # Use modified fibonacci approach 
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (phi * phi)
            
            # Add voronoi-like perturbations to avoid perfect regularity
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            # Map to [0.05, 0.95] range
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            # Add voronoi-style clustering influence
            if i % 4 == 0:
                x += np.random.normal(0, 0.05)
                y += np.random.normal(0, 0.05)
            elif i % 4 == 1:
                x -= np.random.normal(0, 0.03)
                y += np.random.normal(0, 0.03)
            elif i % 4 == 2:
                x += np.random.normal(0, 0.02)
                y -= np.random.normal(0, 0.02)
            else:
                x -= np.random.normal(0, 0.04)
                y -= np.random.normal(0, 0.04)
                
            points.append([x, y])
        
        initial_points = np.array(points)
        initial_points = np.clip(initial_points, 0.001, 0.999)
        return initial_points
    
    def generate_population(self) -> List[np.ndarray]:
        """Generate diverse starting population."""
        population = []
        
        # 1. Clustered initial pattern
        population.append(self.generate_clustered_initial())
        
        # 2. Voronoi-inspired pattern
        population.append(self.generate_voronoi_initial())
        
        # 3. Random pattern with geometric awareness
        np.random.seed(42)
        random_points = np.random.uniform(0.05, 0.95, (self.num_points, 2))
        population.append(random_points)
        
        # 4. Modified grid pattern
        grid_points = []
        side = int(math.ceil(math.sqrt(self.num_points)))
        for i in range(side):
            for j in range(side):
                if len(grid_points) >= self.num_points:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                grid_points.append([x, y])
        population.append(np.array(grid_points[:self.num_points]))
        
        # 5. Add perturbed versions with different magnitudes
        for base_points in population[:3]:  # Perturb first three
            for perturbation_mag in [0.015, 0.025, 0.035]:
                perturbed = base_points + np.random.normal(0, perturbation_mag, base_points.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                population.append(perturbed)
        
        return population
    
    def adaptive_optimize(self, x0: np.ndarray, max_iter: int = 150) -> np.ndarray:
        """Adaptive optimization with method switching."""
        try:
            # Try L-BFGS-B first (often faster)
            result = minimize(
                self.objective_function,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if result.success:
                return result.x.reshape(-1, self.dimension)
        except Exception:
            pass
            
        # Fallback to SLSQP  
        try:
            result = minimize(
                self.objective_function,
                x0,
                method='SLSQP',
                bounds=self.bounds,
                options={'maxiter': max_iter}
            )
            
            if result.success:
                return result.x.reshape(-1, self.dimension)
        except Exception:
            pass
            
        # Last resort - return original points
        return x0.reshape(-1, self.dimension)
    
    def diversity_score(self, population: List[np.ndarray]) -> float:
        """Measure diversity of population."""
        if len(population) < 2:
            return 0.0
        
        # Calculate average distance between population centroids
        centroids = [np.mean(points, axis=0) for points in population]
        distances = []
        
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def evolve_population(self, population: List[np.ndarray], max_generations: int = 5) -> np.ndarray:
        """Evolve population through generations with adaptive strategies."""
        best_ratio = -np.inf
        best_points = None
        
        # Track convergence
        prev_best_ratio = -np.inf
        stagnation_count = 0
        
        for generation in range(max_generations):
            generation_best_ratio = -np.inf
            generation_best_points = None
            
            # Optimize each individual in current population
            for i, config in enumerate(population):
                # Adaptive optimization based on generation
                iter_count = 100 if generation < 3 else 150
                optimized_points = self.adaptive_optimize(config.flatten(), iter_count)
                
                ratio, _, _ = self.calculate_ratio(optimized_points)
                
                if ratio > generation_best_ratio:
                    generation_best_ratio = ratio
                    generation_best_points = optimized_points.copy()
                    
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
            
            # Check for stagnation
            if abs(generation_best_ratio - prev_best_ratio) < 1e-6:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_best_ratio = generation_best_ratio
            
            # If stagnated, regenerate population with higher diversity
            if stagnation_count >= 2 and generation < max_generations - 1:
                # Generate more diverse population
                population = self.generate_population()
                stagnation_count = 0
            else:
                # Create next generation by perturbing best performers
                if generation_best_points is not None:
                    population = [generation_best_points]
                    # Add 2 more diverse configurations
                    additional = self.generate_population()[:2]
                    population.extend(additional)
            
            # Early termination for good solutions
            if best_ratio > 0.3:
                break
                
        return best_points if best_points is not None else population[0]

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize evolver with novel hybrid approach
    evolver = SpectralVoronoiEvolver(16, 2)
    
    # Generate diverse initial population
    population = evolver.generate_population()
    
    # Evolve population through multiple generations
    best_points = evolver.evolve_population(population, max_generations=4)
    
    # Final refinement with more iterations
    final_points = evolver.adaptive_optimize(best_points.flatten(), 200)
    
    return final_points

# EVOLVE-BLOCK-END
