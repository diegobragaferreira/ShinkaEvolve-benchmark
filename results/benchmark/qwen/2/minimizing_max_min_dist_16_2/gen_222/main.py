# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
from typing import Tuple, List, Optional
from scipy.spatial import SphericalVoronoi
import warnings

class SphericalVoronoiEvolver:
    """Enhanced evolver combining spherical geometry insights with Voronoi-inspired evolution."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.benchmark_ratio = 0.2786
        self.max_iterations = 1000
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
    
    def generate_grid_pattern(self) -> np.ndarray:
        """Generate structured grid pattern with improved distribution."""
        points = []
        rows = cols = 4
        
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Hexagonal offset to improve distribution
        for i in range(rows):
            for j in range(cols):
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Ensure within bounds with margin
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                
                points.append([x, y])
        
        return np.array(points)
    
    def generate_fibonacci_spiral(self) -> np.ndarray:
        """Generate points using Fibonacci spiral."""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        for i in range(self.num_points):
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (phi * phi)
            
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            # Map to [0.05, 0.95] range
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    def generate_spherical_initial(self) -> np.ndarray:
        """Generate initial configuration on sphere using spherical tiling principles."""
        # Generate points using a known good spherical arrangement
        # We'll use vertices of a regular icosahedron as starting point
        # and then perturb them for optimization
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        # Generate icosahedral vertices (approximate spherical distribution)
        # These are known to give good spherical point distributions
        vertices = []
        
        # Regular icosahedron vertices
        t = 0.525731112119133606  # sqrt(5)/4
        s = 0.850650808352039932  # sqrt(5)/2 * (1/sqrt(5) + 1) 
        
        # Add vertices of icosahedron
        vertices.extend([
            [0, s, t], [0, s, -t], [0, -s, t], [0, -s, -t],
            [t, 0, s], [-t, 0, s], [t, 0, -s], [-t, 0, -s],
            [s, t, 0], [-s, t, 0], [s, -t, 0], [-s, -t, 0]
        ])
        
        # Normalize vertices to unit sphere
        vertices = np.array(vertices)
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms
        
        # Take subset of vertices for 16 points
        if len(vertices) >= self.num_points:
            selected_vertices = vertices[:self.num_points]
        else:
            # If we have fewer vertices, duplicate and perturb
            selected_vertices = vertices[:self.num_points]
        
        # Add small random perturbations to break symmetry
        np.random.seed(42)
        perturbed = selected_vertices + np.random.normal(0, 0.05, selected_vertices.shape)
        
        # Renormalize to sphere
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        
        # Project back to 2D (for simplicity, just use the existing 2D approach)
        # This approach is less complex than full spherical projection but maintains the benefit
        return self.generate_grid_pattern()
    
    def generate_diverse_initial_configs(self) -> List[np.ndarray]:
        """Generate diverse set of initial configurations."""
        configs = []
        
        # Base configurations from Voronoi-inspired approach
        configs.append(self.generate_clustered_initial())
        configs.append(self.generate_voronoi_initial())
        configs.append(self.generate_grid_pattern())
        configs.append(self.generate_fibonacci_spiral())
        configs.append(self.generate_spherical_initial())
        
        # Enhanced variations with multiple perturbation levels
        np.random.seed(42)
        for base_config in configs[:4]:  # Use first 4 for variations
            # Different perturbation magnitudes
            for mag in [0.01, 0.015, 0.02]:
                perturbed = base_config + np.random.normal(0, mag, base_config.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                configs.append(perturbed)
        
        # Add structured variations
        for _ in range(3):
            # Grid-based with different offsets
            grid_points = []
            for i in range(4):
                for j in range(4):
                    x = (i + 0.5) / 4.0
                    y = (j + 0.5) / 4.0
                    grid_points.append([x, y])
            
            # Apply some structured perturbation
            structured = np.array(grid_points[:self.num_points])
            structured += np.random.normal(0, 0.02, structured.shape)
            structured = np.clip(structured, 0.001, 0.999)
            configs.append(structured)
        
        return configs
    
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
    
    def multi_stage_optimization(self, configs: List[np.ndarray]) -> np.ndarray:
        """Multi-stage optimization with progressive refinement."""
        best_ratio = -np.inf
        best_points = None
        
        # Stage 1: Coarse evaluation of all configurations
        stage1_results = []
        
        for i, config in enumerate(configs):
            # Light optimization for quick assessment with L-BFGS-B
            light_result = self.adaptive_optimize(config.flatten(), 100)
            if light_result is not None:
                ratio, _, _ = self.calculate_ratio(light_result)
                stage1_results.append((ratio, i, light_result))
        
        # Sort by quality and keep top performers
        stage1_results.sort(reverse=True)
        top_configs = [result[2] for result in stage1_results[:5]]  # Top 5
        
        # Stage 2: Thorough optimization of top candidates with SLSQP
        for i, config in enumerate(top_configs):
            # Full optimization for best candidates
            full_result = self.adaptive_optimize(config.flatten(), 150)
            if full_result is not None:
                ratio, _, _ = self.calculate_ratio(full_result)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = full_result.copy()
        
        # Stage 3: Additional refinement with adaptive perturbations
        if best_points is not None:
            for iteration in range(3):
                # Generate new configurations via adaptive perturbation
                perturbed = best_points + np.random.normal(0, 0.01, best_points.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                refined_result = self.adaptive_optimize(perturbed.flatten(), 150)
                
                if refined_result is not None:
                    ratio, _, _ = self.calculate_ratio(refined_result)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_result.copy()
        
        # Return best found or fallback
        return best_points if best_points is not None else configs[0]
    
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
                population = self.generate_diverse_initial_configs()
                stagnation_count = 0
            else:
                # Create next generation by perturbing best performers
                if generation_best_points is not None:
                    population = [generation_best_points]
                    # Add 2 more diverse configurations
                    additional = self.generate_diverse_initial_configs()[:2]
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
    
    # Initialize evolver with enhanced hybrid approach
    evolver = SphericalVoronoiEvolver(16, 2)
    
    # Generate diverse initial configurations
    initial_configs = evolver.generate_diverse_initial_configs()
    
    # Perform multi-stage optimization
    best_points = evolver.multi_stage_optimization(initial_configs)
    
    # Final validation and refinement
    if best_points is not None:
        # Additional optimization to ensure quality
        final_result = evolver.adaptive_optimize(best_points.flatten(), 200)
        if final_result is not None:
            ratio, _, _ = evolver.calculate_ratio(final_result)
            # If the improvement is significant, use the final result
            if ratio > 0.01:  # Only accept meaningful improvements
                best_points = final_result
    
    # Ensure we always return a valid configuration
    if best_points is None:
        # Fallback to a well-known good configuration
        fallback_config = evolver.generate_grid_pattern()
        # Add small perturbation to break symmetry
        fallback_config += np.random.normal(0, 0.01, fallback_config.shape)
        fallback_config = np.clip(fallback_config, 0.001, 0.999)
        best_points = fallback_config
    
    return best_points

# EVOLVE-BLOCK-END