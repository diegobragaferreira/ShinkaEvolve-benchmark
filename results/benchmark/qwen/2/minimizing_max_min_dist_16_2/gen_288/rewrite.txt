# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi
import time
from typing import Tuple, List, Optional
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    class VoronoiPointOptimizer:
        def __init__(self, num_points: int = 16):
            self.num_points = num_points
            self.best_points = None
            self.best_ratio = -np.inf
            
        def _compute_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
            """Compute min/max distance ratio."""
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
            
        def _smooth_penalty_constraints(self, points: np.ndarray, penalty_weight: float = 100.0) -> np.ndarray:
            """Apply smooth penalty for boundary constraints."""
            # Soft penalty for points outside bounds
            penalized = points.copy()
            penalty = 0.0
            
            # Check for out-of-bounds points and add soft penalties
            for i in range(len(points)):
                x, y = points[i]
                # Penalty for being too close to boundaries
                penalty_x = penalty_weight * (np.minimum(x, 1-x)**2)
                penalty_y = penalty_weight * (np.minimum(y, 1-y)**2)
                penalty += penalty_x + penalty_y
                
                # Clip but keep tracking
                penalized[i, 0] = np.clip(x, 0.001, 0.999)
                penalized[i, 1] = np.clip(y, 0.001, 0.999)
                
            return penalized
            
        def _voronoi_uniformity_score(self, points: np.ndarray) -> float:
            """Calculate uniformity based on Voronoi cell areas."""
            try:
                vor = Voronoi(points)
                # Sum of squared differences from mean area
                areas = []
                for region in vor.regions:
                    if len(region) > 0 and -1 not in region:
                        # Skip infinite regions
                        try:
                            area = self._polygon_area([vor.vertices[i] for i in region])
                            areas.append(area)
                        except:
                            continue
                            
                if len(areas) == 0:
                    return 0.0
                    
                mean_area = np.mean(areas)
                if mean_area == 0:
                    return 0.0
                    
                uniformity = 1.0 - np.std(areas) / mean_area
                return max(0.0, uniformity)
            except:
                return 0.0
                
        def _polygon_area(self, vertices: List[Tuple[float, float]]) -> float:
            """Calculate polygon area using shoelace formula."""
            if len(vertices) < 3:
                return 0.0
                
            n = len(vertices)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += vertices[i][0] * vertices[j][1]
                area -= vertices[j][0] * vertices[i][1]
            return abs(area) / 2.0
            
        def _generate_initial_voronoi_distribution(self) -> np.ndarray:
            """Generate points using Voronoi-inspired distribution."""
            # Start with regular grid pattern
            grid_size = 4
            points = []
            
            # Create a modified grid with some randomness
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) >= self.num_points:
                        break
                    # Offset odd rows for better distribution
                    x_offset = 0.25 if i % 2 == 1 else 0.0
                    x = (j + x_offset) * 0.25 + 0.125
                    y = i * 0.25 + 0.125
                    
                    # Add slight perturbation
                    x += np.random.normal(0, 0.01)
                    y += np.random.normal(0, 0.01)
                    
                    points.append([x, y])
                    
            initial = np.array(points[:self.num_points])
            
            # Further improve by applying a simple Lloyd relaxation step
            for _ in range(3):
                try:
                    vor = Voronoi(initial)
                    new_points = []
                    
                    for i in range(len(initial)):
                        region_vertices = []
                        # Collect vertices belonging to this point's Voronoi cell
                        for j in range(len(vor.point_region)):
                            if vor.point_region[j] == i:
                                region_vertices.extend(vor.regions[j])
                        
                        if len(region_vertices) > 0:
                            # Compute centroid of Voronoi cell
                            valid_vertices = [vor.vertices[k] for k in region_vertices if k >= 0]
                            if len(valid_vertices) >= 3:
                                # Simple centroid calculation (not perfect but works)
                                centroid_x = np.mean([v[0] for v in valid_vertices])
                                centroid_y = np.mean([v[1] for v in valid_vertices])
                                new_points.append([centroid_x, centroid_y])
                            else:
                                new_points.append(initial[i])
                        else:
                            new_points.append(initial[i])
                            
                    initial = np.array(new_points)
                    initial = np.clip(initial, 0.001, 0.999)
                except:
                    break
                    
            return initial
            
        def _local_search_with_adaptive_perturbation(self, points: np.ndarray, 
                                                   iterations: int = 50, 
                                                   temperature: float = 1.0) -> np.ndarray:
            """Local search with adaptive perturbations."""
            current_points = points.copy()
            current_ratio, _, _ = self._compute_ratio(current_points)
            
            for iter_num in range(iterations):
                # Adaptive temperature decay
                adaptive_temp = temperature * (1.0 - iter_num / iterations)
                
                # Generate candidate via perturbation
                candidate_points = current_points.copy()
                
                # Select random point to perturb
                idx = np.random.randint(0, len(candidate_points))
                
                # Adaptive perturbation magnitude based on current state
                if current_ratio > 0.25:
                    # High quality, small steps
                    perturbation_mag = 0.005
                elif current_ratio > 0.15:
                    # Medium quality, medium steps
                    perturbation_mag = 0.01
                else:
                    # Low quality, larger steps
                    perturbation_mag = 0.02
                    
                # Add noise scaled by adaptive temperature
                noise = np.random.normal(0, perturbation_mag * adaptive_temp, 2)
                candidate_points[idx] += noise
                
                # Apply boundary penalty
                candidate_points = self._smooth_penalty_constraints(candidate_points)
                
                # Evaluate candidate
                candidate_ratio, _, _ = self._compute_ratio(candidate_points)
                
                # Accept or reject with a probabilistic criterion for low temperatures
                if candidate_ratio > current_ratio or (
                    adaptive_temp < 0.1 and np.random.random() < np.exp((candidate_ratio - current_ratio) / (adaptive_temp + 1e-8))
                ):
                    current_points = candidate_points.copy()
                    current_ratio = candidate_ratio
                    
            return current_points
            
        def _multi_stage_optimization(self, initial_guesses: List[np.ndarray]) -> np.ndarray:
            """Perform multi-stage optimization with increasing strictness."""
            best_points = None
            best_ratio = -np.inf
            
            # Stage 1: Coarse search with fast local optimization
            for i, initial in enumerate(initial_guesses):
                try:
                    # Initial rough optimization
                    x0 = initial.flatten()
                    bounds = [(0.001, 0.999) for _ in range(32)]
                    
                    # Use Nelder-Mead for initial coarse optimization
                    result = minimize(
                        lambda x: -self._compute_ratio(x.reshape(-1, 2))[0],
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'adaptive': True}
                    )
                    
                    if result.success:
                        optimized = result.x.reshape(-1, 2)
                        ratio, _, _ = self._compute_ratio(optimized)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized.copy()
                            
                except Exception as e:
                    continue
                    
            # Stage 2: Refinement with more precise local search
            if best_points is not None:
                refined = self._local_search_with_adaptive_perturbation(best_points, iterations=100, temperature=1.0)
                ratio, _, _ = self._compute_ratio(refined)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined.copy()
                    
            # Stage 3: Final aggressive refinement
            if best_points is not None:
                final = self._local_search_with_adaptive_perturbation(best_points, iterations=150, temperature=0.5)
                ratio, _, _ = self._compute_ratio(final)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final.copy()
                    
            return best_points
            
        def optimize(self) -> np.ndarray:
            """Main optimization routine."""
            # Generate diverse initial configurations based on Voronoi insights
            initial_configs = []
            
            # Configuration 1: Voronoi-inspired grid
            config1 = self._generate_initial_voronoi_distribution()
            initial_configs.append(config1)
            
            # Configuration 2: Regular grid with perturbation
            grid_points = []
            for i in range(4):
                for j in range(4):
                    x = (i + 0.5) / 4.0
                    y = (j + 0.5) / 4.0
                    grid_points.append([x, y])
            config2 = np.array(grid_points[:self.num_points])
            config2 += np.random.normal(0, 0.02, config2.shape)
            config2 = np.clip(config2, 0.001, 0.999)
            initial_configs.append(config2)
            
            # Configuration 3: Fibonacci-like spiral
            config3 = []
            phi = (1 + np.sqrt(5)) / 2
            for i in range(self.num_points):
                theta = np.arccos(-1 + (2 * i) / (self.num_points - 1))
                phi_angle = (i * 2 * np.pi) / (phi * phi)
                
                x = np.sin(theta) * np.cos(phi_angle)
                y = np.sin(theta) * np.sin(phi_angle)
                
                x = 0.05 + 0.9 * (x + 1) / 2
                y = 0.05 + 0.9 * (y + 1) / 2
                
                config3.append([x, y])
            initial_configs.append(np.array(config3))
            
            # Configuration 4: Polar arrangement 
            config4 = []
            radii = [0.1, 0.25, 0.4, 0.55]
            angles_per_ring = [4, 6, 8, 10]
            
            config4.append([0.5, 0.5])  # Center point
            
            for i, (radius, num_angles) in enumerate(zip(radii, angles_per_ring)):
                if len(config4) >= self.num_points:
                    break
                for j in range(num_angles):
                    if len(config4) >= self.num_points:
                        break
                    angle = (j * 2 * np.pi) / num_angles
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                    config4.append([x, y])
                    
            # Fill remaining points randomly
            remaining = self.num_points - len(config4)
            for _ in range(remaining):
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                config4.append([x, y])
                
            initial_configs.append(np.array(config4[:self.num_points]))
            
            # Execute multi-stage optimization
            best_solution = self._multi_stage_optimization(initial_configs)
            
            # Final validation and improvement
            if best_solution is not None:
                try:
                    # Final check with full optimization
                    x0 = best_solution.flatten()
                    bounds = [(0.001, 0.999) for _ in range(32)]
                    
                    result = minimize(
                        lambda x: -self._compute_ratio(x.reshape(-1, 2))[0],
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        ratio, _, _ = self._compute_ratio(final_points)
                        
                        if ratio > self.best_ratio:
                            return final_points
                except:
                    pass
                    
            return best_solution if best_solution is not None else initial_configs[0]
    
    # Main optimization process
    try:
        optimizer = VoronoiPointOptimizer(16)
        result = optimizer.optimize()
        return result
    except Exception as e:
        # Fallback to simple grid if anything fails
        np.random.seed(42)
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                grid_points.append([x, y])
        return np.array(grid_points[:16])

# EVOLVE-BLOCK-END