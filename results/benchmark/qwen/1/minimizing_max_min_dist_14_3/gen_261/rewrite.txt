# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
import time
from typing import Tuple, List, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

class PointInitializer:
    """Handles all point initialization strategies"""
    
    @staticmethod
    def sobol_points_sphere(n_points: int, seed: int = 42) -> np.ndarray:
        """Generate points on sphere using 3D Sobol sequence"""
        try:
            from sobol_seq import i4_sobol_generate
            np.random.seed(seed)
            sobol_points = i4_sobol_generate(3, n_points)
            
            points = np.zeros((n_points, 3))
            for i in range(n_points):
                u = sobol_points[i, 0]
                v = sobol_points[i, 1]
                theta = 2 * np.pi * u
                phi = np.arccos(2 * v - 1)
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                points[i] = [x, y, z]
            return points
        except ImportError:
            return PointInitializer.fibonacci_spiral_sphere(n_points, seed)
    
    @staticmethod
    def fibonacci_spiral_sphere(n_points: int, seed: int = 42) -> np.ndarray:
        """Generate points on a sphere using Fibonacci spiral method"""
        np.random.seed(seed)
        points = []
        phi = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    @staticmethod
    def icosahedron_points(n: int = 14, seed: int = 42) -> np.ndarray:
        """Generate points using icosahedron-based construction"""
        np.random.seed(seed)
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        if n <= 12:
            return vertices[:n]
        else:
            points = vertices.copy()
            points = np.vstack([points, [[0, 0, 1], [0, 0, -1]]])
            points += np.random.normal(0, 0.05, (points.shape[0], 3))
            norms = np.linalg.norm(points, axis=1)
            points = points / np.maximum(norms[:, np.newaxis], 1e-12)
            return points[:n]
    
    @staticmethod
    def spherical_voronoi_initialization(n_points: int, seed: int = 42) -> np.ndarray:
        """Improved initialization using Sobol sequence"""
        sobol_points = PointInitializer.sobol_points_sphere(n_points, seed)
        refined_points = sobol_points.copy()
        np.random.seed(seed)
        noise = np.random.normal(0, 0.02, (n_points, 3))
        refined_points += noise
        norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
        refined_points = refined_points / np.maximum(norms, 1e-12)
        return refined_points

class OptimizationStage:
    """Encapsulates a single optimization stage with its configuration"""
    
    def __init__(self, method: str, maxiter: int, ftol: float, gtol: float, constraints_required: bool = True):
        self.method = method
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol
        self.constraints_required = constraints_required
    
    def execute(self, objective_func: Callable, x0: np.ndarray, bounds: List[Tuple[float, float]], 
                constraints: Optional[dict] = None) -> Tuple[np.ndarray, bool]:
        """Execute this optimization stage"""
        try:
            options = {'maxiter': self.maxiter, 'ftol': self.ftol, 'gtol': self.gtol}
            
            if self.constraints_required and constraints is not None:
                result = minimize(
                    objective_func,
                    x0,
                    method=self.method,
                    bounds=bounds,
                    constraints=constraints,
                    options=options
                )
            else:
                result = minimize(
                    objective_func,
                    x0,
                    method=self.method,
                    bounds=bounds,
                    options=options
                )
            
            return result.x, result.success
        except Exception:
            return x0, False

class OptimizerPipeline:
    """Manages the complete optimization pipeline with multiple stages"""
    
    def __init__(self):
        # Define optimization stages with increasing precision
        self.stages = [
            OptimizationStage("SLSQP", 200, 1e-8, 1e-8, True),
            OptimizationStage("L-BFGS-B", 200, 1e-12, 1e-12, False),
            OptimizationStage("trust-constr", 400, 1e-14, 1e-14, False)
        ]
    
    def optimize(self, initial_points: np.ndarray, max_iter: int = 800) -> np.ndarray:
        """Execute the complete optimization pipeline"""
        n, d = initial_points.shape
        
        def objective(x_flat):
            points = x_flat.reshape(n, d)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            normalized_points = points / np.maximum(norms, 1e-12)
            
            # Calculate ratio with penalty
            distances = pdist(normalized_points)
            if len(distances) == 0:
                return 1e10
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist < 1e-12:
                return 1e10
            
            ratio = min_dist / max_dist
            
            # Apply penalty for extreme ratios
            if ratio < 0.1:
                ratio -= 0.1 * (0.1 - ratio)
            
            return -ratio if ratio > 0 else 1e10
        
        def constraint_sphere(x_flat):
            points = x_flat.reshape(n, d)
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0
        
        constraints = {'type': 'eq', 'fun': constraint_sphere}
        bounds = [(-2, 2) for _ in range(n * d)]
        
        current_points = initial_points.copy()
        
        for stage in self.stages:
            x0 = current_points.flatten()
            x_opt, success = stage.execute(objective, x0, bounds, constraints if stage.constraints_required else None)
            
            if success:
                current_points = x_opt.reshape(n, d)
                norms = np.linalg.norm(current_points, axis=1, keepdims=True)
                current_points = current_points / np.maximum(norms, 1e-12)
            else:
                # Continue with current points if stage fails
                pass
        
        return current_points

class MultiStartOptimizer:
    """Handles multi-start optimization with diverse strategies"""
    
    def __init__(self, pipeline: OptimizerPipeline):
        self.pipeline = pipeline
        self.init_strategies = [
            ("sobol", PointInitializer.sobol_points_sphere),
            ("icosahedron", PointInitializer.icosahedron_points),
            ("fibonacci", PointInitializer.fibonacci_spiral_sphere),
            ("spherical_voronoi", PointInitializer.spherical_voronoi_initialization)
        ]
    
    def run(self, n_points: int = 14, max_restarts: int = 25, time_limit: float = 350.0) -> Tuple[np.ndarray, float]:
        """Run multi-start optimization with time management"""
        start_time = time.time()
        best_ratio = -np.inf
        best_points = None
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            strategy_idx = restart % len(self.init_strategies)
            strategy_name, init_func = self.init_strategies[strategy_idx]
            seed = restart * 100 + 42
            
            try:
                initial_points = init_func(n_points, seed)
                
                # Add perturbation
                np.random.seed(seed)
                noise = np.random.normal(0, 0.02, (n_points, 3))
                initial_points += noise
                
                # Project to sphere
                norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
                initial_points = initial_points / np.maximum(norms, 1e-12)
                
                # Optimize
                optimized_points = self.pipeline.optimize(initial_points, max_iter=800)
                
                # Evaluate
                distances = pdist(optimized_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 1e-12:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
            except Exception:
                continue
        
        # Additional fallback strategy
        if best_points is None or best_ratio < 0.4:
            if time.time() - start_time < time_limit:
                try:
                    np.random.seed(42)
                    random_points = np.random.rand(n_points, 3) * 2 - 1
                    norms = np.linalg.norm(random_points, axis=1, keepdims=True)
                    normalized_points = random_points / np.maximum(norms, 1e-12)
                    optimized_points = self.pipeline.optimize(normalized_points, max_iter=600)
                    
                    distances = pdist(optimized_points)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        if max_dist > 1e-12:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
                except Exception:
                    pass
        
        # Final safeguard
        if best_points is None:
            np.random.seed(42)
            points = np.random.rand(n_points, 3) * 2 - 1
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            best_points = points / np.maximum(norms, 1e-12)
        
        return best_points, best_ratio

def spherical_code_14():
    """Generate initial configuration using known spherical code for 14 points"""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    
    for i in range(14):
        y = 1 - (i / float(13)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i  # golden angle increment
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points.append([x, y, z])
    
    return np.array(points)

def normalize_to_unit_sphere(points):
    """Normalize points to lie exactly on unit sphere"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def voronoi_uniformity_score(points):
    """Calculate how uniform the Voronoi cells are on the sphere"""
    try:
        sv = SphericalVoronoi(points)
        areas = sv.voronoi_cell_areas()
        # Return variance of areas - lower is better (more uniform)
        return np.var(areas)
    except:
        # Fallback to inverse of min distance spread
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return max_dist / (min_dist + 1e-12) if min_dist > 1e-12 else 1000

def geometric_objective(points):
    """Objective function that combines distance ratios with geometric uniformity"""
    # Compute pairwise distances
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    
    # Get min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    # Avoid division by zero
    if d_max < 1e-10:
        return -1e10
    
    # Ratio of min to max distance (what we want to maximize)
    ratio = d_min / d_max
    
    # Add penalty for non-uniform Voronoi cells (higher variance means less uniform)
    uniformity_penalty = 1.0 / (1.0 + voronoi_uniformity_score(points))
    
    # Combined objective (maximize ratio AND uniformity)
    return ratio * (1.0 + 0.1 * uniformity_penalty)

def mutate_points(points, strength=0.05):
    """Apply mutation to points with adaptive scaling"""
    # Create different mutation types
    mutated = points.copy()
    
    # Random perturbation with adaptive strength
    noise = np.random.normal(0, strength, points.shape)
    mutated += noise
    
    # Normalize back to sphere
    mutated = normalize_to_unit_sphere(mutated)
    
    return mutated

def adaptive_cooling_schedule(iteration, max_iterations):
    """Adaptive cooling schedule for mutation strength"""
    # Start with higher mutation strength, cool down gradually
    base_strength = 0.1
    min_strength = 0.001
    return max(min_strength, base_strength * (1 - iteration / max_iterations))

def evolutionary_refinement(initial_points, max_iterations=500):
    """Main evolutionary refinement process"""
    current_points = initial_points.copy()
    best_points = current_points.copy()
    best_score = geometric_objective(current_points)
    
    # Evolutionary parameters
    population_size = 10
    elite_count = 2
    
    for iteration in range(max_iterations):
        # Adaptive cooling
        mutation_strength = adaptive_cooling_schedule(iteration, max_iterations)
        
        # Generate new population via mutations
        population = [current_points]
        
        # Add mutated versions
        for _ in range(population_size - 1):
            mutated = mutate_points(current_points, mutation_strength)
            population.append(mutated)
        
        # Evaluate population
        scores = [geometric_objective(p) for p in population]
        
        # Find best in population
        best_in_pop_idx = np.argmax(scores)
        best_in_pop = population[best_in_pop_idx]
        best_in_pop_score = scores[best_in_pop_idx]
        
        # Update global best
        if best_in_pop_score > best_score:
            best_score = best_in_pop_score
            best_points = best_in_pop.copy()
        
        # Selection: keep top performers
        sorted_indices = np.argsort(scores)[::-1]
        selected = [population[i] for i in sorted_indices[:elite_count]]
        
        # Continue evolution with selected
        current_points = selected[0]  # Keep the best
        
        # Occasionally add some diversity through random restarts
        if iteration % 50 == 0 and iteration > 0:
            current_points = spherical_code_14()
            current_points = normalize_to_unit_sphere(current_points)
    
    return best_points

def geometric_local_refinement(points, iterations=100):
    """Local refinement using gradient-based approach on sphere"""
    # Convert to flattened representation for optimization
    flattened = points.flatten()
    
    def objective_flat(flat_points):
        points_matrix = flat_points.reshape(-1, 3)
        return -geometric_objective(points_matrix)
    
    # Use differential evolution with spherical constraints
    bounds = [(-1.0, 1.0)] * len(flattened)
    
    try:
        from scipy.optimize import differential_evolution
        result = differential_evolution(
            objective_flat,
            bounds,
            seed=42,
            maxiter=iterations,
            popsize=8,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=False
        )
        
        refined_points = result.x.reshape(-1, 3)
        # Ensure points remain on sphere
        refined_points = normalize_to_unit_sphere(refined_points)
        return refined_points
        
    except:
        # Fallback to simple iterative refinement
        return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses hybrid approach combining multi-start optimization with evolutionary refinement techniques.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize components
    pipeline = OptimizerPipeline()
    multi_start_optimizer = MultiStartOptimizer(pipeline)
    
    # Run classical multi-start optimization
    best_points, _ = multi_start_optimizer.run(n_points=14, max_restarts=25, time_limit=350.0)
    
    # Apply evolutionary refinement to the best solution for additional improvement
    start_time = time.time()
    if time.time() - start_time < 300:
        # Try evolutionary refinement if time allows
        try:
            # Normalize to unit sphere first
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            normalized = best_points / np.maximum(norms, 1e-12)
            
            # Apply evolutionary refinement
            evolved_points = evolutionary_refinement(normalized, max_iterations=200)
            
            # Local refinement
            refined_points = geometric_local_refinement(evolved_points, iterations=50)
            
            # Ensure final normalization
            final_norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
            final_points = refined_points / np.maximum(final_norms, 1e-12)
            
            # Final validation
            distances = cdist(final_points, final_points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist >= 1e-10 and min_dist >= 1e-10:
                # Convert to unit cube [0,1]^3
                centered = final_points - np.mean(final_points, axis=0)
                max_coord = np.max(np.abs(centered))
                if max_coord > 0:
                    scaled = centered / max_coord * 0.5
                else:
                    scaled = centered
                final_points = scaled + 0.5
                
                return final_points
        except:
            pass
    
    # Fall back to standard conversion if evolutionary failed
    # Convert to unit cube [0,1]^3
    centered = best_points - np.mean(best_points, axis=0)
    max_coord = np.max(np.abs(centered))
    if max_coord > 0:
        scaled = centered / max_coord * 0.5
    else:
        scaled = centered
    final_points = scaled + 0.5
    
    return final_points

# EVOLVE-BLOCK-END