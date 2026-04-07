# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
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

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses advanced initialization and multi-stage optimization approach.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize components
    pipeline = OptimizerPipeline()
    multi_start_optimizer = MultiStartOptimizer(pipeline)
    
    # Run optimization
    best_points, _ = multi_start_optimizer.run(n_points=14, max_restarts=25, time_limit=350.0)
    
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