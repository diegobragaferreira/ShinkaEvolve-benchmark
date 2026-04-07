# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

class PointDistributionOptimizer:
    """Optimizes 14 points in 3D space to maximize min/max distance ratio."""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        
    def compute_min_max_ratio(self, points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return 0.0

        return d_min / d_max
    
    def fibonacci_sphere(self, n):
        """Generate points on sphere using Fibonacci spiral."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = golden_angle * i  # Golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    def spherical_constraint(self, points):
        """Normalize points to lie on the unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def generate_initialization_strategies(self):
        """Generate multiple diverse initial point sets."""
        strategies = []
        
        # Strategy 1: Fibonacci sphere distribution
        fib_points = self.fibonacci_sphere(14)
        # Add small perturbations
        perturbed = fib_points + np.random.normal(0, 0.03, fib_points.shape)
        strategies.append(self.spherical_constraint(perturbed))

        # Strategy 2: Random points on sphere
        random_points = np.random.randn(14, 3)
        strategies.append(self.spherical_constraint(random_points))

        # Strategy 3: Structured distribution along axes
        struct_points = np.zeros((14, 3))
        for i in range(14):
            if i < 3:
                # Along axes
                struct_points[i] = [1 if j==i else 0 for j in range(3)]
            elif i < 6:
                # Opposite axes
                struct_points[i] = [-1 if j==i-3 else 0 for j in range(3)]
            elif i < 9:
                # Diagonal combinations
                j = i - 6
                struct_points[i] = [1 if k==j else -1 if k==(j+1)%3 else 0 for k in range(3)]
            else:
                # Random points on sphere
                struct_points[i] = np.random.randn(3)
        strategies.append(self.spherical_constraint(struct_points))

        # Strategy 4: Slightly perturbed Fibonacci with larger variance
        fib_perturbed = self.fibonacci_sphere(14) + np.random.normal(0, 0.07, (14, 3))
        strategies.append(self.spherical_constraint(fib_perturbed))

        # Strategy 5: Icosahedron-based initialization
        # Vertices of regular icosahedron scaled to unit sphere
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        icosahedron_vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ])
        
        # Add 2 more points to make 14 total
        extra_points = []
        for i in range(2):
            phi = np.arccos(1 - 2 * ((12 + i) / (14 - 1)))
            theta = np.sqrt(14) * phi
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            extra_points.append([x, y, z])

        icos_points = np.vstack([icosahedron_vertices, extra_points])
        # Apply slight random perturbations
        icos_points += np.random.normal(0, 0.05, icos_points.shape)
        strategies.append(self.spherical_constraint(icos_points))
        
        return strategies
    
    def objective_function(self, points_flat):
        """Objective function to maximize - negative of min/max ratio."""
        # Reshape flat array to 2D points array
        points = points_flat.reshape(-1, 3)

        # Apply spherical constraint to keep points on unit sphere
        points = self.spherical_constraint(points)

        # Compute ratio
        ratio = self.compute_min_max_ratio(points)

        # Return negative because we want to maximize ratio, but optimizers minimize
        return -ratio
    
    def global_optimization(self, initial_points):
        """Perform global optimization using differential evolution."""
        bounds = [(-1, 1)] * (14 * 3)
        
        try:
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=50,
                popsize=25,
                seed=self.seed,
                disp=False,
                polish=True,
                strategy='best1bin'
            )

            if result.success:
                points = result.x.reshape(-1, 3)
                return self.spherical_constraint(points)
        except:
            pass
        return initial_points
    
    def local_refinement(self, points):
        """Perform local refinement with L-BFGS-B."""
        def local_obj(x_flat):
            points = x_flat.reshape(-1, 3)
            points = self.spherical_constraint(points)
            ratio = self.compute_min_max_ratio(points)
            return -ratio  # Negative for minimization

        try:
            x0 = points.flatten()
            bounds = [(-1, 1)] * (14 * 3)
            result = minimize(
                local_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-9, 'disp': False}
            )
            if result.success:
                return self.spherical_constraint(result.x.reshape(-1, 3))
        except:
            pass
        return points
    
    def adaptive_refinement(self, points, max_iterations=50):
        """Apply adaptive refinement to improve solution quality."""
        best_points = points.copy()
        best_ratio = self.compute_min_max_ratio(points)
        
        # Adaptive step size control
        step_size = 0.01
        patience = 0
        max_patience = 10

        for iteration in range(max_iterations):
            current_ratio = self.compute_min_max_ratio(best_points)

            # Check for improvement
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = best_points.copy()
                patience = 0
                step_size = min(0.01, step_size * 1.1)  # Increase step size
            else:
                patience += 1
                if patience > max_patience:
                    step_size = max(0.0001, step_size * 0.8)  # Decrease step size

            # Try small perturbations
            improved = False
            for i in range(14):
                for dim in range(3):
                    # Try perturbing in both directions
                    for direction in [-1, 1]:
                        test_points = best_points.copy()
                        test_points[i, dim] += direction * step_size

                        # Project back to unit sphere
                        norm = np.linalg.norm(test_points[i])
                        if norm > 0:
                            test_points[i] = test_points[i] / norm

                        test_ratio = self.compute_min_max_ratio(test_points)

                        if test_ratio > best_ratio:
                            best_ratio = test_ratio
                            best_points = test_points.copy()
                            improved = True
                            patience = 0
                            break
                if improved:
                    break

            if not improved and patience > max_patience:
                break

        return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    optimizer = PointDistributionOptimizer(seed=42)
    
    # Generate multiple initial sets
    initial_strategies = optimizer.generate_initialization_strategies()
    
    # Try each initialization
    best_solution = None
    best_ratio = 0.0
    
    for i, initial_points in enumerate(initial_strategies):
        # Global optimization
        global_optimized = optimizer.global_optimization(initial_points)
        
        # Local refinement
        locally_refined = optimizer.local_refinement(global_optimized)
        
        # Compute ratio for this attempt
        ratio = optimizer.compute_min_max_ratio(locally_refined)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = locally_refined.copy()
    
    # Additional adaptive refinement
    if best_solution is not None:
        refined_points = optimizer.adaptive_refinement(best_solution, max_iterations=30)
        ratio = optimizer.compute_min_max_ratio(refined_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = refined_points.copy()
    
    # Final high-precision optimization
    if best_solution is not None:
        def final_obj(x_flat):
            points = x_flat.reshape(-1, 3)
            points = optimizer.spherical_constraint(points)
            ratio = optimizer.compute_min_max_ratio(points)
            return -ratio  # Negative because we minimize

        try:
            x0 = best_solution.flatten()
            bounds = [(-1, 1)] * (14 * 3)
            result = minimize(
                final_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            if result.success:
                return optimizer.spherical_constraint(result.x.reshape(-1, 3))
        except:
            pass

    # If nothing worked, return the best initialization
    if best_solution is not None:
        return best_solution

    # Fallback to Fibonacci with small perturbation
    fib_points = optimizer.fibonacci_sphere(14)
    fib_points = fib_points + np.random.normal(0, 0.05, fib_points.shape)
    return optimizer.spherical_constraint(fib_points)

# EVOLVE-BLOCK-END