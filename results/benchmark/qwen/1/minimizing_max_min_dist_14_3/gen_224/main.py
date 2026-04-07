# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
import time
from typing import Tuple, List, Optional
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def fibonacci_spiral_on_sphere(n: int) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(max(0, 1 - y * y))  # radius at y
            theta = golden_angle * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    def sobol_initialization(n: int, seed: int = 42) -> np.ndarray:
        """Generate points using 3D Sobol sequence for better space-filling properties"""
        try:
            from sobol_seq import i4_sobol_generate
            # Generate Sobol points in [0,1]^3
            sobol_points = i4_sobol_generate(3, n)
            
            # Map to sphere using spherical coordinates
            points = np.zeros((n, 3))
            for i in range(n):
                # Map to sphere using similar approach as Fibonacci but with Sobol
                u = sobol_points[i, 0]  # Uniform random in [0,1]
                v = sobol_points[i, 1]  # Uniform random in [0,1]
                
                # Use these as parameters for spherical coordinates
                theta = 2 * np.pi * u  # azimuthal angle
                phi = np.arccos(2 * v - 1)  # polar angle
                
                # Convert to Cartesian
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                
                points[i] = [x, y, z]
            
            return points
        except ImportError:
            # Fallback to random initialization if qmc not available
            np.random.seed(seed)
            points = np.random.uniform(-1, 1, (n, 3))
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0, 1, norms)
            return points / safe_norms
    
    def icosahedron_points(n: int = 14) -> np.ndarray:
        """Generate points using icosahedron-based construction"""
        # Vertices of a regular icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])

        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # If we need more than 12 points, distribute additional points
        if n <= 12:
            # Just return subset of vertices
            return vertices[:n]
        else:
            # For 14 points, we'll start with icosahedron vertices and add two more
            points = vertices.copy()

            # Add two more points that are well-distributed
            # Add points along major axes
            points = np.vstack([points, [[0, 0, 1], [0, 0, -1]]])

            # Apply slight random perturbation to ensure good distribution
            np.random.seed(42)
            points += np.random.normal(0, 0.05, (points.shape[0], 3))

            # Normalize again to maintain unit sphere
            norms = np.linalg.norm(points, axis=1)
            points = points / np.maximum(norms[:, np.newaxis], 1e-12)

            return points[:n]
    
    def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        return points / safe_norms
    
    def calculate_min_max_ratio(points: np.ndarray) -> float:
        """Calculate the minimum-to-maximum distance ratio"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist < 1e-12:
            return 0.0
        return min_dist / max_dist
    
    def energy_based_objective(points_flat: np.ndarray) -> float:
        """Energy-based objective function that encourages uniform distribution"""
        points = points_flat.reshape(-1, 3)
        points = normalize_to_unit_sphere(points)
        
        # Standard distance ratio
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            ratio = 0.0
        else:
            ratio = min_dist / max_dist
            
        # Energy component: add penalty for small distances (repulsion energy)
        energy_penalty = 0.0
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = distances[i*len(points) + j - i*(i+1)//2]  # Correct indexing for pdist
                if dist > 0:
                    # Repulsive energy inversely proportional to distance squared
                    energy_penalty += 1.0 / (dist * dist + 1e-10)
        
        # Add regularization term to prevent extreme variations
        if len(distances) > 0:
            dist_std = np.std(distances)
            dist_mean = np.mean(distances)
            if dist_mean > 1e-12:
                uniformity_penalty = dist_std / dist_mean
                return -(ratio - 0.005 * uniformity_penalty - 0.001 * energy_penalty)
        
        return -ratio
    
    def adaptive_constraint_adjustment(iteration: int, max_iterations: int) -> float:
        """Dynamically adjust optimization constraints based on iteration progress"""
        # Early iterations: focus on spreading out points
        if iteration < max_iterations * 0.3:
            return 0.5
        # Mid iterations: balance spreading and uniformity
        elif iteration < max_iterations * 0.7:
            return 0.8
        # Late iterations: prioritize uniformity
        else:
            return 1.0
    
    def simulate_energy_minimization(initial_points: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Simulate energy minimization to distribute points uniformly"""
        points = initial_points.copy()
        best_points = points.copy()
        best_ratio = calculate_min_max_ratio(points)
        
        # Adaptive learning rate that decreases over time
        learning_rates = np.linspace(0.1, 0.01, max_iterations)
        
        for i in range(max_iterations):
            # Compute all pairwise distances efficiently using cdist
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            
            # Calculate energy-based forces
            gradients = np.zeros_like(points)
            for j in range(len(points)):
                for k in range(len(points)):
                    if j != k:
                        diff = points[j] - points[k]
                        dist = np.linalg.norm(diff)
                        
                        if dist > 1e-10:
                            # Force magnitude inversely proportional to distance cubed (for repulsion)
                            force_magnitude = 1.0 / (dist * dist * dist + 1e-10)
                            # Unit vector from k to j  
                            diff_unit = diff / dist
                            
                            # Add force to point j from point k
                            gradients[j] += force_magnitude * diff_unit
            
            # Project gradients onto tangent plane of sphere
            for j in range(len(points)):
                gradients[j] = gradients[j] - np.dot(gradients[j], points[j]) * points[j]
            
            # Apply constraint adjustment
            constraint_adjustment = adaptive_constraint_adjustment(i, max_iterations)
            
            # Update points with learning rate and constraint adjustment
            update = learning_rates[i] * gradients * constraint_adjustment
            
            # Apply update
            new_points = points - update
            
            # Project back to sphere
            new_points = normalize_to_unit_sphere(new_points)
            
            # Check if this improves the solution
            current_ratio = calculate_min_max_ratio(new_points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = new_points.copy()
            
            points = new_points
            
            # Early stopping if improvement is minimal
            if i > 10 and abs(current_ratio - best_ratio) < 1e-8:
                break
                
        return best_points
    
    def generate_initial_strategies(n: int = 14) -> List[Tuple[str, np.ndarray]]:
        """Generate multiple initial strategies for better diversity"""
        strategies = []
        
        # Original strategies
        strategies.append(("fibonacci", fibonacci_spiral_on_sphere(n)))
        
        # Add Sobol-based initialization
        strategies.append(("sobol", sobol_initialization(n, 42)))
        
        # Add icosahedron-based initialization
        strategies.append(("icosahedron", icosahedron_points(n)))
        
        # Add random initialization
        np.random.seed(42)
        strategies.append(("random", np.random.randn(n, 3)))
        
        # Add perturbed Fibonacci variants for diversity
        fib_points = fibonacci_spiral_on_sphere(n)
        strategies.append(("fibonacci_perturbed", fib_points + np.random.normal(0, 0.01, (n, 3))))
        
        # Add more variants with different perturbations
        strategies.append(("fibonacci_perturbed_2", fib_points + np.random.normal(0, 0.02, (n, 3))))
        
        # Add Sobol variants
        strategies.append(("sobol_perturbed", sobol_initialization(n, 123) + np.random.normal(0, 0.01, (n, 3))))
        
        # Add icosahedron variants
        ico_points = icosahedron_points(n)
        strategies.append(("icosahedron_perturbed", ico_points + np.random.normal(0, 0.01, (n, 3))))
        
        # Add randomized uniform distribution
        np.random.seed(123)
        strategies.append(("randomized_uniform", np.random.rand(n, 3) * 2 - 1))
        
        return strategies
    
    def optimize_with_adaptive_strategy(initial_points: np.ndarray, max_iterations: int = 500) -> np.ndarray:
        """Optimize using adaptive multi-stage approach with better convergence"""
        points = initial_points.copy()
        
        # First phase: Energy-based simulation for coarse distribution
        try:
            coarse_points = simulate_energy_minimization(points, max_iterations=50)
            points = coarse_points
        except Exception as e:
            warnings.warn(f"Energy minimization failed: {e}")
        
        # Second phase: Fine optimization with constrained L-BFGS-B
        try:
            coarse_flat = points.flatten()
            bounds = [(-1, 1) for _ in range(42)]
            
            # Use the energy-based objective for final refinement
            result = minimize(
                lambda x: energy_based_objective(x),
                coarse_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            # Extract refined points
            refined_points = result.x.reshape(-1, 3)
            
            # Final normalization
            final_points = normalize_to_unit_sphere(refined_points)
            return final_points
            
        except Exception as e:
            warnings.warn(f"L-BFGS-B optimization failed: {e}")
            return normalize_to_unit_sphere(points)
    
    # Multi-start optimization with different initializations
    best_ratio = -np.inf
    best_points = None
    
    # Generate multiple initial strategies
    initial_strategies = generate_initial_strategies(14)
    
    # Run optimization from each initialization
    for init_name, initial_points in initial_strategies:
        try:
            # Add slight random perturbation to break symmetry
            np.random.seed(42)
            noisy_points = initial_points + np.random.normal(0, 0.01, (14, 3))
            
            # Ensure all points are on unit sphere
            normalized_points = normalize_to_unit_sphere(noisy_points)
            
            # Optimize with adaptive strategy
            optimized_points = optimize_with_adaptive_strategy(normalized_points, max_iterations=500)
            
            # Evaluate final result
            final_ratio = calculate_min_max_ratio(optimized_points)
            
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            # Continue with other initializations if one fails
            warnings.warn(f"Optimization with {init_name} failed: {e}")
            continue
    
    # Fallback if nothing worked
    if best_points is None:
        # Use simple Fibonacci with L-BFGS refinement
        points = fibonacci_spiral_on_sphere(14)
        points_flat = points.flatten()
        bounds = [(-1, 1) for _ in range(42)]
        
        result = minimize(
            lambda x: energy_based_objective(x),
            points_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        refined_points = result.x.reshape(-1, 3)
        best_points = normalize_to_unit_sphere(refined_points)
    
    return best_points

# EVOLVE-BLOCK-END