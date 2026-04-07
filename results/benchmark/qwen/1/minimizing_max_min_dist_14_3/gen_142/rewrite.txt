# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def fibonacci_spiral_on_sphere(n):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = golden_angle * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    def sobol_initialization(n: int, seed: int = 42) -> np.ndarray:
        """Generate points using Sobol sequence for better space-filling properties"""
        try:
            from scipy.stats import qmc
            # Create Sobol sequence sampler
            sampler = qmc.Sobol(d=3, seed=seed)
            # Generate points
            points = sampler.random(n)
            # Scale to [-1, 1]^3
            points = points * 2 - 1
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0, 1, norms)
            return points / safe_norms
        except ImportError:
            # Fallback to random initialization if qmc not available
            points = np.random.uniform(-1, 1, (n, 3))
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0, 1, norms)
            return points / safe_norms
    
    def normalize_to_unit_sphere(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        return points / safe_norms
    
    def calculate_min_max_ratio(points):
        """Calculate the minimum-to-maximum distance ratio"""
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist
    
    def spherical_voronoi_quality(points):
        """Evaluate quality based on spherical Voronoi diagram properties"""
        # Normalize points
        points = normalize_to_unit_sphere(points)
        
        try:
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points, radius=1.0, center=np.zeros(3))
            
            # Calculate Voronoi cell areas
            cell_areas = sv.voronoi_regions_area()
            
            # Return variance of cell areas (lower variance = more uniform distribution)
            return np.var(cell_areas)
        except:
            # Fallback if Voronoi computation fails
            return np.inf
    
    def energy_based_objective(points_flat):
        """Energy-based objective function that encourages uniform distribution"""
        points = points_flat.reshape(-1, 3)
        points = normalize_to_unit_sphere(points)
        
        # Standard distance ratio
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
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
                dist = distances[i,j]
                if dist > 0:
                    # Repulsive energy inversely proportional to distance squared
                    energy_penalty += 1.0 / (dist * dist + 1e-10)
        
        # Add regularization term based on Voronoi quality
        voronoi_penalty = spherical_voronoi_quality(points)
        
        # Combine objective: maximize ratio with energy and uniformity considerations
        # Negative because we minimize in optimization
        return -(ratio - 0.005 * voronoi_penalty - 0.001 * energy_penalty)
    
    def energy_gradient(points_flat):
        """Compute analytical gradient for energy-based optimization"""
        points = points_flat.reshape(-1, 3)
        points = normalize_to_unit_sphere(points)
        
        gradients = np.zeros_like(points)
        
        # Compute all pairwise distances
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Calculate energy-based forces
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    diff = points[i] - points[j]
                    dist = np.linalg.norm(diff)
                    
                    if dist > 1e-10:
                        # Force magnitude inversely proportional to distance cubed (for repulsion)
                        force_magnitude = 1.0 / (dist * dist * dist + 1e-10)
                        # Unit vector from j to i  
                        diff_unit = diff / dist
                        
                        # Add force to point i from point j
                        gradients[i] += force_magnitude * diff_unit
                        
                        # Add equal and opposite force to point j from point i
                        gradients[j] -= force_magnitude * diff_unit
        
        # Project gradients onto tangent plane of sphere
        for i in range(len(points)):
            gradients[i] = gradients[i] - np.dot(gradients[i], points[i]) * points[i]
            
        return gradients.flatten()
    
    def adaptive_constraint_adjustment(points, iteration):
        """Dynamically adjust optimization constraints based on distribution quality"""
        # Measure current uniformity via Voronoi variance
        voronoi_var = spherical_voronoi_quality(points)
        
        # Adjust constraint strength based on iteration and quality
        if iteration < 20:
            # Early iterations: focus on spreading out points
            constraint_strength = 0.5
        elif iteration < 50:
            # Mid iterations: balance spreading and uniformity
            constraint_strength = 0.8
        else:
            # Late iterations: prioritize uniformity
            constraint_strength = 1.0
            
        # Scale based on current quality
        adjusted_strength = constraint_strength * (1.0 + 0.2 * np.exp(-voronoi_var))
        return adjusted_strength
    
    def simulate_energy_minimization(initial_points, max_iterations=100):
        """Simulate energy minimization to distribute points uniformly"""
        points = initial_points.copy()
        
        # Store best solution so far
        best_points = points.copy()
        best_ratio = calculate_min_max_ratio(points)
        
        # Adaptive learning rate that decreases over time
        learning_rates = np.linspace(0.1, 0.01, max_iterations)
        
        for i in range(max_iterations):
            # Get current energy-based objective and gradient
            current_flat = points.flatten()
            
            # Compute gradient manually instead of using scipy
            grad = energy_gradient(current_flat)
            
            # Apply constraint adjustment
            constraint_adjustment = adaptive_constraint_adjustment(points, i)
            
            # Update points with learning rate and constraint adjustment
            update = learning_rates[i] * grad * constraint_adjustment
            
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
    
    # Multi-start optimization with different initializations
    best_ratio = -np.inf
    best_points = None
    
    # Try multiple initialization strategies
    initializations = [
        ("fibonacci", fibonacci_spiral_on_sphere(14)),
        ("sobol", sobol_initialization(14, 42)),
        ("fibonacci_perturbed", fibonacci_spiral_on_sphere(14) + np.random.normal(0, 0.05, (14, 3))),
        ("sobol_perturbed", sobol_initialization(14, 123) + np.random.normal(0, 0.05, (14, 3))),
    ]
    
    # Run optimization from each initialization
    for init_name, initial_points in initializations:
        try:
            # First phase: Energy-based simulation for coarse distribution
            coarse_points = simulate_energy_minimization(initial_points, max_iterations=50)
            
            # Second phase: Fine optimization with constrained L-BFGS
            coarse_flat = coarse_points.flatten()
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
            
            # Evaluate final result
            final_ratio = calculate_min_max_ratio(final_points)
            
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = final_points.copy()
                
        except Exception as e:
            # Continue with other initializations if one fails
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
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        refined_points = result.x.reshape(-1, 3)
        best_points = normalize_to_unit_sphere(refined_points)
    
    return best_points

# EVOLVE-BLOCK-END