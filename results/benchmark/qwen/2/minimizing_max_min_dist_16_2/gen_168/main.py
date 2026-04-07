# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
from typing import Tuple, List, Optional

class GeometricForceBalancingOptimizer:
    """Geometric force balancing optimizer for point dispersion problems."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        self.max_iterations = 300
        
    def calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min/max distance ratio with proper error handling."""
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
    
    def compute_forces(self, points: np.ndarray, k_attraction: float = 1.0, 
                      k_repulsion: float = 10.0, min_distance: float = 0.01) -> np.ndarray:
        """
        Compute net forces on each point using attraction and repulsion mechanisms.
        """
        n = points.shape[0]
        forces = np.zeros_like(points)
        
        # Repulsion forces (Coulomb-like)
        for i in range(n):
            for j in range(i+1, n):
                diff = points[i] - points[j]
                distance = np.linalg.norm(diff)
                
                if distance > min_distance:
                    # Repulsive force (inverse square law)
                    force_magnitude = k_repulsion / (distance * distance)
                    force_vector = force_magnitude * diff / distance
                    forces[i] += force_vector
                    forces[j] -= force_vector
        
        # Attraction to center (to prevent points from going to extremes)
        center = np.mean(points, axis=0)
        for i in range(n):
            diff = center - points[i]
            distance = np.linalg.norm(diff)
            if distance > 0.001:
                # Attractive force to center
                force_magnitude = k_attraction * distance
                force_vector = force_magnitude * diff / distance
                forces[i] -= force_vector
                
        return forces
    
    def force_relaxation(self, points: np.ndarray, iterations: int = 100, 
                        initial_step_size: float = 0.01) -> np.ndarray:
        """
        Perform force relaxation to distribute points evenly.
        """
        current_points = points.copy()
        step_size = initial_step_size
        
        for iteration in range(iterations):
            # Dynamic step size adjustment
            if iteration > 0 and iteration % 20 == 0:
                step_size *= 0.8
            
            # Compute forces
            forces = self.compute_forces(current_points)
            
            # Apply forces with step size
            current_points += forces * step_size
            
            # Enforce bounds
            current_points = np.clip(current_points, 0.001, 0.999)
            
        return current_points
    
    def generate_hexagonal_lattice(self) -> np.ndarray:
        """Generate high-quality hexagonal lattice with optimized spacing."""
        points = []
        rows = cols = 4
        
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Enhanced spacing for better distribution
        spacing_x *= 0.88
        spacing_y *= 0.88
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Maintain bounds precisely
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                
                points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def generate_fibonacci_distribution(self) -> np.ndarray:
        """Generate points using Fibonacci spiral with improved distribution."""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        for i in range(self.num_points):
            # Better spiral parameterization
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (phi * phi)
            
            # Cartesian conversion with better mapping
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            # Map to [0.05, 0.95] range with boundary safety
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    def generate_regular_grid(self) -> np.ndarray:
        """Generate regular grid with proper boundary handling."""
        points = []
        side_length = int(math.ceil(math.sqrt(self.num_points)))
        
        for i in range(side_length):
            for j in range(side_length):
                if len(points) >= self.num_points:
                    break
                x = (i + 0.5) / side_length
                y = (j + 0.5) / side_length
                points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def generate_polar_arrangement(self) -> np.ndarray:
        """Generate polar arrangement with concentric rings."""
        points = []
        # Concentric circles with increasing angular density
        radii = [0.15, 0.3, 0.45, 0.6]
        angles_per_ring = [4, 6, 8, 10]
        
        # Center point
        points.append([0.5, 0.5])
        
        # Ring points
        for i, (radius, num_angles) in enumerate(zip(radii, angles_per_ring)):
            for j in range(num_angles):
                if len(points) >= self.num_points:
                    break
                angle = (j * 2 * math.pi) / num_angles
                x = 0.5 + radius * math.cos(angle)
                y = 0.5 + radius * math.sin(angle)
                points.append([x, y])
            if len(points) >= self.num_points:
                break
        
        # Fill remaining spots with random distribution
        remaining = self.num_points - len(points)
        for _ in range(remaining):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            points.append([x, y])
        
        return np.array(points)
    
    def generate_diverse_initial_configs(self) -> List[np.ndarray]:
        """Generate highly diverse set of initial configurations."""
        configs = []
        
        # Base configurations
        configs.append(self.generate_hexagonal_lattice())
        configs.append(self.generate_fibonacci_distribution())
        configs.append(self.generate_regular_grid())
        configs.append(self.generate_polar_arrangement())
        
        # Enhanced variations with multiple perturbation levels
        np.random.seed(42)
        for base_config in configs[:3]:  # Only first 3 base configs for variation
            # Different perturbation magnitudes
            for mag in [0.01, 0.015, 0.02]:
                perturbed = base_config + np.random.normal(0, mag, base_config.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                configs.append(perturbed)
        
        # Add a few more structured variations
        for _ in range(4):
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
    
    def hybrid_optimization(self, points: np.ndarray) -> np.ndarray:
        """
        Hybrid optimization combining force relaxation with gradient-based refinement.
        """
        current_points = points.copy()
        
        # Phase 1: Force relaxation for global distribution
        relaxed_points = self.force_relaxation(current_points, iterations=150, initial_step_size=0.01)
        
        # Phase 2: Gradient-based refinement
        def objective(x):
            points_temp = x.reshape(-1, self.dimension)
            distances = pdist(points_temp)
            if len(distances) == 0:
                return 0.0
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 0.0
            return -min_dist / max_dist
            
        try:
            result = minimize(
                objective,
                relaxed_points.flatten(),
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            if result.success:
                refined_points = result.x.reshape(-1, self.dimension)
                # Ensure bounds
                refined_points = np.clip(refined_points, 0.001, 0.999)
                return refined_points
        except Exception:
            pass
        
        return relaxed_points
    
    def optimize_with_force_balancing(self, configs: List[np.ndarray]) -> np.ndarray:
        """
        Main optimization loop using force balancing approach.
        """
        best_ratio = -np.inf
        best_points = None
        
        # Try each configuration through force balancing
        for i, config in enumerate(configs):
            # Apply force relaxation
            relaxed = self.force_relaxation(config, iterations=100, initial_step_size=0.01)
            
            # Hybrid optimization
            optimized = self.hybrid_optimization(relaxed)
            
            # Evaluate
            ratio, _, _ = self.calculate_ratio(optimized)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized.copy()
        
        # Additional refinement using the best configuration
        if best_points is not None:
            # Do a few more rounds of force relaxation with smaller steps
            final_points = self.force_relaxation(best_points, iterations=50, initial_step_size=0.005)
            
            # Final gradient-based optimization
            final_points = self.hybrid_optimization(final_points)
            
            ratio, _, _ = self.calculate_ratio(final_points)
            if ratio > best_ratio:
                best_points = final_points
        
        return best_points if best_points is not None else configs[0]

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize optimizer
    optimizer = GeometricForceBalancingOptimizer(16, 2)
    
    # Generate diverse initial configurations
    initial_configs = optimizer.generate_diverse_initial_configs()
    
    # Optimize using force balancing approach
    best_points = optimizer.optimize_with_force_balancing(initial_configs)
    
    return best_points

# EVOLVE-BLOCK-END