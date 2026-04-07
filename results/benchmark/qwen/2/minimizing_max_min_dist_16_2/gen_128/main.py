# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi
import math
from typing import Tuple, List, Optional

class GeometricAnnealingEvolver:
    """Novel geometric annealing optimizer that maximizes min/max distance ratio."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        self.max_iterations = 2000
        
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
    
    def generate_voronoi_initial(self) -> np.ndarray:
        """Generate initial configuration using constrained Voronoi-like distribution."""
        # Create points that respect Voronoi constraints for good spread
        points = []
        
        # Generate points in a way that avoids clustering
        # Use a modified Fibonacci approach with constraint sampling
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        # Distribute points to avoid extreme clustering
        for i in range(self.num_points):
            # Modified spiral to avoid regular patterns
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (phi * phi)
            
            # Add geometric variance to avoid regularity
            variance_factor = 0.8 + 0.4 * np.sin(i * 0.7) * np.cos(i * 0.3)
            
            x = math.sin(theta) * math.cos(phi_angle) * variance_factor
            y = math.sin(theta) * math.sin(phi_angle) * variance_factor
            
            # Map to [0.05, 0.95] range with extra margin to prevent boundary issues
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            # Add some geometric jitter to break symmetry
            jitter = 0.02 * np.sin(i * 1.3) * np.cos(i * 0.8)
            x += jitter * 0.5
            y += jitter * 0.5
            
            points.append([x, y])
            
        initial_points = np.array(points)
        
        # Apply Voronoi-based constraint adjustment
        # This ensures points don't cluster together too tightly
        adjusted_points = self._apply_voronoi_constraint(initial_points)
        
        return adjusted_points
    
    def _apply_voronoi_constraint(self, points: np.ndarray) -> np.ndarray:
        """Apply Voronoi-like constraint to ensure good distribution."""
        # Sample points to create a more uniform distribution
        # This reduces clustering effects that can hurt min/max ratio
        
        # Create a grid-based seed pattern that gets refined
        grid_points = []
        grid_size = int(math.ceil(math.sqrt(self.num_points)))
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(grid_points) >= self.num_points:
                    break
                x = 0.1 + 0.8 * i / (grid_size - 1) if grid_size > 1 else 0.5
                y = 0.1 + 0.8 * j / (grid_size - 1) if grid_size > 1 else 0.5
                grid_points.append([x, y])
        
        # Start with grid points and perturb them
        refined_points = np.array(grid_points[:self.num_points])
        
        # Add controlled random perturbations
        np.random.seed(42)
        for _ in range(50):
            # Pick a random point to adjust
            idx = np.random.randint(0, self.num_points)
            # Small perturbation with variance based on distance to neighbors
            neighbors = []
            for i in range(self.num_points):
                if i != idx:
                    dist = np.linalg.norm(refined_points[idx] - refined_points[i])
                    neighbors.append(dist)
            
            if neighbors:
                avg_dist = np.mean(neighbors)
                max_dist = np.max(neighbors)
                # Adjust perturbation based on distribution
                perturbation_scale = max(0.005, min(0.02, avg_dist * 0.1))
                if max_dist < 0.1:  # If points are very close, allow bigger moves
                    perturbation_scale = 0.03
                
                perturbation = np.random.normal(0, perturbation_scale, 2)
                refined_points[idx] += perturbation
                
                # Keep within bounds
                refined_points[idx][0] = np.clip(refined_points[idx][0], 0.001, 0.999)
                refined_points[idx][1] = np.clip(refined_points[idx][1], 0.001, 0.999)
        
        return refined_points
    
    def generate_structured_initial(self) -> np.ndarray:
        """Generate structured initial configuration with deliberate spacing."""
        # Create a 4x4 grid with hexagonal offset pattern
        points = []
        rows = cols = 4
        
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Apply hexagonal offset
        for i in range(rows):
            for j in range(cols):
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Apply slight perturbation to break exact symmetry
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                
                # Ensure within bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                
                points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def geometric_annealing_optimize(self, initial_points: np.ndarray) -> np.ndarray:
        """Main optimization using geometric annealing approach."""
        # Initialize
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio, _, _ = self.calculate_ratio(current_points)
        
        # Temperature schedule parameters
        initial_temp = 0.1
        final_temp = 1e-6
        alpha = 0.98  # Cooling rate
        
        # Iterations counter
        iteration = 0
        max_iterations = 2000
        
        # Track convergence
        stagnant_count = 0
        last_improvement = 0
        
        while iteration < max_iterations:
            # Temperature
            temperature = initial_temp * (alpha ** iteration)
            
            # Skip cooling below threshold
            if temperature < final_temp:
                temperature = final_temp
            
            # Try a perturbation
            test_points = current_points.copy()
            
            # Choose which point to modify
            idx = np.random.randint(0, self.num_points)
            
            # Determine perturbation size based on current solution quality
            ratio, _, _ = self.calculate_ratio(current_points)
            if ratio < 0.1:
                # Poor solution: larger steps
                step_size = 0.015
            elif ratio < 0.2:
                # Moderate solution: medium steps
                step_size = 0.01
            else:
                # Good solution: smaller steps
                step_size = 0.005
            
            # Apply perturbation with temperature-dependent magnitude
            perturbation = np.random.normal(0, step_size * temperature, 2)
            test_points[idx] += perturbation
            
            # Enforce bounds
            test_points[idx][0] = np.clip(test_points[idx][0], 0.001, 0.999)
            test_points[idx][1] = np.clip(test_points[idx][1], 0.001, 0.999)
            
            # Calculate ratio for new configuration
            test_ratio, _, _ = self.calculate_ratio(test_points)
            
            # Acceptance criteria with geometric consideration
            if test_ratio > best_ratio:
                # Always accept improvements
                current_points = test_points.copy()
                best_ratio = test_ratio
                best_points = current_points.copy()
                stagnant_count = 0
                last_improvement = iteration
            elif test_ratio > ratio:
                # Accept better solution with some probability
                if np.random.random() < np.exp((test_ratio - ratio) / temperature):
                    current_points = test_points.copy()
                stagnant_count += 1
            else:
                # Worse solution - accept with low probability
                if np.random.random() < np.exp((test_ratio - ratio) / temperature):
                    current_points = test_points.copy()
                stagnant_count += 1
            
            # Early stopping if stagnated for too long
            if stagnant_count > 500 and iteration - last_improvement > 300:
                break
                
            iteration += 1
            
        return best_points
    
    def get_best_solution(self, configs: List[np.ndarray]) -> np.ndarray:
        """Find best solution among all starting configurations."""
        best_ratio = -np.inf
        best_points = None
        
        # Try each initial configuration with geometric annealing
        for i, config in enumerate(configs):
            # Apply geometric annealing optimization
            optimized_points = self.geometric_annealing_optimize(config)
            
            ratio, _, _ = self.calculate_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
        
        return best_points if best_points is not None else configs[0]


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize evolver
    evolver = GeometricAnnealingEvolver(16, 2)
    
    # Generate multiple diversified initial configurations
    initial_configs = []
    
    # 1. Voronoi-inspired configuration
    initial_configs.append(evolver.generate_voronoi_initial())
    
    # 2. Structured grid pattern
    initial_configs.append(evolver.generate_structured_initial())
    
    # 3. Random with geometric constraints
    np.random.seed(42)
    random_points = np.random.uniform(0.05, 0.95, (16, 2))
    initial_configs.append(random_points)
    
    # 4. Another variant of structured pattern
    structured_config = evolver.generate_structured_initial()
    # Add variation with slight perturbation
    structured_config += np.random.normal(0, 0.01, structured_config.shape)
    structured_config[:, 0] = np.clip(structured_config[:, 0], 0.001, 0.999)
    structured_config[:, 1] = np.clip(structured_config[:, 1], 0.001, 0.999)
    initial_configs.append(structured_config)
    
    # 5. Another random variant
    np.random.seed(123)
    random_points2 = np.random.uniform(0.05, 0.95, (16, 2))
    initial_configs.append(random_points2)
    
    # Find best solution using geometric annealing
    best_points = evolver.get_best_solution(initial_configs)
    
    # Final refinement with a quick optimization
    try:
        ratio, _, _ = evolver.calculate_ratio(best_points)
        if ratio < 0.25:  # If not very good, do another round of geometric annealing
            best_points = evolver.geometric_annealing_optimize(best_points)
    except:
        pass
    
    return best_points

# EVOLVE-BLOCK-END
