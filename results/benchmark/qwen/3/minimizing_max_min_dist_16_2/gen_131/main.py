# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
import random
from typing import Tuple, List, Optional
import warnings

class PointArrangementOptimizer:
    """Enhanced optimizer for 16-point arrangement maximizing min/max distance ratio."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.max_eval_time = 175.0
        
    def compute_min_max_ratio(self, points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0
        
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def generate_hexagonal_grid(self) -> np.ndarray:
        """Generate points in a hexagonal grid pattern with symmetry breaking."""
        points = []
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                x_base = j * 0.25 + (i % 2) * 0.125
                y_base = i * 0.25
                
                # Add deterministic symmetry-breaking pattern
                symmetry_factor = (i * 7 + j * 3) % 10
                x_pert = np.sin(symmetry_factor * 0.5) * 0.005
                y_pert = np.cos(symmetry_factor * 0.3) * 0.005
                
                x = x_base + x_pert + np.random.normal(0, 0.003)
                y = y_base + y_pert + np.random.normal(0, 0.003)
                points.append([x, y])
        
        result = np.array(points)
        return np.clip(result, 0, 1)
    
    def generate_fibonacci_spiral(self) -> np.ndarray:
        """Generate points using Fibonacci spiral for good distribution."""
        points = np.zeros((16, 2))
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(16):
            z = 1 - (i / 15) * 2  # z from -1 to 1
            radius = np.sqrt(1 - z*z)
            theta = np.arccos(z)
            phi = (i * golden_ratio) % (2 * np.pi)
            
            x = radius * np.cos(phi)
            y = radius * np.sin(phi)
            
            # Map to [0,1] x [0,1]
            x_norm = (x + 1) / 2
            y_norm = (y + 1) / 2
            
            points[i] = [x_norm, y_norm]
        
        # Add small perturbations
        points += np.random.normal(0, 0.01, points.shape)
        return np.clip(points, 0, 1)
    
    def generate_triangular_lattice(self) -> np.ndarray:
        """Generate points in a triangular lattice pattern."""
        points = []
        # Using triangular lattice parameters for 16 points
        spacing = 0.25
        spacing_y = spacing * np.sqrt(3) / 2
        
        for i in range(4):
            for j in range(4):
                if len(points) < 16:
                    x = j * spacing + (i % 2) * spacing / 2
                    y = i * spacing_y
                    
                    # Add asymmetry
                    asym_x = np.sin(i * 1.7) * 0.01 * (1 + j * 0.1)
                    asym_y = np.cos(j * 1.9) * 0.01 * (1 + i * 0.1)
                    
                    points.append([x + asym_x + np.random.normal(0, 0.002),
                                 y + asym_y + np.random.normal(0, 0.002)])
        
        result = np.array(points[:16])
        return np.clip(result, 0, 1)
    
    def generate_random_points(self) -> np.ndarray:
        """Generate random points."""
        return np.random.rand(16, 2)
    
    def generate_initial_population(self) -> List[np.ndarray]:
        """Generate diverse initial configurations."""
        configs = []
        
        # Add different geometric configurations
        configs.append(self.generate_hexagonal_grid())
        configs.append(self.generate_fibonacci_spiral())
        configs.append(self.generate_triangular_lattice())
        configs.append(self.generate_random_points())
        
        # Add variations with different noise levels
        for _ in range(2):
            base_config = self.generate_hexagonal_grid()
            base_config += np.random.normal(0, 0.005, base_config.shape)
            configs.append(np.clip(base_config, 0, 1))
        
        return configs
    
    def optimize_with_simulated_annealing(self, initial_points: np.ndarray) -> np.ndarray:
        """Optimize using enhanced simulated annealing."""
        start_time = time.time()
        
        points = np.clip(initial_points, 0, 1)
        current_ratio = self.compute_min_max_ratio(points)
        
        # Enhanced parameters
        temperature = 1.0
        cooling_rate = 0.9995
        min_temperature = 1e-6
        max_iterations = 50000
        iteration = 0
        
        best_points = points.copy()
        best_ratio = current_ratio
        
        # Track recent improvements for adaptive cooling
        recent_improvements = []
        patience = 0
        max_patience = 1000
        
        # Adaptive cooling logic
        adaptive_cooling = True
        
        while (temperature > min_temperature and 
               iteration < max_iterations and 
               (time.time() - start_time) < self.max_eval_time):
            
            # Create candidate solution with cluster moves
            candidate_points = points.copy()
            
            # Choose move type: cluster move (30%) or individual move (70%)
            if random.random() < 0.3:
                # Cluster move: move 2-4 points together
                num_points_to_move = random.randint(2, 4)
                selected_indices = random.sample(range(len(points)), num_points_to_move)
                
                # Calculate centroid of selected points
                centroid = np.mean(candidate_points[selected_indices], axis=0)
                
                # Move centroid and adjust all points
                move_vector = np.random.normal(0, 0.015, 2)
                new_centroid = np.clip(centroid + move_vector, 0, 1)
                delta = new_centroid - centroid
                
                for idx in selected_indices:
                    candidate_points[idx] += delta
            else:
                # Individual point move
                idx = np.random.randint(0, len(points))
                candidate_points[idx] += np.random.normal(0, 0.02, 2)
            
            # Keep within bounds
            candidate_points = np.clip(candidate_points, 0, 1)
            
            # Calculate acceptance probability
            candidate_ratio = self.compute_min_max_ratio(candidate_points)
            
            # Accept or reject based on Metropolis criterion
            if (candidate_ratio > current_ratio or 
                np.random.rand() < np.exp((candidate_ratio - current_ratio) / temperature)):
                
                points = candidate_points
                current_ratio = candidate_ratio
                
                # Update best solution
                if current_ratio > best_ratio:
                    best_points = points.copy()
                    best_ratio = current_ratio
                    recent_improvements = []
                    patience = 0
                else:
                    patience += 1
                    recent_improvements.append(current_ratio)
                    if len(recent_improvements) > 50:
                        recent_improvements.pop(0)
            else:
                patience += 1
            
            # Early stopping
            if patience > max_patience:
                if len(recent_improvements) > 10:
                    recent_avg = np.mean(recent_improvements[-10:])
                    if recent_avg > 0.99 * best_ratio:
                        break
            
            # Adaptive cooling
            if adaptive_cooling:
                if temperature > 0.1:
                    temperature *= cooling_rate  
                else:
                    temperature *= 0.99999
            
            iteration += 1
        
        return best_points
    
    def local_refinement(self, points: np.ndarray) -> np.ndarray:
        """Apply local refinement to improve final solution."""
        current_points = points.copy()
        current_ratio = self.compute_min_max_ratio(current_points)
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Gradient-free local search
        for _ in range(200):
            improved = False
            
            # Try small perturbations to each point
            for i in range(len(current_points)):
                original_point = current_points[i].copy()
                
                # Try perturbing this point
                new_point = original_point + np.random.normal(0, 0.005, 2)
                new_point = np.clip(new_point, 0, 1)
                
                current_points[i] = new_point
                new_ratio = self.compute_min_max_ratio(current_points)
                
                if new_ratio > current_ratio:
                    current_ratio = new_ratio
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_points = current_points.copy()
                    improved = True
                else:
                    current_points[i] = original_point
            
            # Occasionally try larger perturbations
            if np.random.random() < 0.1:
                indices_to_perturb = np.random.choice(len(current_points), 
                                                    size=max(1, len(current_points) // 10), 
                                                    replace=False)
                for idx in indices_to_perturb:
                    original_point = current_points[idx].copy()
                    new_point = original_point + np.random.normal(0, 0.01, 2)
                    new_point = np.clip(new_point, 0, 1)
                    
                    current_points[idx] = new_point
                    new_ratio = self.compute_min_max_ratio(current_points)
                    
                    if new_ratio > current_ratio:
                        current_ratio = new_ratio
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_points = current_points.copy()
                        improved = True
                    else:
                        current_points[idx] = original_point
            
            # Stop if no improvement for many iterations
            if not improved:
                break
                
        return best_points
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine."""
        # Generate initial population
        initial_configs = self.generate_initial_population()
        
        best_points = None
        best_ratio = -np.inf
        
        # Try multiple optimization paths
        for i, initial_config in enumerate(initial_configs):
            try:
                # Optimize from this initial configuration
                optimized_points = self.optimize_with_simulated_annealing(initial_config)
                
                # Local refinement
                refined_points = self.local_refinement(optimized_points)
                
                # Evaluate final result
                final_ratio = self.compute_min_max_ratio(refined_points)
                
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = refined_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Error in optimization path {i}: {str(e)}")
                continue
        
        # Fallback if nothing worked
        if best_points is None:
            fallback_points = self.generate_hexagonal_grid()
            best_points = self.local_refinement(fallback_points)
        
        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointArrangementOptimizer(seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END
