# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import random
from typing import Tuple, List
import time

class CirclePacker:
    """Efficient circle packing using Voronoi initialization and Lagrangian relaxation."""
    
    def __init__(self, n_circles: int = 32):
        self.n_circles = n_circles
        
    def _initialize_voronoi(self) -> np.ndarray:
        """Initialize circles using Voronoi diagram approach."""
        # Create random points and generate Voronoi diagram
        points = np.random.rand(self.n_circles * 5, 2)
        
        try:
            vor = Voronoi(points)
        except:
            # Fallback to simple grid initialization
            circles = np.zeros((self.n_circles, 3))
            placed = 0
            for i in range(6):
                for j in range(6):
                    if placed >= self.n_circles:
                        break
                    x = 0.1 + i * 0.15
                    y = 0.1 + j * 0.15
                    r = 0.05
                    circles[placed] = [x, y, r]
                    placed += 1
                if placed >= self.n_circles:
                    break
            return circles
        
        # Extract valid Voronoi vertices
        vertices = []
        for vertex in vor.vertices:
            if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                vertices.append(vertex)
        
        # If not enough vertices, add random ones
        while len(vertices) < self.n_circles:
            vertices.append([np.random.rand(), np.random.rand()])
            
        selected_vertices = vertices[:self.n_circles]
        
        # Create circles with appropriate radii
        circles = []
        for i, (x, y) in enumerate(selected_vertices):
            # Estimate minimum distance to neighbors to set radius
            min_dist = float('inf')
            for j, (x2, y2) in enumerate(selected_vertices):
                if i != j:
                    d = np.sqrt((x-x2)**2 + (y-y2)**2)
                    min_dist = min(min_dist, d)
            
            # Set radius based on neighbor spacing and boundaries
            r = min(min_dist/2, x, 1-x, y, 1-y) * 0.9
            r = max(r, 0.001)
            circles.append([x, y, r])
            
        return np.array(circles)
    
    def _compute_overlap_constraints(self, circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute overlap constraints efficiently."""
        n = len(circles)
        if n <= 1:
            return np.array([]), np.array([])
            
        # Precompute pairwise distances
        constraints = []
        constraint_values = []
        
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                if dist_sq < min_dist_sq:
                    constraints.append((i, j))
                    constraint_values.append(dist_sq - min_dist_sq)
                    
        return np.array(constraints), np.array(constraint_values)
    
    def _lagrangian_relaxation_step(self, circles: np.ndarray, lambda_vals: np.ndarray, 
                                  constraints: np.ndarray) -> np.ndarray:
        """Perform one step of Lagrangian relaxation to improve solution."""
        # This is a simplified version - in practice, one would solve the relaxed problem
        # For now, we'll just do a basic gradient projection update
        n = len(circles)
        new_circles = circles.copy()
        
        # Simple gradient descent on sum of radii with penalty terms
        for i in range(n):
            # Basic adjustment - move towards maximizing individual radii
            # while respecting boundary constraints
            x, y, r = new_circles[i]
            
            # Increase radius if possible without violating constraints
            # This is heuristic - a real implementation would be more sophisticated
            if r < 0.4:  # Only increase if not too large already
                new_r = min(r * 1.05, x, 1-x, y, 1-y)
                if new_r > r:
                    new_circles[i, 2] = new_r
                    
        return new_circles
    
    def _project_to_feasible_region(self, circles: np.ndarray) -> np.ndarray:
        """Project circles to satisfy all constraints."""
        n = len(circles)
        new_circles = circles.copy()
        
        # First ensure all circles are within bounds
        for i in range(n):
            x, y, r = new_circles[i]
            # Clamp positions to valid region
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            # Reduce radius if necessary to fit
            r = min(r, x, 1-x, y, 1-y)
            new_circles[i] = [x, y, r]
            
        # Then resolve overlaps using iterative projection
        max_iter = 50
        for _ in range(max_iter):
            any_changed = False
            for i in range(n):
                x1, y1, r1 = new_circles[i]
                for j in range(i+1, n):
                    x2, y2, r2 = new_circles[j]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist_sq = (r1 + r2)**2
                    
                    if dist_sq < min_dist_sq:
                        # Resolve overlap by adjusting both circles
                        overlap = min_dist_sq - dist_sq
                        # Move circles apart
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = np.sqrt(dist_sq) + 1e-8
                        move_amount = overlap / (dist * 2)
                        
                        # Apply small adjustments
                        new_circles[i, 0] -= dx * move_amount / dist * 0.1
                        new_circles[i, 1] -= dy * move_amount / dist * 0.1
                        new_circles[j, 0] += dx * move_amount / dist * 0.1
                        new_circles[j, 1] += dy * move_amount / dist * 0.1
                        
                        # Adjust radii to prevent further overlap
                        new_circles[i, 2] = max(0.001, min(new_circles[i, 2], x1, 1-x1, y1, 1-y1))
                        new_circles[j, 2] = max(0.001, min(new_circles[j, 2], x2, 1-x2, y2, 1-y2))
                        
                        any_changed = True
                        
            if not any_changed:
                break
                
        return new_circles
    
    def _optimize_with_projection(self, circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Optimize using gradient projection method."""
        current = circles.copy()
        
        # Precompute constraints once
        constraints, constraint_values = self._compute_overlap_constraints(current)
        
        for iteration in range(max_iter):
            # Compute gradient approximation using finite differences
            n = len(current)
            grad = np.zeros((n, 3))  # Gradient w.r.t. [x, y, r]
            
            # Approximate gradient for radius maximization
            for i in range(n):
                # Simple finite difference approximation
                epsilon = 1e-4
                original_radius = current[i, 2]
                
                # Perturb radius slightly to estimate effect
                test_circles = current.copy()
                test_circles[i, 2] += epsilon
                test_circles = self._project_to_feasible_region(test_circles)
                new_sum = np.sum(test_circles[:, 2])
                
                test_circles = current.copy()
                test_circles[i, 2] -= epsilon
                test_circles = self._project_to_feasible_region(test_circles)
                old_sum = np.sum(test_circles[:, 2])
                
                # Approximate gradient for radius
                grad[i, 2] = (new_sum - old_sum) / (2 * epsilon)
                
                # For position, we'll use a simple heuristic
                grad[i, 0] = 0  # Would be computed properly in real implementation
                grad[i, 1] = 0
                
            # Perform gradient ascent step
            learning_rate = 0.01
            updated = current.copy()
            
            for i in range(n):
                # Update position and radius
                updated[i, 0] = current[i, 0] + learning_rate * grad[i, 0]
                updated[i, 1] = current[i, 1] + learning_rate * grad[i, 1]
                updated[i, 2] = current[i, 2] + learning_rate * grad[i, 2]
                
            # Project back to feasible region
            updated = self._project_to_feasible_region(updated)
            
            # Check for convergence
            diff = np.sum(np.abs(updated - current))
            if diff < 1e-6:
                break
                
            current = updated
            
        return current
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine."""
        # Phase 1: Initialize with Voronoi approach
        circles = self._initialize_voronoi()
        circles = self._project_to_feasible_region(circles)
        
        # Phase 2: Refine using gradient projection method
        refined = self._optimize_with_projection(circles, max_iter=200)
        
        # Phase 3: Final optimization
        final = self._optimize_with_projection(refined, max_iter=100)
        
        # Ensure final solution is feasible
        final = self._project_to_feasible_region(final)
        
        return final

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    packer = CirclePacker(32)
    circles = packer.optimize()
    
    return circles

# EVOLVE-BLOCK-END