# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

class CirclePackingOptimizer:
    def __init__(self, n_circles=32):
        self.n = n_circles
        self.circles = np.zeros((n_circles, 3))
        self.radii = None
        self.positions = None
        
    def initialize_hexagonal_grid(self):
        """Initialize circles using a refined hexagonal grid pattern"""
        # Better hexagonal layout for 32 circles
        rows = 6
        cols = 6
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                x = (j + 1) * spacing_x + ((i % 2) * spacing_x / 2)
                y = (i + 1) * spacing_y
                if x <= 1.0 and y <= 1.0:
                    positions.append([x, y])
                    if len(positions) >= self.n:
                        break
            if len(positions) >= self.n:
                break
        
        # Ensure exact count
        positions = positions[:self.n]
        
        # Set initial positions with reasonable radii
        initial_radius = 0.05
        for i, (x, y) in enumerate(positions):
            self.circles[i] = [x, y, initial_radius]
            
        self.update_arrays()
        
    def update_arrays(self):
        """Update cached arrays for efficient access"""
        self.positions = self.circles[:, :2]
        self.radii = self.circles[:, 2]
        
    def compute_forces(self):
        """Compute repulsion forces between overlapping circles"""
        # Vectorized distance matrix computation
        dist_matrix = cdist(self.positions, self.positions)
        
        # Create overlap matrix
        radii_sum = np.add.outer(self.radii, self.radii)
        np.fill_diagonal(radii_sum, np.inf)
        
        # Find overlaps
        overlap_mask = dist_matrix < radii_sum
        
        # Initialize force magnitudes
        force_magnitudes = np.zeros_like(self.positions)
        
        # Compute forces for overlapping pairs
        for i in range(self.n):
            overlap_indices = np.where(overlap_mask[i])[0]
            for j in overlap_indices:
                if i != j:
                    dx = self.positions[j, 0] - self.positions[i, 0]
                    dy = self.positions[j, 1] - self.positions[i, 1]
                    dist_ij = max(np.sqrt(dx*dx + dy*dy), 1e-10)
                    
                    # Repulsion force
                    overlap = self.radii[i] + self.radii[j] - dist_ij
                    if overlap > 0:
                        force_mag = overlap * 0.05
                        
                        # Apply force
                        force_magnitudes[i, 0] -= force_mag * dx / dist_ij
                        force_magnitudes[i, 1] -= force_mag * dy / dist_ij
                        
        return force_magnitudes
        
    def update_positions(self, forces, step_size=0.02):
        """Update circle positions with boundary constraints"""
        new_positions = self.positions + forces * step_size
        
        # Apply boundary constraints
        for i in range(self.n):
            min_r = self.radii[i]
            new_positions[i, 0] = np.clip(new_positions[i, 0], min_r, 1.0 - min_r)
            new_positions[i, 1] = np.clip(new_positions[i, 1], min_r, 1.0 - min_r)
            
        # Update positions in circles array
        self.circles[:, :2] = new_positions
        self.update_arrays()
        
    def expand_radii(self):
        """Attempt to increase radii while maintaining constraints"""
        # Sort by radius (smaller first) to prioritize expansion
        sorted_indices = np.argsort(self.radii)
        
        growth_factor = 0.003
        improved = True
        attempts = 0
        max_attempts = 100
        
        while improved and attempts < max_attempts:
            improved = False
            attempts += 1
            
            # Try to grow each circle in order of increasing radius
            for i in sorted_indices:
                current_radius = self.radii[i]
                new_radius = current_radius + growth_factor
                
                # Check containment constraints
                if (new_radius > self.circles[i, 0] or 
                    new_radius > 1.0 - self.circles[i, 0] or
                    new_radius > self.circles[i, 1] or
                    new_radius > 1.0 - self.circles[i, 1]):
                    continue
                    
                # Check overlap constraints
                valid_overlap = True
                for j in range(self.n):
                    if i == j:
                        continue
                    dx = self.circles[i, 0] - self.circles[j, 0]
                    dy = self.circles[i, 1] - self.circles[j, 1]
                    dist_ij = np.sqrt(dx*dx + dy*dy)
                    if dist_ij < (new_radius + self.radii[j]):
                        valid_overlap = False
                        break
                        
                if valid_overlap:
                    self.circles[i, 2] = new_radius
                    improved = True
                    
        self.update_arrays()
        
    def validate_and_correct(self):
        """Final validation to ensure all constraints are met"""
        # Recheck all constraints and correct minor violations
        for i in range(self.n):
            # Boundary check
            x, y, r = self.circles[i]
            self.circles[i, 0] = np.clip(x, r, 1.0 - r)
            self.circles[i, 1] = np.clip(y, r, 1.0 - r)
            
            # Overlap check with neighbors
            for j in range(self.n):
                if i != j:
                    dx = self.circles[i, 0] - self.circles[j, 0]
                    dy = self.circles[i, 1] - self.circles[j, 1]
                    dist_ij = np.sqrt(dx*dx + dy*dy)
                    min_dist = self.circles[i, 2] + self.circles[j, 2]
                    
                    if dist_ij < min_dist:
                        # Resolve by moving apart
                        if dist_ij > 0:
                            scale = (min_dist - dist_ij) / dist_ij * 0.5
                            dx *= scale
                            dy *= scale
                            self.circles[i, 0] -= dx
                            self.circles[i, 1] -= dy
                            self.circles[i, 0] = np.clip(self.circles[i, 0], 
                                                       self.circles[i, 2], 1.0 - self.circles[i, 2])
                            self.circles[i, 1] = np.clip(self.circles[i, 1], 
                                                       self.circles[i, 2], 1.0 - self.circles[i, 2])
                            
        self.update_arrays()
        
    def optimize(self):
        """Main optimization loop with multiple phases"""
        max_iterations = 1000
        
        for iteration in range(max_iterations):
            # Phase 1: Force-based optimization
            forces = self.compute_forces()
            step_size = 0.02
            if iteration > 500:
                step_size = 0.01
            elif iteration > 200:
                step_size = 0.015
                
            self.update_positions(forces, step_size)
            
            # Phase 2: Radius expansion
            if iteration % 10 == 0:  # Do radius expansion periodically
                self.expand_radii()
                
            # Phase 3: Adaptive parameter adjustment
            if iteration % 200 == 0 and iteration > 0:
                # Reduce growth factor gradually
                pass  # Growth factor already handled in expand_radii
                
            # Occasionally re-scale for better convergence
            if iteration % 100 == 0:
                pass  # Step size adapts automatically
                
        # Final validation and refinement
        self.validate_and_correct()

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Create optimizer instance
    optimizer = CirclePackingOptimizer(32)
    
    # Initialize with hexagonal grid
    optimizer.initialize_hexagonal_grid()
    
    # Run optimization
    optimizer.optimize()
    
    return optimizer.circles

# EVOLVE-BLOCK-END