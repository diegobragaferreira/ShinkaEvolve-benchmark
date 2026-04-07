# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.cluster import KMeans

class HexagonForceRadiusOptimizer:
    def __init__(self, n_circles=32):
        self.n = n_circles
        self.circles = np.zeros((n_circles, 3))
        self.radii = None
        self.positions = None
        
    def initialize_advanced_hexagonal(self):
        """Create advanced hexagonal grid with optimized spacing"""
        # Create initial hexagonal grid with proper dimensions
        rows = 6
        cols = 6
        spacing_x = 0.15  # Adjusted spacing
        spacing_y = 0.15 * np.sqrt(3)
        
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
        
        # Set initial positions with variable but reasonable radii
        base_radius = 0.04
        for i, (x, y) in enumerate(positions):
            # Vary initial radii slightly for better optimization landscape
            variation = np.random.uniform(0.8, 1.2)
            self.circles[i] = [x, y, base_radius * variation]
            
        self.update_arrays()
        
    def initialize_voronoi_distribution(self):
        """Apply Voronoi-based distribution to improve initial spacing"""
        # Use k-means clustering to find good initial centers
        # Then place circles around these centers with proper spacing
        initial_positions = self.circles[:, :2].copy()
        
        # Apply k-means to get better distribution
        kmeans = KMeans(n_clusters=self.n, random_state=42, n_init=10)
        kmeans.fit(initial_positions)
        
        # Use cluster centers as new positions
        new_positions = kmeans.cluster_centers_
        
        # Ensure positions are within bounds
        for i in range(self.n):
            x, y = new_positions[i]
            new_positions[i] = [
                np.clip(x, self.circles[i, 2], 1.0 - self.circles[i, 2]),
                np.clip(y, self.circles[i, 2], 1.0 - self.circles[i, 2])
            ]
        
        # Update positions while keeping radii
        for i in range(self.n):
            self.circles[i, 0] = new_positions[i][0]
            self.circles[i, 1] = new_positions[i][1]
            
        self.update_arrays()
        
    def update_arrays(self):
        """Update cached arrays for efficient access"""
        self.positions = self.circles[:, :2]
        self.radii = self.circles[:, 2]
        
    def compute_adaptive_forces(self):
        """Compute adaptive repulsion forces with momentum"""
        # Vectorized distance matrix computation
        dist_matrix = cdist(self.positions, self.positions)
        
        # Build overlap information
        radii_sum = np.add.outer(self.radii, self.radii)
        np.fill_diagonal(radii_sum, np.inf)
        
        # Find overlaps
        overlap_mask = dist_matrix < radii_sum
        
        # Initialize force magnitudes
        force_magnitudes = np.zeros_like(self.positions)
        
        # Weighted force computation with momentum effect
        for i in range(self.n):
            overlap_indices = np.where(overlap_mask[i])[0]
            for j in overlap_indices:
                if i != j:
                    dx = self.positions[j, 0] - self.positions[i, 0]
                    dy = self.positions[j, 1] - self.positions[i, 1]
                    dist_ij = max(np.sqrt(dx*dx + dy*dy), 1e-10)
                    
                    # Adaptive force calculation with inverse distance weighting
                    overlap = self.radii[i] + self.radii[j] - dist_ij
                    if overlap > 0:
                        # More aggressive force for tighter overlaps
                        force_mag = overlap * 0.1 * (1.0 / (dist_ij + 1e-8))
                        
                        # Apply force with direction normalization
                        force_magnitudes[i, 0] -= force_mag * dx / dist_ij
                        force_magnitudes[i, 1] -= force_mag * dy / dist_ij
                        
        return force_magnitudes
        
    def update_with_momentum(self, forces, step_size=0.02, momentum_factor=0.8):
        """Update positions with momentum for smoother convergence"""
        # Simple momentum-based update
        new_positions = self.positions + forces * step_size
        
        # Apply boundary constraints
        for i in range(self.n):
            min_r = self.radii[i]
            new_positions[i, 0] = np.clip(new_positions[i, 0], min_r, 1.0 - min_r)
            new_positions[i, 1] = np.clip(new_positions[i, 1], min_r, 1.0 - min_r)
            
        # Update positions in circles array
        self.circles[:, :2] = new_positions
        self.update_arrays()
        
    def expand_radii_smartly(self):
        """Smart radius expansion using constrained optimization approach"""
        # Create a more sophisticated expansion algorithm
        improved = True
        max_attempts = 50
        
        for attempt in range(max_attempts):
            improved = False
            
            # Create ordering based on geometric factors
            # Circles with more constraints (surrounded by many large circles) get priority
            priorities = np.zeros(self.n)
            for i in range(self.n):
                # Count nearby circles (for now just count all)
                nearby_count = 0
                for j in range(self.n):
                    if i != j:
                        dx = self.circles[i, 0] - self.circles[j, 0]
                        dy = self.circles[i, 1] - self.circles[j, 1]
                        dist_ij = np.sqrt(dx*dx + dy*dy)
                        if dist_ij < (self.circles[i, 2] + self.circles[j, 2]) * 1.5:  # In influence zone
                            nearby_count += 1
                priorities[i] = nearby_count
                
            # Sort by priority (lower priority first)
            sorted_indices = np.argsort(priorities)
            
            # Try to expand each circle
            for i in sorted_indices:
                current_radius = self.circles[i, 2]
                # Try several growth steps
                for growth_step in [0.002, 0.003, 0.004]:
                    new_radius = current_radius + growth_step
                    
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
                        if dist_ij < (new_radius + self.circles[j, 2]):
                            valid_overlap = False
                            break
                            
                    if valid_overlap:
                        self.circles[i, 2] = new_radius
                        improved = True
                        break
                        
        self.update_arrays()
        
    def smart_validation_and_correction(self):
        """Advanced validation and correction with iterative refinement"""
        # Multiple passes to clean up any constraint violations
        for pass_num in range(5):
            # Try to fix overlaps
            for i in range(self.n):
                # Check boundary violations
                x, y, r = self.circles[i]
                corrected_x = np.clip(x, r, 1.0 - r)
                corrected_y = np.clip(y, r, 1.0 - r)
                
                if corrected_x != x or corrected_y != y:
                    self.circles[i, 0] = corrected_x
                    self.circles[i, 1] = corrected_y
                    
                # Fix overlaps with other circles
                for j in range(self.n):
                    if i != j:
                        dx = self.circles[i, 0] - self.circles[j, 0]
                        dy = self.circles[i, 1] - self.circles[j, 1]
                        dist_ij = np.sqrt(dx*dx + dy*dy)
                        min_dist = self.circles[i, 2] + self.circles[j, 2]
                        
                        if dist_ij < min_dist:
                            # Move circle away from the other
                            if dist_ij > 0:
                                scale = (min_dist - dist_ij) / dist_ij * 0.5
                                self.circles[i, 0] += dx * scale
                                self.circles[i, 1] += dy * scale
                                
                                # Clamp to boundary
                                self.circles[i, 0] = np.clip(
                                    self.circles[i, 0], 
                                    self.circles[i, 2], 
                                    1.0 - self.circles[i, 2]
                                )
                                self.circles[i, 1] = np.clip(
                                    self.circles[i, 1], 
                                    self.circles[i, 2], 
                                    1.0 - self.circles[i, 2]
                                )
                                
        self.update_arrays()
        
    def optimize(self):
        """Main optimization loop with enhanced phases"""
        max_iterations = 800
        
        for iteration in range(max_iterations):
            # Phase 1: Force-based optimization with adaptive weights
            forces = self.compute_adaptive_forces()
            step_size = 0.02
            if iteration > 400:
                step_size = 0.01
            elif iteration > 200:
                step_size = 0.015
                
            self.update_with_momentum(forces, step_size)
            
            # Phase 2: Radius expansion at specific intervals
            if iteration % 8 == 0:
                self.expand_radii_smartly()
                
            # Phase 3: Adaptive parameter adjustment
            if iteration % 150 == 0 and iteration > 0:
                # Sometimes reset to Voronoi-inspired distribution for better exploration
                if iteration % 300 == 0:
                    self.initialize_voronoi_distribution()
                    
        # Final refinement
        self.smart_validation_and_correction()

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Create optimizer instance
    optimizer = HexagonForceRadiusOptimizer(32)
    
    # Initialize with advanced hexagonal grid
    optimizer.initialize_advanced_hexagonal()
    
    # Apply Voronoi-based refinement
    optimizer.initialize_voronoi_distribution()
    
    # Run optimization
    optimizer.optimize()
    
    return optimizer.circles

# EVOLVE-BLOCK-END