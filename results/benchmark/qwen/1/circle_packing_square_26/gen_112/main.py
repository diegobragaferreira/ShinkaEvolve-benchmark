# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from itertools import product
import time

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize sum of radii.
    Uses Voronoi lattice optimization with geometric constraint solving.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    
    n_circles = 26
    
    # Phase 1: Generate high-quality initial configuration using Voronoi lattice
    initial_config = generate_voronoi_lattice_config(n_circles)
    
    # Phase 2: Optimize using geometric constraint solver
    optimized_config = optimize_with_geometric_constraints(initial_config)
    
    # Phase 3: Fine-tune with local refinement
    final_config = local_refinement(optimized_config)
    
    return np.array(final_config)

def generate_voronoi_lattice_config(n):
    """Generate initial configuration using Voronoi lattice approach with lattice sampling"""
    
    # Create lattice points that form a triangular/hexagonal pattern
    # This gives better coverage and more uniform distribution
    
    # Determine lattice size
    sqrt_n = int(np.ceil(np.sqrt(n)))
    lattice_rows = sqrt_n
    lattice_cols = sqrt_n
    
    # Create hexagonal lattice points
    lattice_points = []
    spacing = 0.8 / max(lattice_rows, lattice_cols)
    
    # Hexagonal packing offset
    for i in range(lattice_rows):
        for j in range(lattice_cols):
            x = 0.1 + j * spacing
            y = 0.1 + i * spacing * np.sqrt(3)/2
            
            # Apply horizontal offset for even rows
            if i % 2 == 1:
                x += spacing/2
            
            if x <= 0.9 and y <= 0.9:
                lattice_points.append([x, y])
                
    # If we don't have enough points, add extra points
    while len(lattice_points) < n:
        # Add random points to fill gaps
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        lattice_points.append([x, y])
    
    # Trim to exact number
    lattice_points = lattice_points[:n]
    
    # Create initial configuration with maximum possible radii at each point
    config = []
    for i, (x, y) in enumerate(lattice_points):
        # Calculate maximum radius at this point without violating boundaries
        max_radius = min(x, 1-x, y, 1-y)
        if max_radius > 0:
            # Start with 80% of maximum radius to allow for overlaps during optimization
            initial_radius = max_radius * 0.8
            config.append([x, y, initial_radius])
        else:
            # Fallback small radius
            config.append([x, y, 0.02])
    
    return config

def optimize_with_geometric_constraints(config):
    """Use geometric optimization to resolve overlaps and improve configuration"""
    
    # Create initial working copy
    working_config = [list(circle) for circle in config]
    
    # Iterative improvement using geometric constraints
    max_iterations = 100
    improvement_threshold = 1e-6
    
    for iteration in range(max_iterations):
        improved = False
        
        # For each circle, try to maximize radius while maintaining constraints
        for i in range(len(working_config)):
            original_circle = working_config[i]
            original_x, original_y, original_r = original_circle
            
            # Calculate new maximum radius for this circle
            new_max_radius = min(original_x, 1-original_x, original_y, 1-original_y)
            
            if new_max_radius <= original_r:
                continue
                
            # Check what we can safely increase to
            safe_radius = new_max_radius
            
            # Check overlap constraints with all other circles
            for j in range(len(working_config)):
                if i != j:
                    xj, yj, rj = working_config[j]
                    dist = np.sqrt((original_x - xj)**2 + (original_y - yj)**2)
                    # We need: dist >= (original_r + rj) for no overlap
                    # So: original_r <= dist - rj  
                    max_safe_radius = dist - rj - 1e-8
                    if max_safe_radius > 0:
                        safe_radius = min(safe_radius, max_safe_radius)
            
            # Increase radius if beneficial
            if safe_radius > original_r:
                working_config[i] = [original_x, original_y, safe_radius]
                improved = True
                
        # Also try to move circles to reduce overlaps
        for i in range(len(working_config)):
            x, y, r = working_config[i]
            
            # Try to find a better position
            best_x, best_y = x, y
            best_r = r
            best_improvement = 0
            
            # Sample potential positions around current location
            for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                    test_x = max(0.01, min(0.99, x + dx))
                    test_y = max(0.01, min(0.99, y + dy))
                    test_r = r
                    
                    # Check if this position is valid
                    valid = True
                    for j in range(len(working_config)):
                        if i != j:
                            xj, yj, rj = working_config[j]
                            dist = np.sqrt((test_x - xj)**2 + (test_y - yj)**2)
                            if dist < (test_r + rj + 1e-8):
                                valid = False
                                break
                    
                    if valid:
                        # Calculate improvement - radius increase with better position
                        new_max_r = min(test_x, 1-test_x, test_y, 1-test_y)
                        improvement = min(new_max_r, test_r) - test_r
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_x, best_y, best_r = test_x, test_y, test_r
            
            if best_improvement > 0:
                working_config[i] = [best_x, best_y, best_r]
                improved = True
        
        if not improved:
            break
    
    return working_config

def local_refinement(config):
    """Apply local refinement to maximize sum of radii"""
    
    # Convert to numpy for efficient operations
    circles = np.array(config)
    n = len(circles)
    
    # Apply binary search for each circle to maximize its radius
    for attempt in range(50):
        improved = False
        
        # Try to increase radius of each circle
        for i in range(n):
            x, y, r = circles[i]
            
            # Maximum increase allowed by boundary constraints
            max_boundary_increase = min(x, 1-x, y, 1-y) - r
            
            if max_boundary_increase <= 0:
                continue
                
            # Maximum increase allowed by overlap constraints
            max_overlap_increase = float('inf')
            
            # Check overlap with all other circles
            for j in range(n):
                if i != j:
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                    
                    # Maximum allowable radius to maintain separation
                    max_safe_radius = dist - rj - 1e-8
                    if max_safe_radius > 0:
                        max_overlap_increase = min(max_overlap_increase, max_safe_radius)
            
            # Determine safe increase amount
            safe_increase = min(max_boundary_increase, max_overlap_increase)
            
            if safe_increase > 1e-6:
                # Binary search for optimal increase
                low = 0
                high = safe_increase
                best_r = r
                
                for _ in range(10):  # binary search iterations
                    test_r = (low + high) / 2
                    
                    # Check if this increase is valid
                    valid = True
                    test_r_total = r + test_r
                    
                    for j in range(n):
                        if i != j:
                            xj, yj, rj = circles[j]
                            dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                            
                            if dist < (test_r_total + rj + 1e-8):
                                valid = False
                                break
                    
                    if valid:
                        best_r = r + test_r
                        low = test_r
                    else:
                        high = test_r
                
                if best_r > r:
                    circles[i, 2] = best_r
                    improved = True
        
        # If no improvement, try small position adjustments
        if not improved:
            for i in range(n):
                x, y, r = circles[i]
                
                # Try small movements
                best_x, best_y = x, y
                best_r = r
                best_improved = False
                
                # Test small movements
                for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                    for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                        test_x = max(0.01, min(0.99, x + dx))
                        test_y = max(0.01, min(0.99, y + dy))
                        
                        # Check validity
                        valid = True
                        test_r = r
                        
                        for j in range(n):
                            if i != j:
                                xj, yj, rj = circles[j]
                                dist = np.sqrt((test_x - xj)**2 + (test_y - yj)**2)
                                if dist < (test_r + rj + 1e-8):
                                    valid = False
                                    break
                        
                        if valid:
                            # Try to maximize radius at this new position
                            max_r_at_pos = min(test_x, 1-test_x, test_y, 1-test_y)
                            if max_r_at_pos > test_r:
                                test_r = max_r_at_pos
                                best_improved = True
                                best_x, best_y, best_r = test_x, test_y, test_r
                
                if best_improved:
                    circles[i] = [best_x, best_y, best_r]
                    improved = True
        
        if not improved:
            break
    
    return circles.tolist()

# EVOLVE-BLOCK-END
