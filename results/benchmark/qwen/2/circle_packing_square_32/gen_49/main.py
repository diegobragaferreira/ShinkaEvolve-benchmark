# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class CirclePackingOptimizer:
    def __init__(self, n_circles=32, max_iterations=500):
        self.n_circles = n_circles
        self.max_iterations = max_iterations
        self.seed = 42
        np.random.seed(self.seed)
        
    def initialize_grid(self):
        """Initialize circles in a hexagonal grid pattern with adaptive density adjustment"""
        # Create hexagonal grid
        rows = int(np.ceil(np.sqrt(self.n_circles)))
        cols = int(np.ceil(self.n_circles / rows))
        
        # Calculate spacing with padding
        padding = 0.05
        spacing_x = (1 - 2*padding) / cols
        spacing_y = (1 - 2*padding) / rows
        
        # Adjust for hexagonal arrangement
        hex_spacing_x = spacing_x
        hex_spacing_y = spacing_y * np.sqrt(3)/2
        
        circles = []
        count = 0
        
        # Fill grid with hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if count >= self.n_circles:
                    break
                    
                # Hexagonal offset
                x_offset = (j if i % 2 == 0 else j + 0.5) * hex_spacing_x + padding
                y_offset = i * hex_spacing_y + padding
                
                # Add slight randomness
                x = max(padding, min(1-padding, x_offset + np.random.normal(0, 0.005*hex_spacing_x)))
                y = max(padding, min(1-padding, y_offset + np.random.normal(0, 0.005*hex_spacing_y)))
                
                # Initialize with adaptive radius based on spacing
                radius = min(hex_spacing_x, hex_spacing_y) * 0.3
                
                circles.append([x, y, radius])
                count += 1
                
            if count >= self.n_circles:
                break
        
        # Ensure we have exactly n_circles
        while len(circles) < self.n_circles:
            x = np.random.uniform(padding, 1-padding)
            y = np.random.uniform(padding, 1-padding)
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles.append([x, y, r])
            
        return np.array(circles[:self.n_circles])
    
    def validate_placement(self, circles):
        """Validate that all circles are within bounds and don't overlap"""
        if len(circles) == 0:
            return False
            
        # Check boundary constraints
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlap constraints
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < r1 + r2:
                    return False
                    
        return True
    
    def calculate_violation_penalty(self, circles):
        """Calculate penalty based on constraint violations"""
        penalty = 0.0
        
        # Boundary penalties - exponential penalty based on violation amount
        for x, y, r in circles:
            # Penalties for going outside boundaries
            if x - r < 0:
                penalty += 1000 * np.exp(20 * (x - r))
            if x + r > 1:
                penalty += 1000 * np.exp(20 * (x + r - 1))
            if y - r < 0:
                penalty += 1000 * np.exp(20 * (y - r))
            if y + r > 1:
                penalty += 1000 * np.exp(20 * (y + r - 1))
        
        # Overlap penalties - exponential penalty based on violation amount
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < r1 + r2:
                    penalty += 1000 * np.exp(20 * (distance - (r1 + r2)))
                    
        return penalty
    
    def objective_function(self, flat_params):
        """Objective function to maximize sum of radii with penalty for violations"""
        # Reshape flat parameters back to circles array
        circles = flat_params.reshape(-1, 3)
        
        # We want to maximize sum of radii, so return negative sum plus penalties
        radius_sum = np.sum(circles[:, 2])
        penalty = self.calculate_violation_penalty(circles)
        
        return -radius_sum + penalty
    
    def optimize(self, initial_circles):
        """Perform optimization using L-BFGS-B"""
        # Flatten initial circles for optimization
        initial_params = initial_circles.flatten()
        
        # Define bounds for optimization (x,y in [0.05, 0.95], r in [0.01, 0.4])
        bounds = []
        for _ in range(self.n_circles):
            bounds.extend([(0.05, 0.95), (0.05, 0.95), (0.01, 0.4)])
        
        try:
            # Optimize using L-BFGS-B
            result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.max_iterations, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            # Extract optimized parameters
            optimized_params = result.x
            optimized_circles = optimized_params.reshape(-1, 3)
            
            # Validate the result
            if self.validate_placement(optimized_circles):
                return optimized_circles
            else:
                # Try to fix by adjusting radii if optimization didn't yield valid solution
                return self.fix_invalid_configuration(optimized_circles)
                
        except Exception as e:
            # If optimization fails, return the initial configuration
            print(f"Optimization failed: {e}")
            return initial_circles
    
    def fix_invalid_configuration(self, circles):
        """Attempt to make configuration valid by reducing radii"""
        # Make copy to avoid modifying original
        fixed_circles = circles.copy()
        
        # Reduce all radii gradually
        reduction_factor = 0.95
        for _ in range(100):  # Max attempts
            fixed_circles[:, 2] *= reduction_factor
            if self.validate_placement(fixed_circles):
                return fixed_circles
                
        # Fallback to original grid if unable to fix
        return self.initialize_grid()
    
    def run(self):
        """Main execution method"""
        # Step 1: Initialize
        circles = self.initialize_grid()
        
        # Step 2: Optimize
        optimized_circles = self.optimize(circles)
        
        # Step 3: Final validation and correction
        if not self.validate_placement(optimized_circles):
            optimized_circles = self.initialize_grid()
        
        # Final correction to ensure all constraints
        for i in range(len(optimized_circles)):
            x, y, r = optimized_circles[i]
            # Constrain positions to valid range
            optimized_circles[i][0] = max(r, min(1-r, x))
            optimized_circles[i][1] = max(r, min(1-r, y))
            # Constrain radii to valid range
            optimized_circles[i][2] = max(0.01, min(0.4, r))
            
        return optimized_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer(n_circles=32)
    return optimizer.run()

# EVOLVE-BLOCK-END
