# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import random
from scipy.spatial.distance import cdist

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    def validate_solution(circles):
        """Check if solution satisfies all constraints"""
        n = len(circles)
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
                return False
        
        # Check overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < r1 + r2:
                    return False
        return True
    
    def calculate_sum_radii(circles):
        """Calculate total sum of radii"""
        return np.sum(circles[:, 2])
    
    def voronoi_initialization(n_circles):
        """Initialize circles using Voronoi diagram approach"""
        # Generate random points
        points = np.random.rand(n_circles*3, 2)  # Generate extra points for better coverage
        
        # Create Voronoi diagram
        vor = Voronoi(points)
        
        # Select valid Voronoi vertices to place circles
        selected_vertices = []
        for i in range(min(n_circles, len(vor.vertices))):
            v = vor.vertices[i]
            if 0 <= v[0] <= 1 and 0 <= v[1] <= 1:
                selected_vertices.append(v)
                
        # If not enough valid vertices, add random points
        while len(selected_vertices) < n_circles:
            selected_vertices.append([np.random.rand(), np.random.rand()])
            
        # Take first n_circles vertices
        selected_vertices = selected_vertices[:n_circles]
        
        # Create circles with radius based on proximity to neighbors
        circles = []
        for i, (x, y) in enumerate(selected_vertices):
            # Calculate minimum distance to other points to determine max radius
            min_dist = float('inf')
            for j, (x2, y2) in enumerate(selected_vertices):
                if i != j:
                    d = np.sqrt((x-x2)**2 + (y-y2)**2)
                    min_dist = min(min_dist, d)
            
            # Set radius to half the minimum distance to neighbors, but bounded by unit square
            r = min(min_dist/2, x, 1-x, y, 1-y)
            r = max(r, 0.001)  # Minimum radius to avoid degenerate cases
            circles.append([x, y, r])
            
        return np.array(circles)
    
    def mutate_voronoi(circles, mutation_strength=0.05):
        """Create a mutated version of the Voronoi-based solution"""
        # Copy current circles
        new_circles = circles.copy()
        
        # Choose random circles to mutate
        n_mutations = max(1, int(len(circles) * 0.2))  # Mutate about 20% of circles
        indices = np.random.choice(len(circles), size=n_mutations, replace=False)
        
        for idx in indices:
            x, y, r = new_circles[idx]
            
            # Mutate position slightly
            x += np.random.normal(0, mutation_strength)
            y += np.random.normal(0, mutation_strength)
            
            # Bound position to unit square
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            # Mutate radius
            r *= (1 + np.random.normal(0, mutation_strength/2))
            r = max(0.001, r)
            r = min(r, x, 1-x, y, 1-y)  # Ensure containment
            
            new_circles[idx] = [x, y, r]
        
        return new_circles
    
    def optimize_circle_positions(circles, max_iter=100):
        """Fine-tune circle positions using local optimization"""
        # Convert to flattened parameter array for optimization
        params = []
        for i in range(len(circles)):
            params.extend([circles[i][0], circles[i][1], circles[i][2]])
        
        def objective(params_flat):
            # Reconstruct circles
            circles_new = []
            for i in range(0, len(params_flat), 3):
                x, y, r = params_flat[i:i+3]
                circles_new.append([x, y, r])
            
            # Calculate negative sum of radii (we want to maximize)
            return -calculate_sum_radii(circles_new)
        
        def constraint_func(params_flat):
            # Reconstruct circles
            circles_new = []
            for i in range(0, len(params_flat), 3):
                x, y, r = params_flat[i:i+3]
                circles_new.append([x, y, r])
            
            # Return constraint violations
            violations = []
            
            # Boundary constraints
            for i in range(len(circles_new)):
                x, y, r = circles_new[i]
                if x < r or x > 1-r or y < r or y > 1-r:
                    violations.append(1)
                else:
                    violations.append(0)
            
            # Overlap constraints  
            for i in range(len(circles_new)):
                for j in range(i+1, len(circles_new)):
                    x1, y1, r1 = circles_new[i]
                    x2, y2, r2 = circles_new[j]
                    dist_sq = (x1-x2)**2 + (y1-y2)**2
                    min_dist_sq = (r1+r2)**2
                    if dist_sq < min_dist_sq:
                        violations.append(1)
                    else:
                        violations.append(0)
            
            return np.sum(violations)
        
        # Try to optimize
        try:
            result = minimize(objective, params, method='L-BFGS-B', 
                            bounds=[(r, 1-r) for _ in range(len(params)//3) for r in [0.001]*3],
                            options={'maxiter': max_iter})
            if result.success:
                # Reconstruct circles from optimized parameters
                circles_optimized = []
                for i in range(0, len(result.x), 3):
                    x, y, r = result.x[i:i+3]
                    circles_optimized.append([x, y, r])
                return np.array(circles_optimized)
        except:
            pass
        
        return circles
    
    def evaluate_fitness(candidate):
        """Evaluate fitness of a candidate solution"""
        coords = candidate.reshape(-1, 3)
        total_radius = np.sum(coords[:, 2])
        
        # Calculate penalty for constraint violations
        penalty = 0
        
        # Boundary penalties
        for i in range(len(coords)):
            x, y, r = coords[i]
            if r > x or r > y or r > 1-x or r > 1-y:
                penalty += 1000 * (r - x)**2 + 1000 * (r - y)**2 + 1000 * (r - (1-x))**2 + 1000 * (r - (1-y))**2
        
        # Collision penalties
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                x1, y1, r1 = coords[i]
                x2, y2, r2 = coords[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                if dist_sq < min_dist_sq:
                    overlap = min_dist_sq - dist_sq
                    penalty += 10000 * overlap
                    
        return -(total_radius - penalty)  # Negative because we minimize in scipy
    
    # Main algorithm
    best_circles = None
    best_sum_radii = 0
    
    # Try multiple initializations
    for attempt in range(10):
        # Start with Voronoi-based initialization
        circles = voronoi_initialization(32)
        
        # Local optimization
        circles = optimize_circle_positions(circles)
        
        # Validate
        if validate_solution(circles):
            sum_radii = calculate_sum_radii(circles)
            if sum_radii > best_sum_radii:
                best_sum_radii = sum_radii
                best_circles = circles.copy()
        
        # Evolutionary improvement
        for gen in range(50):
            # Create new candidate via mutation
            mutated = mutate_voronoi(circles, 0.02)
            
            # Local optimization on mutated solution
            mutated = optimize_circle_positions(mutated)
            
            # Validate and accept if better
            if validate_solution(mutated):
                sum_radii = calculate_sum_radii(mutated)
                if sum_radii > best_sum_radii:
                    best_sum_radii = sum_radii
                    best_circles = mutated.copy()
                    circles = mutated.copy()
    
    # If we still don't have a good solution, fall back to a simple approach
    if best_circles is None:
        # Simple greedy initialization
        circles = np.zeros((32, 3))
        # Place circles in a grid-like pattern with decreasing sizes
        placed = 0
        for i in range(6):
            for j in range(6):
                if placed >= 32:
                    break
                x = 0.1 + i * 0.15
                y = 0.1 + j * 0.15
                r = 0.05
                circles[placed] = [x, y, r]
                placed += 1
            if placed >= 32:
                break
        best_circles = circles
    
    return best_circles

# EVOLVE-BLOCK-END
