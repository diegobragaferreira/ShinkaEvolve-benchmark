# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time
import random
from typing import Tuple, List
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances
        distances = pdist(points)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def compute_voronoi_uniformity_score(points) -> float:
        """Compute a score based on Voronoi cell uniformity to guide optimization."""
        try:
            vor = Voronoi(points)
            
            # Calculate areas of finite Voronoi regions
            areas = []
            for region in vor.regions:
                if not any(v == -1 for v in region):  # Skip infinite regions
                    polygon = [vor.vertices[i] for i in region]
                    if len(polygon) >= 3:
                        # Calculate polygon area using shoelace formula
                        area = 0.5 * abs(sum(polygon[i][0] * polygon[(i+1)%len(polygon)][1] - 
                                           polygon[(i+1)%len(polygon)][0] * polygon[i][1] 
                                           for i in range(len(polygon))))
                        areas.append(area)
            
            if not areas:
                return 0.0
                
            # Return inverse of coefficient of variation (higher is better)
            mean_area = np.mean(areas)
            if mean_area == 0:
                return 0.0
                
            std_area = np.std(areas)
            cv = std_area / mean_area if mean_area > 0 else 0.0
            return 1.0 / (1.0 + cv) if cv < 10 else 0.0  # Prevent extreme values
        except:
            return 0.0

    def compute_combined_objective(points) -> float:
        """Compute combined objective function incorporating ratio and uniformity."""
        ratio = compute_min_max_ratio(points)
        uniformity = compute_voronoi_uniformity_score(points)
        # Weighted sum: prioritize ratio but penalize non-uniform distributions
        return ratio * (1.0 + 0.5 * uniformity)

    def compute_voronoi_cell_areas(points) -> np.ndarray:
        """Extract Voronoi cell areas for analysis."""
        try:
            vor = Voronoi(points)
            areas = []
            for region in vor.regions:
                if not any(v == -1 for v in region):  # Skip infinite regions
                    polygon = [vor.vertices[i] for i in region]
                    if len(polygon) >= 3:
                        area = 0.5 * abs(sum(polygon[i][0] * polygon[(i+1)%len(polygon)][1] - 
                                           polygon[(i+1)%len(polygon)][0] * polygon[i][1] 
                                           for i in range(len(polygon))))
                        areas.append(area)
            return np.array(areas)
        except:
            return np.array([])

    def create_voronoi_based_initial_points() -> np.ndarray:
        """Create initial point configuration based on Voronoi principles."""
        # Strategy: Create points that would form approximately equilateral triangles or hexagonal packing
        points = []
        
        # Generate a hexagonal pattern (natural for uniform Voronoi cells)
        spacing = 0.25
        height = spacing * np.sqrt(3) / 2
        
        # Create a 4x4 grid with hexagonal offset
        for i in range(4):
            for j in range(4):
                if len(points) < 16:
                    x = j * spacing + (i % 2) * spacing / 2
                    y = i * height
                    
                    # Add subtle perturbations to break perfect symmetry
                    x += np.random.normal(0, 0.005)
                    y += np.random.normal(0, 0.005)
                    
                    points.append([x, y])
        
        points = np.array(points)
        # Ensure we have exactly 16 points and stay within bounds
        points = points[:16]
        points = np.clip(points, 0, 1)
        
        # Add more specific Voronoi-aware perturbations
        for i in range(16):
            # Slight asymmetric perturbations based on position
            # This helps break rotational symmetry that often leads to suboptimal solutions
            asym_x = np.sin(i * 1.3) * 0.003
            asym_y = np.cos(i * 1.7) * 0.003
            points[i, 0] += asym_x
            points[i, 1] += asym_y
            
        points = np.clip(points, 0, 1)
        return points

    def create_fibonacci_spiral_points() -> np.ndarray:
        """Create points using Fibonacci spiral for good spatial distribution."""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Generate 16 points along a Fibonacci spiral
        for i in range(16):
            # Position along spiral
            z = 1 - (i / 15.0) * 2
            radius = np.sqrt(1 - z*z)
            theta = np.arccos(z)
            phi = (i * golden_ratio) % (2 * np.pi)
            
            # Convert to Cartesian on unit sphere
            x = radius * np.cos(phi)
            y = radius * np.sin(phi)
            
            # Project to [0,1]x[0,1]
            x_norm = (x + 1) / 2
            y_norm = (y + 1) / 2
            
            points.append([x_norm, y_norm])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points

    def create_triangular_lattice_points() -> np.ndarray:
        """Create points in a triangular lattice pattern."""
        points = []
        spacing_x = 0.25
        spacing_y = 0.25 * np.sqrt(3) / 2
        
        # Create triangular lattice
        for i in range(4):
            for j in range(4):
                if len(points) < 16:
                    x = j * spacing_x + (i % 2) * spacing_x / 2
                    y = i * spacing_y
                    
                    # Add perturbations
                    x += np.random.normal(0, 0.005)
                    y += np.random.normal(0, 0.005)
                    
                    points.append([x, y])
        
        points = np.array(points[:16])
        points = np.clip(points, 0, 1)
        return points

    def generate_diverse_initial_configs() -> List[np.ndarray]:
        """Generate multiple diverse initial configurations."""
        configs = []
        
        # Config 1: Voronoi-based hexagonal
        configs.append(create_voronoi_based_initial_points())
        
        # Config 2: Fibonacci spiral
        configs.append(create_fibonacci_spiral_points())
        
        # Config 3: Triangular lattice
        configs.append(create_triangular_lattice_points())
        
        # Config 4: Random with Voronoi awareness
        random_points = np.random.rand(16, 2)
        configs.append(random_points)
        
        # Config 5: Slightly perturbed hexagonal
        base_hex = create_voronoi_based_initial_points()
        perturbed = base_hex + np.random.normal(0, 0.01, base_hex.shape)
        configs.append(np.clip(perturbed, 0, 1))
        
        # Config 6: Checkerboard with Voronoi sensitivity
        checkerboard_points = []
        for i in range(4):
            for j in range(4):
                if len(checkerboard_points) < 16:
                    x = j * 0.25 + (i % 2) * 0.125
                    y = i * 0.25
                    
                    # Add Voronoi-sensitive asymmetry
                    asym_x = np.sin(i * 0.8) * 0.004
                    asym_y = np.cos(j * 0.7) * 0.004
                    
                    x += asym_x + np.random.normal(0, 0.003)
                    y += asym_y + np.random.normal(0, 0.003)
                    
                    checkerboard_points.append([x, y])
        
        configs.append(np.clip(np.array(checkerboard_points), 0, 1))
        
        return configs

    def voronoi_evolutionary_optimization(max_generations: int = 300) -> np.ndarray:
        """Evolutionary optimization focused on Voronoi structure."""
        # Initial population of point configurations
        population = generate_diverse_initial_configs()
        best_solution = None
        best_fitness = -np.inf
        
        # Evolution parameters
        elite_size = 2
        mutation_rate = 0.1
        
        # Store history for adaptive strategies  
        recent_improvements = []
        
        for generation in range(max_generations):
            # Evaluate fitness of population
            fitness_scores = []
            for points in population:
                # Ensure points stay within bounds
                points = np.clip(points, 0, 1)
                fitness = compute_combined_objective(points)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = points.copy()
            
            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitness = [fitness_scores[i] for i in sorted_indices]
            
            # Track recent improvements for adaptive cooling
            if len(recent_improvements) >= 10:
                recent_improvements.pop(0)
            recent_improvements.append(sorted_fitness[0] - (recent_improvements[-1] if recent_improvements else 0))
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            new_population.extend(sorted_population[:elite_size])
            
            # Generate offspring through crossover and mutation
            while len(new_population) < len(population):
                # Tournament selection
                tournament_size = 4
                tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                # Select second parent
                tournament_indices2 = np.random.choice(len(population), tournament_size, replace=False)
                tournament_fitness2 = [fitness_scores[i] for i in tournament_indices2]
                winner_idx2 = tournament_indices2[np.argmax(tournament_fitness2)]
                
                # Crossover: blend parent configurations 
                parent1, parent2 = sorted_population[winner_idx], sorted_population[winner_idx2]
                
                # Uniform crossover
                mask = np.random.rand(16, 2) > 0.5
                child = np.where(mask, parent1, parent2).copy()
                
                # Mutation
                if np.random.rand() < mutation_rate:
                    # Apply localized mutations to maintain Voronoi structure awareness
                    mutation_count = max(1, 16 // 4)  # Mutate about 25% of points
                    mutation_indices = np.random.choice(16, mutation_count, replace=False)
                    
                    for idx in mutation_indices:
                        # Add more sophisticated mutation that considers Voronoi effects
                        # Larger mutations in early generations, smaller later
                        mutation_strength = 0.02 * (1.0 - generation / max_generations)
                        child[idx] += np.random.normal(0, mutation_strength, 2)
                
                # Ensure bounds
                child = np.clip(child, 0, 1)
                new_population.append(child)
            
            population = new_population[:len(population)]
            
            # Adaptive parameters based on progress
            if generation > 100:
                # Gradually decrease mutation rate as optimization progresses
                mutation_rate = max(0.02, 0.1 * (1.0 - generation / max_generations))
            
            # Early stopping criteria
            if generation > 50 and len(recent_improvements) >= 10:
                recent_avg = np.mean(recent_improvements[-5:])
                if recent_avg < 1e-6:
                    break
        
        return best_solution if best_solution is not None else population[0]

    def adaptive_local_refinement(initial_points: np.ndarray, max_iterations: int = 300) -> np.ndarray:
        """Apply adaptive local refinement to enhance the solution."""
        points = initial_points.copy()
        current_fitness = compute_combined_objective(points)
        
        best_points = points.copy()
        best_fitness = current_fitness
        
        # Adaptive step sizes and cooling schedules
        step_size = 0.02
        min_step = 0.001
        cooling_factor = 0.995
        
        # Track improvement for adaptive cooling
        patience_counter = 0
        max_patience = 50
        
        for iteration in range(max_iterations):
            # Save current state
            original_points = points.copy()
            original_fitness = current_fitness
            
            # Try various perturbation strategies
            perturbation_strategy = np.random.choice(['small', 'medium', 'large'])
            
            if perturbation_strategy == 'small':
                # Small perturbations
                perturbation_magnitude = step_size * 0.5
                indices_to_move = np.random.choice(16, size=max(1, 16 // 8), replace=False)
            elif perturbation_strategy == 'medium':
                # Medium perturbations
                perturbation_magnitude = step_size
                indices_to_move = np.random.choice(16, size=max(1, 16 // 4), replace=False)
            else:  # large
                # Large perturbations for exploration
                perturbation_magnitude = step_size * 2
                indices_to_move = np.random.choice(16, size=max(1, 16 // 3), replace=False)
            
            # Apply perturbations
            for idx in indices_to_move:
                points[idx] += np.random.normal(0, perturbation_magnitude, 2)
            
            # Keep within bounds
            points = np.clip(points, 0, 1)
            
            # Evaluate new solution
            new_fitness = compute_combined_objective(points)
            
            # Accept or reject based on fitness
            if new_fitness > current_fitness:
                current_fitness = new_fitness
                
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_points = points.copy()
                    patience_counter = 0  # Reset patience
                else:
                    patience_counter += 1
            else:
                # Occasionally accept worse solutions to escape local optima
                if np.random.rand() < 0.05:
                    current_fitness = new_fitness
                else:
                    points = original_points.copy()  # Revert
            
            # Adaptive step size adjustment
            step_size = max(min_step, step_size * cooling_factor)
            
            # Early stopping if no improvement for a while
            if patience_counter > max_patience:
                break
                
        return best_points

    def gradient_free_optimization(points: np.ndarray, max_iter: int = 200) -> np.ndarray:
        """Use gradient-free optimization approach for final refinement."""
        try:
            # Use a combination of Nelder-Mead-like approach and random search
            current_points = points.copy()
            current_fitness = compute_combined_objective(current_points)
            
            best_points = current_points.copy()
            best_fitness = current_fitness
            
            # Iterative improvement using a mix of strategies
            for iter_num in range(max_iter):
                # Strategy 1: Individual point perturbations
                for i in range(len(current_points)):
                    # Try small perturbation
                    original_point = current_points[i].copy()
                    perturbation = np.random.normal(0, 0.005, 2)
                    current_points[i] = original_point + perturbation
                    current_points[i] = np.clip(current_points[i], 0, 1)
                    
                    new_fitness = compute_combined_objective(current_points)
                    
                    if new_fitness <= current_fitness:
                        # Revert if worse
                        current_points[i] = original_point
                    else:
                        # Accept improvement
                        current_fitness = new_fitness
                        
                        if new_fitness > best_fitness:
                            best_fitness = new_fitness
                            best_points = current_points.copy()
                
                # Strategy 2: Group perturbations
                if np.random.rand() < 0.3:
                    # Select random subset of points to move together
                    group_indices = np.random.choice(len(current_points), 
                                                   size=max(2, len(current_points) // 4), 
                                                   replace=False)
                    
                    # Calculate centroid of group
                    group_centroid = np.mean(current_points[group_indices], axis=0)
                    
                    # Move entire group
                    centroid_shift = np.random.normal(0, 0.01, 2)
                    for idx in group_indices:
                        current_points[idx] = current_points[idx] + centroid_shift
                    current_points = np.clip(current_points, 0, 1)
                    
                    new_fitness = compute_combined_objective(current_points)
                    
                    if new_fitness <= current_fitness:
                        # Revert group movement
                        for idx in group_indices:
                            current_points[idx] = current_points[idx] - centroid_shift
                        current_points = np.clip(current_points, 0, 1)
                    else:
                        current_fitness = new_fitness
                        
                        if new_fitness > best_fitness:
                            best_fitness = new_fitness
                            best_points = current_points.copy()
                
                # Occasionally explore with larger steps
                if iter_num % 20 == 0 and iter_num > 0:
                    # Random exploration
                    for i in range(len(current_points)):
                        if np.random.rand() < 0.5:
                            current_points[i] = np.random.rand(2)
                            
                    current_points = np.clip(current_points, 0, 1)
                    new_fitness = compute_combined_objective(current_points)
                    
                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        best_points = current_points.copy()
            
            return best_points
            
        except Exception as e:
            # If optimization fails, return the input points
            warnings.warn(f"Gradient-free optimization failed: {e}")
            return points

    # Main optimization pipeline
    np.random.seed(42)
    
    # Phase 1: Evolutionary optimization focused on Voronoi structure
    print("Starting Voronoi-based evolutionary optimization...")
    evolutionary_result = voronoi_evolutionary_optimization(max_generations=300)
    
    # Phase 2: Adaptive local refinement to fine-tune solution
    print("Performing adaptive local refinement...")
    refined_result = adaptive_local_refinement(evolutionary_result, max_iterations=300)
    
    # Phase 3: Final gradient-free optimization for polishing
    print("Applying final gradient-free refinement...")
    final_result = gradient_free_optimization(refined_result, max_iter=200)
    
    # Final validation
    final_fitness = compute_combined_objective(final_result)
    print(f"Final combined fitness: {final_fitness:.8f}")
    
    return final_result

# EVOLVE-BLOCK-END