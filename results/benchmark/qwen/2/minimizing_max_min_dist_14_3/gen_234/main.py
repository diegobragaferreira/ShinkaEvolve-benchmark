# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import cdist
from numba import jit
import time
import random
from copy import deepcopy
from scipy.optimize import minimize

@jit(nopython=True)
def compute_min_max_ratio_numba(points):
    """Optimized distance computation using numba"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0

    for i in range(n):
        for j in range(i+1, n):
            # Compute squared distance to avoid sqrt computation
            dist_sq = (points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2 + (points[i,2]-points[j,2])**2
            dist = np.sqrt(dist_sq)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

    return min_dist, max_dist

def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n points distributed as evenly as possible on a unit sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def compute_min_max_ratio(points: np.ndarray) -> tuple:
    """Compute the minimum and maximum distances between all pairs of points, and return their ratio."""
    if len(points) < 2:
        return 0.0, 0.0, 0.0

    # Use numba-optimized version
    min_distance, max_distance = compute_min_max_ratio_numba(points)

    # Avoid division by zero
    if max_distance == 0:
        ratio = 0.0
    else:
        ratio = min_distance / max_distance

    return min_distance, max_distance, ratio

def project_to_unit_sphere(points):
    """Project points to the unit sphere"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Handle case where norm might be zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def voronoi_entropy_score(points):
    """
    Calculate entropy-based score of Voronoi cell distribution.
    High entropy indicates more uniform cell distribution.
    """
    try:
        sv = SphericalVoronoi(points)
        areas = sv.calculate_areas()
        # Normalize areas
        areas = areas / np.sum(areas)
        # Entropy calculation
        entropy = -np.sum(areas * np.log(areas + 1e-10))
        return entropy
    except:
        return 0.0

def generate_voronoi_fitness(points):
    """Calculate fitness based on Voronoi diagram properties"""
    try:
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(points, radius=1.0)

        # Calculate Voronoi cell areas
        cell_areas = sv.calculate_areas()

        # Fitness: penalize large variance in cell areas (more uniform distribution)
        area_variance = np.var(cell_areas)

        # Also consider distance distribution
        min_dist, max_dist, _ = compute_min_max_ratio(points)
        if max_dist == 0:
            distance_ratio = 0.0
        else:
            distance_ratio = min_dist / max_dist

        # Combined fitness: balance uniformity and distance ratio
        # Lower area variance + higher distance ratio = better fitness
        fitness = distance_ratio - 0.1 * area_variance

        return fitness, distance_ratio
    except:
        # Fallback to simple distance ratio if Voronoi fails
        min_dist, max_dist, ratio = compute_min_max_ratio(points)
        return ratio, ratio

def create_individual(num_points=14):
    """Create a random individual (point configuration) on unit sphere"""
    points = np.random.randn(num_points, 3)
    points = project_to_unit_sphere(points)
    return points

def adaptive_crossover(parent1, parent2, crossover_rate=0.8):
    """Adaptive crossover that considers point proximity and distribution"""
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    # Analyze point distributions to make smarter crossover decisions
    distances1 = cdist(parent1, parent1)
    distances2 = cdist(parent2, parent2)

    # Compute average distances for each individual
    avg_dist1 = np.mean(distances1[distances1 > 0]) if np.any(distances1 > 0) else 1.0
    avg_dist2 = np.mean(distances2[distances2 > 0]) if np.any(distances2 > 0) else 1.0

    # Choose crossover strategy based on distribution characteristics
    if avg_dist1 < avg_dist2 * 0.8:  # parent1 is more clustered
        # Prefer more spread out parent for first half
        crossover_point = random.randint(int(len(parent1) * 0.3), int(len(parent1) * 0.7))
        child1 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
        child2 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
    elif avg_dist2 < avg_dist1 * 0.8:  # parent2 is more clustered
        # Prefer more spread out parent for first half
        crossover_point = random.randint(int(len(parent1) * 0.3), int(len(parent1) * 0.7))
        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
    else:
        # Standard crossover
        crossover_point = random.randint(1, len(parent1)-1)
        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])

    # Project back to unit sphere
    child1 = project_to_unit_sphere(child1)
    child2 = project_to_unit_sphere(child2)

    return child1, child2

def intelligent_mutation(individual, mutation_rate=0.1, base_strength=0.05):
    """Mutation operator that intelligently targets under-separated regions"""
    mutated = individual.copy()

    # Analyze current distribution
    distances = cdist(mutated, mutated)
    np.fill_diagonal(distances, np.inf)

    # Find points that are too close to others (potential bottlenecks)
    min_distances = np.min(distances, axis=1)
    avg_min_dist = np.mean(min_distances)

    # Identify under-separated points (those with below-average minimum distance)
    under_separated_mask = min_distances < avg_min_dist * 0.7

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Increase mutation strength for under-separated points
            strength = base_strength
            if under_separated_mask[i]:
                strength *= 2.0  # Double strength for clustered points

            # Add small random perturbation
            delta = np.random.normal(0, strength, 3)
            mutated[i] += delta

            # Project back to unit sphere
            mutated[i] = project_to_unit_sphere(mutated[i].reshape(1, 3)).flatten()

    return mutated

def diversity_preservation(population, fitness_scores, population_size):
    """Maintain diversity by checking Voronoi uniformity and reintroducing variety"""
    # Calculate Voronoi uniformity scores for population
    uniformity_scores = []
    for individual in population:
        try:
            sv = SphericalVoronoi(individual)
            areas = sv.calculate_areas()
            # Standard deviation of areas as measure of uniformity
            uniformity = np.std(areas) if len(areas) > 1 else 0.0
            uniformity_scores.append(uniformity)
        except:
            uniformity_scores.append(np.inf)

    # If population has low diversity (high uniformity variance), add fresh individuals
    if len(uniformity_scores) > 1:
        std_uniformity = np.std(uniformity_scores)
        mean_uniformity = np.mean(uniformity_scores)

        # If diversity is low, inject some random individuals
        if std_uniformity < 0.1 * mean_uniformity and mean_uniformity > 0:
            # Replace worst performers with random individuals
            worst_indices = np.argsort(fitness_scores)[:len(population)//4]
            for idx in worst_indices:
                if random.random() < 0.3:  # 30% chance to replace
                    population[idx] = create_individual(14)

    return population

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select individuals using tournament selection"""
    selected_indices = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected_indices.append(winner_index)

    return [population[i] for i in selected_indices]

def spherical_perturb(points: np.ndarray, target_point_idx: int, temperature: float) -> np.ndarray:
    """
    Apply perturbation on the tangent plane of the unit sphere at target point,
    ensuring resulting point stays on unit sphere.
    """
    # Create a copy of the points
    new_points = points.copy()
    
    # Get the target point
    target_point = points[target_point_idx]
    
    # Generate random perturbation in tangent plane
    # We generate a random vector and subtract its projection onto the normal
    perturbation = np.random.normal(0, 0.01 * temperature, 3)
    
    # Project the perturbation onto the tangent plane (orthogonal to the normal)
    # Normal is just the point itself on unit sphere
    normal = target_point
    proj = np.dot(perturbation, normal)
    tangent_perturbation = perturbation - proj * normal
    
    # Apply the perturbation
    new_points[target_point_idx] = target_point + tangent_perturbation
    
    # Project back to unit sphere
    new_points = project_to_unit_sphere(new_points)
    
    return new_points

def adaptive_perturbation_strategy(points: np.ndarray, current_ratio: float, temperature: float) -> np.ndarray:
    """
    Apply adaptive perturbation based on current configuration analysis.
    """
    # Analyze current distribution to decide which point to perturb
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    
    # Get minimum and maximum distances for analysis
    min_distances = np.min(distances, axis=1)
    max_distances = np.max(distances, axis=1)
    
    # Calculate average distance per point for reference
    avg_distances = np.mean(distances, axis=1)
    
    # Prefer perturbing points that:
    # 1. Are among the closest points (to possibly increase minimum)
    # 2. Are among the farthest points (to possibly decrease maximum)  
    # 3. Or are in intermediate positions
    
    # Score points based on their potential impact
    scores = np.zeros(len(points))
    for i in range(len(points)):
        # Weight by how close they are to min vs max distances
        min_dist = min_distances[i]
        max_dist = max_distances[i]
        avg_dist = avg_distances[i]
        
        # Score based on being too close (helps increase min) or too far (helps decrease max)
        if min_dist < avg_dist * 0.5:  # Very close - prioritize increasing their distance
            scores[i] = -min_dist
        elif max_dist > avg_dist * 2.0:  # Very far - prioritize decreasing their distance
            scores[i] = max_dist
        else:  # Medium distance - less critical
            scores[i] = 0
    
    # Choose point to perturb based on scores (higher score means more important)
    if np.sum(np.abs(scores)) > 0:
        # Use weighted probability based on scores
        probs = np.abs(scores)
        probs = probs / np.sum(probs)
        target_idx = np.random.choice(len(points), p=probs)
    else:
        # Fallback to random selection
        target_idx = np.random.randint(len(points))
    
    return spherical_perturb(points, target_idx, temperature)

def adaptive_cooling(initial_temp, iteration, max_iterations, ratio_history):
    """
    Adaptive cooling schedule that adjusts based on convergence
    """
    # Base cooling rate
    base_cooling = 0.9995

    # Check recent convergence
    if len(ratio_history) > 10:
        recent_improvement = ratio_history[-1] - ratio_history[-10]
        if recent_improvement < 1e-8:
            # Slow improvement, cool faster
            return base_cooling * 1.05
        elif recent_improvement > 1e-6:
            # Fast improvement, cool slower
            return base_cooling * 0.95

    return base_cooling

def initialize_icosahedron_points(n):
    """Initialize points based on regular icosahedron vertices for better starting configuration."""
    # Regular icosahedron vertices (normalized)
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [-1, 0, phi], [1, 0, phi], [-1, 0, -phi], [1, 0, -phi],
        [0, phi, 1], [0, phi, -1], [0, -phi, 1], [0, -phi, -1],
        [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]
    ])
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices[0])

    # For 14 points, we'll use 12 vertices plus 2 additional points
    if n == 14:
        # Get 12 vertices and add 2 points from edge midpoints
        # We'll use a simple method to add 2 more points
        additional_points = []
        # Add two points that are well distributed
        additional_points.append([0, 0, 1])  # North pole
        additional_points.append([0, 0, -1])  # South pole
        return np.vstack([vertices, additional_points])
    elif n == 12:
        return vertices
    else:
        # Fall back to fibonacci for other sizes
        return fibonacci_sphere(n)

def generate_initial_points(num_strategies: int = 5) -> list:
    """Generate multiple initialization strategies"""
    initial_points_list = []
    
    # Strategy 1: Fibonacci sphere
    points1 = fibonacci_sphere(14)
    initial_points_list.append(points1)
    
    # Strategy 2: Icosahedron-based
    try:
        points2 = initialize_icosahedron_points(14)
        initial_points_list.append(points2)
    except:
        pass

    # Strategy 3: Random points on sphere
    np.random.seed(123)
    points3 = np.random.randn(14, 3)
    points3 = project_to_unit_sphere(points3)
    initial_points_list.append(points3)
    
    # Strategy 4: Perturbed Fibonacci
    np.random.seed(456)
    points4 = fibonacci_sphere(14)
    noise = np.random.normal(0, 0.03, points4.shape)
    points4 = points4 + noise
    points4 = project_to_unit_sphere(points4)
    initial_points_list.append(points4)
    
    # Strategy 5: Another random distribution
    np.random.seed(789)
    points5 = np.random.randn(14, 3)
    points5 = project_to_unit_sphere(points5)
    initial_points_list.append(points5)
    
    return initial_points_list

def multi_start_optimization(initial_points_list, max_generations=1500):
    """Run optimization from multiple starting points and return the best result"""
    best_points = None
    best_fitness = -np.inf

    for i, initial_points in enumerate(initial_points_list):
        print(f"Starting optimization run {i+1}/{len(initial_points_list)}")
        # Apply multi-phase evolutionary optimization from this starting point
        optimized_points, fitness = multi_phase_evolutionary_optimize(initial_points, max_generations=max_generations)

        if fitness > best_fitness:
            best_fitness = fitness
            best_points = optimized_points.copy()

    return best_points, best_fitness

def multi_phase_evolutionary_optimize(initial_points, max_generations=1500, population_size=20):
    """
    Multi-phase evolutionary optimization with adaptive strategies.
    """
    # Track optimization history for diversity management and novelty calculation
    history = []
    velocity_history = []

    # Phase 1: Diverse initialization
    population = [initial_points.copy()]
    for i in range(population_size - 1):
        # Create diverse starting configurations
        individual = adaptive_perturbation_strategy(initial_points, 0.0, 0.1)
        population.append(individual)

    best_individual = None
    best_fitness = -np.inf

    # Phase 2: Adaptive evolutionary optimization
    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness, _ = generate_voronoi_fitness(individual)
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()

        # Update history for novelty calculations
        if len(history) < 10:  # Keep only recent configurations
            history.append(best_individual.copy())
        else:
            history.pop(0)
            history.append(best_individual.copy())

        # Adaptive selection pressure based on convergence
        normalized_fitness = np.array(fitness_scores) - np.min(fitness_scores) + 1e-10
        if np.sum(normalized_fitness) > 0:
            probabilities = normalized_fitness / np.sum(normalized_fitness)
        else:
            probabilities = np.ones(len(population)) / len(population)

        # Selection with adaptive diversity consideration
        selected_population = []
        for _ in range(population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size, p=probabilities)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected_population.append(population[winner_index].copy())

        # Crossover and mutation with adaptive strategies
        new_population = []

        # Calculate population diversity
        diversity_score = 0
        if len(population) >= 2:
            # Sample a subset for diversity calculation
            sample_indices = np.random.choice(len(population), min(5, len(population)), replace=False)
            sample_pop = [population[i] for i in sample_indices]
            pairwise_distances = []
            for i in range(len(sample_pop)):
                for j in range(i+1, len(sample_pop)):
                    dist = np.linalg.norm(sample_pop[i].flatten() - sample_pop[j].flatten())
                    pairwise_distances.append(dist)
            if pairwise_distances:
                diversity_score = np.mean(pairwise_distances)

        # Determine phase based on generation for adaptive strategies
        if generation < max_generations * 0.3:  # Exploration phase
            phase = "exploration"
        elif generation < max_generations * 0.7:  # Exploitation phase  
            phase = "exploitation"
        else:  # Refinement phase
            phase = "refinement"

        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[min(i+1, len(selected_population)-1)]

            # Adaptive crossover
            if phase == "exploration":
                alpha = 0.2 + 0.3 * np.random.random()  # More random mixing
            elif phase == "exploitation":
                # Favor fit parent
                parent_fitness_ratio = fitness_scores[i] / (fitness_scores[i] + fitness_scores[min(i+1, len(selected_population)-1)] + 1e-10)
                alpha = 0.3 + 0.4 * parent_fitness_ratio
            else:  # refinement
                alpha = 0.4 + 0.2 * np.random.random()  # Balanced mixing

            child1 = parent1 * alpha + parent2 * (1 - alpha)
            child2 = parent2 * alpha + parent1 * (1 - alpha)

            # Project children back to sphere
            for j in range(len(child1)):
                norm = np.linalg.norm(child1[j])
                if norm > 0:
                    child1[j] = child1[j] / norm
                norm = np.linalg.norm(child2[j])
                if norm > 0:
                    child2[j] = child2[j] / norm

            # Adaptive mutation with phase-specific strengths
            if phase == "exploration":
                mutation_strength = 0.1
            elif phase == "exploitation":
                mutation_strength = 0.05
            else:  # refinement
                mutation_strength = 0.01

            # Apply mutation
            child1 = intelligent_mutation(child1, mutation_strength, 0.03)
            child2 = intelligent_mutation(child2, mutation_strength, 0.03)

            new_population.extend([child1, child2])

        # Trim population to exact size
        population = new_population[:population_size]

        # Diversity maintenance
        if generation % 100 == 0 and generation > 0:
            # Add diversity if needed
            if diversity_score < 0.03:
                for i in range(3):
                    random_individual = adaptive_perturbation_strategy(initial_points, 0.0, 0.25)
                    if len(population) < population_size:
                        population.append(random_individual)
                    else:
                        population[np.random.randint(len(population))] = random_individual

    return best_individual, best_fitness

def enhanced_gradient_refinement(points):
    """Multi-stage gradient-based refinement with adaptive tolerances."""
    def objective(x_flat):
        points_local = x_flat.reshape(-1, 3)
        # Keep points on unit sphere constraint
        for i in range(len(points_local)):
            norm = np.linalg.norm(points_local[i])
            if norm > 0:
                points_local[i] = points_local[i] / norm
        return -compute_min_max_ratio(points_local)[2]  # Negative because we want to maximize

    def objective_with_grad(x_flat):
        points_local = x_flat.reshape(-1, 3)
        # Keep points on unit sphere constraint
        for i in range(len(points_local)):
            norm = np.linalg.norm(points_local[i])
            if norm > 0:
                points_local[i] = points_local[i] / norm
        value = -compute_min_max_ratio(points_local)[2]
        # Compute simple finite difference gradients
        eps = 1e-6
        grad = np.zeros_like(x_flat)
        for i in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_minus = x_flat.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return value, grad

    try:
        refined_points = points.copy()

        # Stage 1: Coarse refinement with relaxed tolerances
        result1 = minimize(objective, refined_points.flatten(), method='L-BFGS-B',
                         options={'maxiter': 150, 'ftol': 1e-5, 'gtol': 1e-5})
        refined_points = result1.x.reshape(-1, 3)

        # Project back to sphere
        for i in range(len(refined_points)):
            norm = np.linalg.norm(refined_points[i])
            if norm > 0:
                refined_points[i] = refined_points[i] / norm

        # Stage 2: Medium refinement with tighter tolerances
        result2 = minimize(objective, refined_points.flatten(), method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-7, 'gtol': 1e-7})
        refined_points = result2.x.reshape(-1, 3)

        # Project back to sphere again
        for i in range(len(refined_points)):
            norm = np.linalg.norm(refined_points[i])
            if norm > 0:
                refined_points[i] = refined_points[i] / norm

        # Stage 3: Fine refinement with extremely tight tolerances
        result3 = minimize(objective, refined_points.flatten(), method='L-BFGS-B',
                         options={'maxiter': 250, 'ftol': 1e-9, 'gtol': 1e-9})
        refined_points = result3.x.reshape(-1, 3)

        # Final projection back to sphere
        for i in range(len(refined_points)):
            norm = np.linalg.norm(refined_points[i])
            if norm > 0:
                refined_points[i] = refined_points[i] / norm

        return refined_points
    except Exception as e:
        # Fallback to iterative refinement if optimization fails
        current_points = points.copy()
        best_ratio = compute_min_max_ratio(current_points)[2]
        best_points = current_points.copy()

        # Iterative improvement with sphere constraint
        for iteration in range(1000):
            neighbor_points = current_points.copy()
            point_idx = np.random.randint(len(neighbor_points))

            # Small perturbation
            perturbation = np.random.normal(0, 0.0005, 3)
            neighbor_points[point_idx] += perturbation

            # Project back to sphere
            norm = np.linalg.norm(neighbor_points[point_idx])
            if norm > 0:
                neighbor_points[point_idx] = neighbor_points[point_idx] / norm

            new_ratio = compute_min_max_ratio(neighbor_points)[2]

            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = neighbor_points.copy()
                current_points = neighbor_points.copy()

        return best_points

def spherical_voronoi_evolution() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses evolutionary algorithm with Voronoi-based fitness evaluation.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Multi-start optimization with multiple initialization strategies
    best_points = None
    best_ratio = -np.inf

    # Generate multiple initial strategies
    initial_strategies = generate_initial_points()
    
    for strategy_idx, points in enumerate(initial_strategies):
        # Optimization parameters
        max_iterations = 150000  # Increased iterations for better search
        initial_temperature = 1.0
        cooling_rate = 0.9995
        min_temperature = 0.0001

        # Track best solution
        current_best_points = points.copy()
        current_best_ratio = compute_min_max_ratio(points)[2]  # Just get the ratio

        # Current state
        current_points = points.copy()
        current_ratio = current_best_ratio

        # Track ratio history for adaptive cooling
        ratio_history = [current_ratio]
        
        # Different temperature schedules for different phases
        temp_schedule = [
            {"temp": 1.0, "duration": 50000},   # High temperature for exploration
            {"temp": 0.5, "duration": 50000},   # Medium temperature for refinement  
            {"temp": 0.1, "duration": 50000}    # Low temperature for fine-tuning
        ]
        
        current_phase = 0
        phase_iterations = 0
        temp = initial_temperature

        # Simulated Annealing with multi-scale temperature
        last_improvement_iter = 0
        iteration = 0
        
        while iteration < max_iterations:
            # Check if we need to advance to next temperature phase
            if phase_iterations >= temp_schedule[current_phase]["duration"]:
                current_phase = min(current_phase + 1, len(temp_schedule) - 1)
                temp = temp_schedule[current_phase]["temp"]
                phase_iterations = 0
            
            # Perturb the current solution using spherical perturbations
            new_points = adaptive_perturbation_strategy(current_points, current_ratio, temp)

            # Compute new ratio
            new_min_dist, new_max_dist, new_ratio = compute_min_max_ratio(new_points)

            # Accept or reject the new solution using Metropolis criterion
            if new_ratio > current_ratio:
                # Always accept better solutions
                current_points = new_points
                current_ratio = new_ratio

                # Update best solution if this is better
                if new_ratio > current_best_ratio:
                    current_best_points = new_points.copy()
                    current_best_ratio = new_ratio
                    last_improvement_iter = iteration
                    ratio_history.append(new_ratio)
            else:
                # Accept worse solutions with probability based on temperature
                if temp > 0:  # Avoid division by zero
                    acceptance_prob = np.exp((new_ratio - current_ratio) / temp)
                    if np.random.rand() < acceptance_prob:
                        current_points = new_points
                        current_ratio = new_ratio
                        ratio_history.append(new_ratio)

            # Apply adaptive cooling
            temp = max(temp * adaptive_cooling(initial_temperature, iteration, max_iterations, ratio_history), min_temperature)
            
            # Increment counters
            iteration += 1
            phase_iterations += 1

            # Early stopping if no improvement in a long time
            if iteration - last_improvement_iter > 30000:
                break

        # Update global best if this run was better
        if current_best_ratio > best_ratio:
            best_ratio = current_best_ratio
            best_points = current_best_points.copy()

    # Ensure the result is properly normalized (should already be done, but extra safety)
    if best_points is not None:
        best_points = project_to_unit_sphere(best_points)
    else:
        # Fallback to Fibonacci if nothing worked
        best_points = fibonacci_sphere(14)
    
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Run the evolutionary algorithm with simulated annealing refinement
    points = spherical_voronoi_evolution()

    # Apply enhanced gradient refinement to the best solution
    final_points = enhanced_gradient_refinement(points)
    _, _, refined_ratio = compute_min_max_ratio(final_points)

    # Ensure normalization
    norms = np.linalg.norm(final_points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    final_points = final_points / norms

    return final_points

# EVOLVE-BLOCK-END