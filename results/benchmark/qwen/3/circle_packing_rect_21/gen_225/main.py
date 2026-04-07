# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
import math
import random
from deap import base, creator, tools, algorithms
import multiprocessing as mp
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Container setup (perimeter = 4, so width + height = 2)
    container_width, container_height = 1.2, 0.8

    # Number of circles
    n = 21

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Define the individual (chromosome) as [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    IND_SIZE = n * 3
    
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # Attribute generator
    def create_individual():
        # Initialize with Voronoi-based distribution
        anchors = []
        # Corners
        anchors.extend([(0.05, 0.05), (container_width - 0.05, 0.05),
                       (0.05, container_height - 0.05), (container_width - 0.05, container_height - 0.05)])
        # Edge centers
        anchors.extend([(container_width/2, 0.05), (container_width/2, container_height - 0.05),
                       (0.05, container_height/2), (container_width - 0.05, container_height/2)])

        # Generate candidate points
        candidate_points = []
        grid_density = 32
        for i in range(grid_density):
            for j in range(grid_density):
                x = (i + 0.5) / grid_density * container_width
                y = (j + 0.5) / grid_density * container_height
                candidate_points.append((x, y))

        # Add random points
        for _ in range(50):
            x = np.random.uniform(0.05, container_width - 0.05)
            y = np.random.uniform(0.05, container_height - 0.05)
            candidate_points.append((x, y))

        # Use k-means++ to get initial centroids for Voronoi
        all_points = anchors + candidate_points
        all_array = np.array(all_points)
        kmeans = KMeans(n_clusters=min(len(all_points), n), init='k-means++', n_init=10, random_state=42)
        kmeans.fit(all_array)
        selected_centroids = kmeans.cluster_centers_

        # Select positions
        selected_positions = []
        for centroid in selected_centroids:
            distances = np.linalg.norm(all_array - centroid, axis=1)
            closest_idx = np.argmin(distances)
            selected_positions.append(tuple(all_array[closest_idx]))
        
        if len(selected_positions) < n:
            for i in range(n - len(selected_positions)):
                selected_positions.append(candidate_points[i])
        elif len(selected_positions) > n:
            selected_positions = selected_positions[:n]

        # Create individual with positions and small radii
        individual = []
        for i in range(n):
            x, y = selected_positions[i]
            individual.extend([x, y, 0.02])
        
        return creator.Individual(individual)

    # Register functions
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluate function
    def eval_circle_packing(individual):
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)
        
        # Check constraints and compute fitness
        violations = 0
        total_radius = 0
        density_penalty = 0
        
        # Check boundary violations
        for i in range(n):
            x, y, r = circles[i]
            if x < r or x > container_width - r or y < r or y > container_height - r:
                violations += 1
        
        # Check overlap violations
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2):
                    violations += 1
        
        # Calculate density penalty (fewer violations = better)
        if violations > 0:
            # Penalty proportional to number of violations
            density_penalty = violations * 10000
        
        # Calculate total radius
        for i in range(n):
            total_radius += circles[i, 2]
            
        # Density-aware bonus: encourage well-distributed circles
        bonus = 0
        if violations == 0:
            # Compute average distance between circles
            distances = cdist(circles[:, :2], circles[:, :2])
            # Exclude diagonal elements (distance to self)
            np.fill_diagonal(distances, np.inf)
            avg_distance = np.min(distances, axis=1).mean()
            
            # Bonus for well-separated circles
            bonus = avg_distance * 1000
        
        # Fitness: maximize total radius minus penalties
        fitness = total_radius - density_penalty + bonus
        
        # Ensure positive fitness
        fitness = max(0, fitness)
        
        return (fitness,)

    toolbox.register("evaluate", eval_circle_packing)
    
    # Operators
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run genetic algorithm
    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    
    # Run evolution
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=0.7, mutpb=0.3, ngen=50,
        stats=stats, halloffame=hof, verbose=False
    )
    
    # Get the best individual
    best_individual = hof[0]
    best_circles = np.array(best_individual).reshape(-1, 3)
    
    # Extract the best configuration
    final_circles = np.zeros((n, 3))
    
    # Apply post-processing to further optimize
    for i in range(n):
        final_circles[i] = [best_circles[i, 0], best_circles[i, 1], best_circles[i, 2]]
    
    # Local optimization on best solution
    best_sum = np.sum(final_circles[:, 2])
    
    # Try to improve further with local search
    for _ in range(500):
        improved = False
        for i in range(n):
            # Store current values
            current_x, current_y, current_r = final_circles[i]
            
            # Try to increase radius
            max_radius = float('inf')
            
            # Boundary constraints
            boundary_radius = min(current_x, container_width - current_x, 
                                current_y, container_height - current_y)
            max_radius = min(max_radius, boundary_radius)
            
            # Circle-to-circle constraints
            for j in range(n):
                if i != j:
                    dist = math.sqrt((current_x - final_circles[j, 0])**2 + 
                                   (current_y - final_circles[j, 1])**2)
                    if dist > 0.001:
                        constraint_radius = dist - final_circles[j, 2]
                        max_radius = min(max_radius, constraint_radius)
            
            # Limit maximum radius
            max_radius = min(max_radius, 0.4)
            
            if max_radius > current_r:
                final_circles[i, 2] = max_radius
                improved = True
        
        if not improved:
            break
    
    # Final cleanup - ensure circles are within bounds
    for i in range(n):
        x, y, r = final_circles[i]
        x = max(r, min(container_width - r, x))
        y = max(r, min(container_height - r, y))
        final_circles[i] = [x, y, r]
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")