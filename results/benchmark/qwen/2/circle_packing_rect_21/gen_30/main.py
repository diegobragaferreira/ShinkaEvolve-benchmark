# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from platypus import NSGAII, Problem, Real, nondominated
from scipy.spatial import cKDTree
import random
from concurrent.futures import ThreadPoolExecutor
import time

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def check_overlap(circle1, circle2):
    """Check if two circles overlap"""
    dist = distance((circle1[0], circle1[1]), (circle2[0], circle2[1]))
    return dist < (circle1[2] + circle2[2])

def check_boundary(circle, rect_width, rect_height):
    """Check if circle is within rectangle boundaries"""
    x, y, r = circle
    return (r <= x <= rect_width - r) and (r <= y <= rect_height - r)

def compute_fitness(circles, rect_width, rect_height):
    """Compute fitness: sum of radii with penalties for constraint violations"""
    # Calculate sum of radii
    fitness = sum(circle[2] for circle in circles)
    
    # Penalty for boundary violations
    boundary_penalty = 0
    for circle in circles:
        if not check_boundary(circle, rect_width, rect_height):
            boundary_penalty += 1000
    
    # Penalty for overlap violations
    overlap_penalty = 0
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            if check_overlap(circles[i], circles[j]):
                overlap_penalty += 1000
    
    return fitness - boundary_penalty - overlap_penalty

def evaluate_individual(individual, rect_width, rect_height):
    """Evaluate a single individual (solution)"""
    # Convert flattened individual to circles array
    circles = []
    for i in range(0, len(individual), 3):
        circles.append([individual[i], individual[i+1], individual[i+2]])
    
    return compute_fitness(circles, rect_width, rect_height)

def generate_initial_population(pop_size, n_circles, rect_width, rect_height):
    """Generate diverse initial population"""
    population = []
    
    # Generate some grid-based solutions
    for _ in range(pop_size // 3):
        circles = []
        # Simple grid arrangement
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = int(np.ceil(n_circles / rows))
        
        cell_width = rect_width / cols
        cell_height = rect_height / rows
        
        radius = min(cell_width, cell_height) / 4
        
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * cell_width
            y = (row + 0.5) * cell_height
            circles.append([x, y, radius])
        
        # Add small random perturbations
        for circle in circles:
            circle[0] += random.uniform(-radius/4, radius/4)
            circle[1] += random.uniform(-radius/4, radius/4)
            
        # Flatten and add to population
        individual = [item for circle in circles for item in circle]
        population.append(individual)
    
    # Generate some random solutions
    for _ in range(2 * pop_size // 3):
        circles = []
        for i in range(n_circles):
            # Ensure valid initial positions and radii
            max_radius = min(rect_width, rect_height) / 10
            x = random.uniform(max_radius, rect_width - max_radius)
            y = random.uniform(max_radius, rect_height - max_radius)
            r = random.uniform(max_radius/10, max_radius)
            circles.append([x, y, r])
        
        individual = [item for circle in circles for item in circle]
        population.append(individual)
    
    return population

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    rect_width = 1.0  # Rectangle width (since perimeter = 4, width + height = 2)
    rect_height = 1.0  # Rectangle height
    
    # Define the problem
    def evaluate(x):
        # Convert to circles format
        circles = []
        for i in range(0, len(x), 3):
            circles.append([x[i], x[i+1], x[i+2]])
        
        # Return negative fitness since NSGA-II minimizes
        return [-compute_fitness(circles, rect_width, rect_height)]
    
    # Create problem with 63 variables (21 circles * 3 variables each)
    problem = Problem(63, 1)  # 63 variables, 1 objective
    for i in range(0, 63, 3):
        # X coordinates
        problem.types[i] = Real(0.01, rect_width - 0.01)
        # Y coordinates
        problem.types[i+1] = Real(0.01, rect_height - 0.01)
        # Radii
        problem.types[i+2] = Real(0.001, min(rect_width, rect_height) / 4)
    
    problem.function = evaluate
    
    # Run NSGA-II optimization
    algorithm = NSGAII(problem, population_size=50, offspring_size=25)
    
    # Set time limit to 60 seconds
    start_time = time.time()
    
    # Run optimization until time limit
    while time.time() - start_time < 55:  # Leave 5 seconds for final processing
        algorithm.step()
    
    # Get the best solution
    result = algorithm.result
    
    if not result:
        # Fallback to heuristic if optimization fails
        circles = []
        # Grid-based arrangement
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        cell_width = rect_width / cols
        cell_height = rect_height / rows
        
        radius = min(cell_width, cell_height) / 4
        
        for i in range(n):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * cell_width
            y = (row + 0.5) * cell_height
            circles.append([x, y, radius])
            
        # Add small random perturbations
        for circle in circles:
            circle[0] += random.uniform(-radius/4, radius/4)
            circle[1] += random.uniform(-radius/4, radius/4)
            
        return np.array(circles)
    
    # Extract best solution
    best_individual = result[0].variables
    circles = []
    for i in range(0, len(best_individual), 3):
        circles.append([best_individual[i], best_individual[i+1], best_individual[i+2]])
    
    return np.array(circles)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
