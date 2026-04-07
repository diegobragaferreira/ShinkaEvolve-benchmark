# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
import random
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Rectangle dimensions: width + height = 2
    # Allow dynamic optimization of aspect ratio
    rect_width = 1.2
    rect_height = 0.8
    
    n = 21
    
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define the bounds for each variable (x, y, r for each circle)
    # Individual format: [cx1, cy1, r1, cx2, cy2, r2, ..., cxn, cyn, rn]
    def create_individual():
        individual = []
        for i in range(n):
            # Random position within rectangle
            x = np.random.uniform(0.05, rect_width - 0.05)
            y = np.random.uniform(0.05, rect_height - 0.05)
            # Small initial radius
            r = np.random.uniform(0.01, 0.1)
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    def evaluate(individual):
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)
        
        # Calculate sum of radii (we want to maximize this)
        total_radius = np.sum(circles[:, 2])
        
        # Constraint penalty
        penalty = 0
        
        # Boundary constraints
        for i in range(n):
            cx, cy, r = circles[i]
            # Penalty for going outside
            if cx - r < 0:
                penalty += 1000 * (r - cx)**2
            if cx + r > rect_width:
                penalty += 1000 * (cx + r - rect_width)**2
            if cy - r < 0:
                penalty += 1000 * (r - cy)**2
            if cy + r > rect_height:
                penalty += 1000 * (cy + r - rect_height)**2
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cx1, cy1, r1 = circles[i]
                cx2, cy2, r2 = circles[j]
                
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                overlap = (r1 + r2) - dist
                
                if overlap > 0:  # Overlapping
                    penalty += 10000 * overlap**2
        
        # Return fitness (negative penalty plus total radius)
        return (total_radius - penalty,)  # Return tuple
    
    def mutate(individual):
        # Mutate some genes with Gaussian noise
        for i in range(len(individual)):
            if random.random() < 0.1:  # 10% mutation rate
                if i % 3 == 2:  # Radius mutation
                    individual[i] = max(0.001, individual[i] + random.gauss(0, 0.01))
                else:  # Position mutation
                    individual[i] = max(0.05, min(rect_width - 0.05 if i % 3 == 0 else rect_height - 0.05, 
                                                individual[i] + random.gauss(0, 0.05)))
        return individual,
    
    # Initialize population
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    pop = toolbox.population(n=50)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run evolution
    try:
        hof = tools.HallOfFame(1)
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, 
                                         ngen=50, stats=stats, halloffame=hof, verbose=False)
        best_individual = hof[0]
    except Exception as e:
        # Fallback to optimized initial solution
        print(f"Evolution failed: {e}")
        best_individual = create_individual()
    
    # Convert best individual to circles array
    circles = np.array(best_individual).reshape(-1, 3)
    
    # Local refinement to further optimize
    max_refinement_iter = 50
    for iteration in range(max_refinement_iter):
        improved = False
        
        # Try to increase each circle's radius
        for i in range(n):
            cx, cy, r = circles[i]
            
            # Compute max allowable radius
            max_radius = float('inf')
            
            # Boundary constraints
            max_radius = min(max_radius, cx)
            max_radius = min(max_radius, rect_width - cx)
            max_radius = min(max_radius, cy)
            max_radius = min(max_radius, rect_height - cy)
            
            # Overlap constraints with all other circles
            for j in range(n):
                if i != j:
                    other_cx, other_cy, other_r = circles[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    max_radius = min(max_radius, dist - other_r)
            
            # Increase radius if beneficial
            if max_radius > r and max_radius > 0:
                # Try to increase radius by small amount
                new_r = min(r + 0.005, max_radius)
                # Check if the new radius is valid
                valid = True
                for j in range(n):
                    if i != j:
                        other_cx, other_cy, other_r = circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        if dist < new_r + other_r:
                            valid = False
                            break
                if valid:
                    circles[i, 2] = new_r
                    improved = True
        
        if not improved:
            break
    
    # Final check and correction
    for i in range(n):
        cx, cy, r = circles[i]
        # Ensure radius is positive
        circles[i, 2] = max(0.001, r)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
