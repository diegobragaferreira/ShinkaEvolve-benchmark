# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
from typing import Tuple, List, Optional
import time
from joblib import Parallel, delayed
from scipy.optimize import minimize
import warnings
from sklearn.cluster import DBSCAN

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class AdvancedHexagonGeometry:
    """Advanced geometric operations for hexagon computations with enhanced robustness."""
    
    @staticmethod
    def create_regular_hexagon(center_x: float, center_y: float, side_length: float = 1.0, rotation_deg: float = 0.0) -> Polygon:
        """Create a regular hexagon as a Shapely polygon with enhanced numerical stability."""
        angle_rad = np.radians(rotation_deg)
        points = []
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            x = center_x + side_length * np.cos(angle)
            y = center_y + side_length * np.sin(angle)
            points.append((x, y))
        return Polygon(points)
    
    @staticmethod
    def check_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
        """Robust containment check using geometric properties."""
        try:
            return outer_hex.contains(hexagon)
        except:
            # Fallback: check if all vertices are contained
            for point in hexagon.exterior.coords:
                if not outer_hex.contains(Point(point[0], point[1])):
                    return False
            return True
    
    @staticmethod
    def check_collision(hex1: Polygon, hex2: Polygon) -> bool:
        """Efficient collision detection with early exit conditions."""
        try:
            return hex1.intersects(hex2)
        except:
            # Fallback: check if bounding boxes intersect
            bbox1 = hex1.bounds
            bbox2 = hex2.bounds
            return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
                       bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
    
    @staticmethod
    def compute_min_outer_radius(inner_hex_data: np.ndarray) -> float:
        """Compute minimum outer radius with geometric insights."""
        if len(inner_hex_data) == 0:
            return 1.0
        max_dist = 0.0
        for i in range(len(inner_hex_data)):
            center_x, center_y, _ = inner_hex_data[i]
            # Distance from center to hexagon center plus the hexagon's circumradius
            dist = np.sqrt(center_x**2 + center_y**2) + 1.0
            max_dist = max(max_dist, dist)
        return max_dist * 1.05  # Add safety margin

class MultiScaleHexagonEvaluator:
    """Multi-scale evaluator that validates configurations with different granularities."""
    
    @staticmethod
    def evaluate_packing_with_validation(inner_hex_data: np.ndarray, outer_hex_side_length: float) -> Tuple[float, bool, str]:
        """
        Multi-scale evaluation with progressive validation.
        
        Returns:
            tuple: (penalty_score, is_valid, message)
        """
        # Step 1: Fast bounding box validation
        if len(inner_hex_data) != 11:
            return 10000.0, False, "Wrong number of hexagons"
        
        # Step 2: Coarse validation - check if any hexagon is out of bounds
        outer_hex = AdvancedHexagonGeometry.create_regular_hexagon(0, 0, outer_hex_side_length)
        
        # Check all hexagons individually for basic containment
        hexagons = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, rotation = inner_hex_data[i]
            try:
                inner_hex = AdvancedHexagonGeometry.create_regular_hexagon(center_x, center_y, 1.0, rotation)
                if not AdvancedHexagonGeometry.check_containment(inner_hex, outer_hex):
                    return 1000.0, False, "Containment violation"
                hexagons.append(inner_hex)
            except:
                return 1000.0, False, "Invalid hexagon construction"
        
        # Step 3: Collision detection with early exit
        for i in range(len(hexagons)):
            for j in range(i+1, len(hexagons)):
                if AdvancedHexagonGeometry.check_collision(hexagons[i], hexagons[j]):
                    return 100.0, False, "Collision detected"
        
        # Step 4: No violations found
        return 0.0, True, "Valid configuration"

    @staticmethod
    def binary_search_outer_radius(inner_hex_data: np.ndarray, min_radius: float,
                                  max_radius: float, tolerance: float = 0.001) -> float:
        """Binary search with better convergence handling."""
        if min_radius >= max_radius:
            return min_radius
            
        while max_radius - min_radius > tolerance:
            mid_radius = (min_radius + max_radius) / 2.0
            penalty, is_valid, _ = MultiScaleHexagonEvaluator.evaluate_packing_with_validation(inner_hex_data, mid_radius)
            if is_valid:
                max_radius = mid_radius
            else:
                min_radius = mid_radius
        return max_radius

class ClusterBasedHexagonOptimizer:
    """Optimizer using cluster-based initialization and multi-resolution search."""
    
    @staticmethod
    def cluster_hexagon_positions(data: np.ndarray, eps: float = 1.5, min_samples: int = 2) -> Tuple[List[np.ndarray], List[int]]:
        """Cluster hexagon positions to identify natural groupings."""
        # Only consider x,y positions for clustering
        positions = data[:, :2]
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
        labels = clustering.labels_
        
        # Group by cluster
        clusters = {}
        cluster_labels = []
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
                cluster_labels.append(label)
            clusters[label].append(i)
        
        cluster_arrays = [data[clusters[label]] for label in cluster_labels if label != -1]
        return cluster_arrays, labels

    @staticmethod
    def generate_clustered_initial_config() -> np.ndarray:
        """Generate initial configuration based on clustering insights."""
        # Create clusters of hexagon positions that are naturally grouped
        config = np.zeros((11, 3))
        
        # Cluster 1: Central area
        config[0] = [0.0, 0.0, 0.0]  # Center
        
        # Cluster 2: First ring around center
        ring1_angles = [i * 60 for i in range(6)]
        ring1_distance = 2.0
        
        for i, angle in enumerate(ring1_angles):
            rad = np.radians(angle)
            x = ring1_distance * np.cos(rad)
            y = ring1_distance * np.sin(rad)
            config[i+1] = [x, y, 0.0]
        
        # Cluster 3: Second ring (spaced out)
        ring2_angles = [30, 90, 150, 210]
        ring2_distance = 3.5
        
        for i, angle in enumerate(ring2_angles):
            rad = np.radians(angle)
            x = ring2_distance * np.cos(rad)
            y = ring2_distance * np.sin(rad)
            config[i+7] = [x, y, 0.0]
        
        # Add slight jitter for diversity
        np.random.seed(42)
        for i in range(len(config)):
            config[i, 0] += np.random.normal(0, 0.1)
            config[i, 1] += np.random.normal(0, 0.1)
            config[i, 2] += np.random.normal(0, 5)
            config[i, 2] %= 360.0
            
        return config
    
    @staticmethod
    def generate_initial_population(pop_size: int) -> List[np.ndarray]:
        """Generate diverse initial population with clustering strategy."""
        population = []
        
        # Start with clustered configuration  
        clustered_config = ClusterBasedHexagonOptimizer.generate_clustered_initial_config()
        
        for _ in range(pop_size):
            individual = clustered_config.copy()
            
            # Apply cluster-aware perturbations
            np.random.seed(int(time.time() * 1000) % 2**32)
            for i in range(len(individual)):
                # Different perturbation magnitudes based on spatial clustering
                if i <= 1:  # Central hexagons
                    individual[i, 0] += np.random.normal(0, 0.15)
                    individual[i, 1] += np.random.normal(0, 0.15)
                elif i <= 7:  # First ring
                    individual[i, 0] += np.random.normal(0, 0.2)
                    individual[i, 1] += np.random.normal(0, 0.2)
                else:  # Second ring
                    individual[i, 0] += np.random.normal(0, 0.25)
                    individual[i, 1] += np.random.normal(0, 0.25)
                individual[i, 2] += np.random.normal(0, 10)
                individual[i, 2] %= 360.0
                
            population.append(individual)
            
        return population
    
    @staticmethod
    def mutate_individual(individual: np.ndarray, generation: int = 0, 
                         pop_size: int = 50) -> np.ndarray:
        """Smart mutation that adapts to evolution progress."""
        mutated = individual.copy()
        
        # Dynamic mutation rates based on generation
        mutation_rate = max(0.05, 0.3 * (1.0 - generation / (2.0 * pop_size)))
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Cluster-aware mutation strengths
                if i <= 1:  # Central
                    mutated[i, 0] += np.random.normal(0, 0.15)
                    mutated[i, 1] += np.random.normal(0, 0.15)
                elif i <= 7:  # Ring 1
                    mutated[i, 0] += np.random.normal(0, 0.2)
                    mutated[i, 1] += np.random.normal(0, 0.2)
                else:  # Ring 2
                    mutated[i, 0] += np.random.normal(0, 0.25)
                    mutated[i, 1] += np.random.normal(0, 0.25)
                    
                # Rotation mutations
                mutated[i, 2] += np.random.normal(0, 15)
                mutated[i, 2] %= 360.0
                
        return mutated
    
    @staticmethod
    def crossover_parents(parent1: np.ndarray, parent2: np.ndarray, 
                         generation: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover with adaptive strategy."""
        # Use higher crossover rate early, lower later
        crossover_rate = max(0.7, 0.9 - generation / 200.0)
        
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Multi-point crossover that preserves spatial clustering
        # Split into 3 segments to respect spatial grouping
        segment_size = len(parent1) // 3
        split_points = [segment_size, 2 * segment_size]
        
        # Choose split point randomly
        split_point = random.choice(split_points)
        
        child1 = np.vstack([parent1[:split_point], parent2[split_point:]])
        child2 = np.vstack([parent2[:split_point], parent1[split_point:]])
        
        return child1, child2

def smart_local_refinement(config: np.ndarray, outer_radius: float, 
                         max_iter: int = 100) -> np.ndarray:
    """Smart local refinement using hybrid optimization strategies."""
    def objective(params):
        # Reshape to configuration
        test_config = params.reshape(-1, 3)
        penalty, is_valid, _ = MultiScaleHexagonEvaluator.evaluate_packing_with_validation(test_config, outer_radius)
        return penalty if is_valid else 10000.0
    
    # Flatten configuration 
    initial_params = config.flatten()
    
    try:
        # Try multiple optimization strategies
        # Strategy 1: L-BFGS-B with conservative bounds
        result = minimize(objective, initial_params, method='L-BFGS-B',
                         bounds=[(-10, 10), (-10, 10), (0, 360)] * len(config),
                         options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if result.success:
            refined_config = result.x.reshape(-1, 3)
            # Validate result
            penalty, is_valid, _ = MultiScaleHexagonEvaluator.evaluate_packing_with_validation(refined_config, outer_radius)
            if is_valid and penalty < 1000.0:
                return refined_config
    except:
        pass
    
    # Strategy 2: Simple coordinate descent with adaptive step sizes
    try:
        current_params = initial_params.copy()
        current_penalty = objective(current_params)
        
        step_sizes = [0.1, 0.05, 0.01]  # Adaptive step sizes
        for step_idx, step_size in enumerate(step_sizes):
            for _ in range(20):  # Limited iterations per step size
                new_params = current_params.copy()
                
                # Random perturbation of a random subset of parameters
                indices = np.random.choice(len(new_params), size=max(1, len(new_params)//5), replace=False)
                for idx in indices:
                    if idx % 3 < 2:  # Position coordinate
                        new_params[idx] += np.random.normal(0, step_size)
                    else:  # Angle
                        new_params[idx] += np.random.normal(0, step_size * 5)
                        new_params[idx] %= 360
                        
                new_penalty = objective(new_params)
                if new_penalty < current_penalty:
                    current_params = new_params
                    current_penalty = new_penalty
                    
                    # Early termination if improvement is minimal
                    if current_penalty < 1.0:
                        break
                        
        refined_config = current_params.reshape(-1, 3)
        return refined_config
    except:
        pass
    
    # Fallback to original config
    return config

def optimize_hexagon_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Advanced evolutionary optimization using clustering and multi-scale approach.
    
    Returns:
        tuple: (inner_hex_data, outer_hex_data, outer_hex_side_length)
    """
    # Parameters
    pop_size = 60
    generations = 150
    min_outer_radius = 2.0
    max_outer_radius = 12.0
    
    # Time limit enforcement
    start_time = time.time()
    timeout_seconds = 175  # Leave some buffer
    
    # Initialize population with cluster-based approach
    population = ClusterBasedHexagonOptimizer.generate_initial_population(pop_size)
    best_individual = None
    best_penalty = float('inf')
    best_outer_radius = max_outer_radius
    
    # Evolution loop with multi-scale refinement
    for generation in range(generations):
        if time.time() - start_time > timeout_seconds:
            break
            
        # Evaluate population with early exit conditions
        fitness_scores = []
        
        def evaluate_individual(individual):
            penalty, is_valid, msg = MultiScaleHexagonEvaluator.evaluate_packing_with_validation(individual, max_outer_radius)
            return penalty, individual, is_valid
        
        # Process in parallel for efficiency
        results = Parallel(n_jobs=-1)(delayed(evaluate_individual)(individual) for individual in population)
        
        # Filter valid individuals and sort by fitness
        valid_results = [(penalty, individual, is_valid) for penalty, individual, is_valid in results if is_valid]
        valid_results.sort(key=lambda x: x[0])
        
        # Update best solution
        if valid_results and valid_results[0][0] < best_penalty:
            best_penalty = valid_results[0][0]
            best_individual = valid_results[0][1].copy()
            
            # Binary search for tightest outer radius
            if best_individual is not None:
                estimated_min_radius = AdvancedHexagonGeometry.compute_min_outer_radius(best_individual)
                min_test_radius = max(min_outer_radius, estimated_min_radius)
                test_radius = MultiScaleHexagonEvaluator.binary_search_outer_radius(
                    best_individual, min_test_radius, max_outer_radius
                )
                best_outer_radius = test_radius
        
        # Early stopping if good solution found
        if best_penalty < 1.0:
            break
            
        # Multi-scale refinement: local optimization on best solution
        if best_individual is not None:
            refined_config = smart_local_refinement(best_individual, best_outer_radius, max_iter=50)
            penalty, is_valid, _ = MultiScaleHexagonEvaluator.evaluate_packing_with_validation(refined_config, best_outer_radius)
            if is_valid and penalty < best_penalty:
                best_penalty = penalty
                best_individual = refined_config
                # Re-compute tight outer radius
                estimated_min_radius = AdvancedHexagonGeometry.compute_min_outer_radius(best_individual)
                min_test_radius = max(min_outer_radius, estimated_min_radius)
                test_radius = MultiScaleHexagonEvaluator.binary_search_outer_radius(
                    best_individual, min_test_radius, best_outer_radius
                )
                best_outer_radius = test_radius
        
        # Selection and reproduction with adaptive strategies
        # Select top performers
        elite_size = max(3, int(0.1 * pop_size))
        elite = [individual for penalty, individual, is_valid in valid_results[:elite_size]]
        
        # Generate new population
        new_population = elite.copy()
        
        # Add offspring through crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection (larger tournaments for more pressure)
            parent1 = random.choice(valid_results)[1] if valid_results else population[0]
            parent2 = random.choice(valid_results)[1] if valid_results else population[1]
            
            child1, child2 = ClusterBasedHexagonOptimizer.crossover_parents(parent1, parent2, generation)
            
            # Mutate children
            child1 = ClusterBasedHexagonOptimizer.mutate_individual(child1, generation, pop_size)
            child2 = ClusterBasedHexagonOptimizer.mutate_individual(child2, generation, pop_size)
            
            new_population.extend([child1, child2])
            
        # Trim population to exact size
        population = new_population[:pop_size]
        
        # Periodic full validation and cleanup
        if generation % 20 == 0 and best_individual is not None:
            # Perform final validation with binary search
            try:
                estimated_min_radius = AdvancedHexagonGeometry.compute_min_outer_radius(best_individual)
                min_test_radius = max(min_outer_radius, estimated_min_radius)
                test_radius = MultiScaleHexagonEvaluator.binary_search_outer_radius(
                    best_individual, min_test_radius, best_outer_radius
                )
                if test_radius < best_outer_radius:
                    best_outer_radius = test_radius
            except:
                pass
    
    # Final refinement and validation
    if best_individual is not None:
        # Final local optimization
        refined_final = smart_local_refinement(best_individual, best_outer_radius, max_iter=30)
        penalty, is_valid, _ = MultiScaleHexagonEvaluator.evaluate_packing_with_validation(refined_final, best_outer_radius)
        if is_valid and penalty < best_penalty:
            best_penalty = penalty
            best_individual = refined_final
            
        # Final binary search for tightest fit
        try:
            estimated_min_radius = AdvancedHexagonGeometry.compute_min_outer_radius(best_individual)
            min_test_radius = max(min_outer_radius, estimated_min_radius)
            final_radius = MultiScaleHexagonEvaluator.binary_search_outer_radius(
                best_individual, min_test_radius, best_outer_radius
            )
            best_outer_radius = final_radius
        except:
            pass
    
    # Construct final return values
    inner_hex_data = best_individual if best_individual is not None else np.zeros((11, 3))
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    
    return inner_hex_data, outer_hex_data, best_outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()

    # Ensure we return at least the minimum possible result
    if outer_hex_side_length <= 0:
        outer_hex_side_length = 10.0

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END