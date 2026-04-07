# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from numba import njit
import time
import random
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
import warnings
from copy import deepcopy

# Core hexagon utility functions
@njit
def generate_hexagon_vertices(x: float, y: float, angle_deg: float, side_length: float = 1.0) -> np.ndarray:
    """Generate vertices of a regular hexagon given center, rotation, and side length"""
    angle_rad = np.radians(angle_deg)
    vertices = np.empty((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + side_length * np.cos(theta)
        vertices[i, 1] = y + side_length * np.sin(theta)
    return vertices

@njit
def check_containment_inner_to_outer(inner_x: float, inner_y: float, inner_angle: float,
                                   outer_x: float, outer_y: float, outer_angle: float, 
                                   outer_side_length: float) -> bool:
    """Check if inner hexagon is fully contained within outer hexagon"""
    inner_vertices = generate_hexagon_vertices(inner_x, inner_y, inner_angle, 1.0)
    
    # Create outer hexagon vertices
    outer_vertices = generate_hexagon_vertices(outer_x, outer_y, outer_angle, outer_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    # Check if all vertices of inner hexagon are inside outer hexagon
    for i in range(6):
        if not outer_polygon.contains(Point(inner_vertices[i, 0], inner_vertices[i, 1])):
            return False
            
    return True

@njit
def check_overlap_hexagons(x1: float, y1: float, angle1: float, 
                          x2: float, y2: float, angle2: float) -> bool:
    """Check if two hexagons overlap using vertex-based collision detection"""
    vertices1 = generate_hexagon_vertices(x1, y1, angle1, 1.0)
    vertices2 = generate_hexagon_vertices(x2, y2, angle2, 1.0)
    
    # Simple bounding box check first
    min1 = np.min(vertices1, axis=0)
    max1 = np.max(vertices1, axis=0)
    min2 = np.min(vertices2, axis=0)
    max2 = np.max(vertices2, axis=0)
    
    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return False
    
    # Create polygons and check intersection
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)
    
    # If intersection exists, they overlap
    return poly1.intersects(poly2)

class HexagonPackingOptimizer:
    """Specialized optimizer for 12-hexagon packing problem using hybrid evolutionary approach"""
    
    def __init__(self):
        self.best_solution = None
        self.best_score = float('inf')
        
    def _initialize_base_pattern(self) -> List[Tuple[float, float, float]]:
        """Create mathematically-informed base configuration for 12 hexagons"""
        # Inspired by optimal known arrangements and packing theory
        base_pattern = [
            [0.0, 0.0, 0.0],        # Center (primary)
            [0.0, 3.1, 0.0],        # Top (secondary)
            [0.0, -3.1, 0.0],       # Bottom (secondary)
            [2.65, 1.53, 0.0],      # Top Right (tertiary)
            [-2.65, 1.53, 0.0],     # Top Left (tertiary)
            [2.65, -1.53, 0.0],     # Bottom Right (tertiary)
            [-2.65, -1.53, 0.0],    # Bottom Left (tertiary)
            [3.6, 0.0, 0.0],        # Far Right (quaternary)
            [-3.6, 0.0, 0.0],       # Far Left (quaternary)
            [1.8, 2.9, 0.0],        # Upper Middle Right (quinary)
            [-1.8, 2.9, 0.0],       # Upper Middle Left (quinary)
            [1.8, -2.9, 0.0],       # Lower Middle Right (quinary)
            [-1.8, -2.9, 0.0],      # Lower Middle Left (quinary)
        ]
        return base_pattern[:-1]  # Remove outer side length
    
    def _generate_initial_population(self, pop_size: int) -> List[List[float]]:
        """Generate diverse initial population with mathematical guidance"""
        population = []
        base_pattern = self._initialize_base_pattern()
        
        for _ in range(pop_size):
            config = []
            # Add perturbations with varying intensities
            for i, (x, y, angle) in enumerate(base_pattern):
                # Vary perturbation intensity based on hexagon position
                if i == 0:  # Center
                    pert_x = x + random.uniform(-0.1, 0.1)
                    pert_y = y + random.uniform(-0.1, 0.1)
                    pert_angle = angle + random.uniform(-5, 5)
                elif i <= 6:  # Primary ring
                    pert_x = x + random.uniform(-0.2, 0.2)
                    pert_y = y + random.uniform(-0.2, 0.2)
                    pert_angle = angle + random.uniform(-8, 8)
                else:  # Outer ring
                    pert_x = x + random.uniform(-0.3, 0.3)
                    pert_y = y + random.uniform(-0.3, 0.3)
                    pert_angle = angle + random.uniform(-12, 12)
                config.extend([pert_x, pert_y, pert_angle])
            
            # Add outer side length with reasonable starting range
            config.append(5.5 + random.uniform(0, 1.5))
            population.append(config)
        return population
    
    def _evaluate_individual(self, params: List[float]) -> float:
        """Fast evaluation of individual fitness with early termination"""
        # Extract hexagon params
        hexagons = []
        idx = 0
        for i in range(12):
            hexagons.append((params[idx], params[idx+1], params[idx+2]))
            idx += 3
        outer_side_length = params[-1]
        
        # Quick containment check - early termination if violated
        containment_penalty = 0
        for i in range(12):
            if not check_containment_inner_to_outer(
                hexagons[i][0], hexagons[i][1], hexagons[i][2],
                0, 0, 0, outer_side_length
            ):
                containment_penalty += 1e7
        
        # If containment violated, return immediately
        if containment_penalty > 0:
            return containment_penalty + 1e10
        
        # Overlap checking with early termination
        overlap_penalty = 0
        for i in range(12):
            for j in range(i+1, 12):
                if check_overlap_hexagons(
                    hexagons[i][0], hexagons[i][1], hexagons[i][2],
                    hexagons[j][0], hexagons[j][1], hexagons[j][2]
                ):
                    overlap_penalty += 1e6
                    # Early termination if overlaps found
                    if overlap_penalty >= 1e9:
                        break
            if overlap_penalty >= 1e9:
                break
        
        # Calculate objective value
        objective = -1.0 / outer_side_length + containment_penalty + overlap_penalty
        return objective
    
    def _evolve_generation(self, population: List[List[float]], 
                          bounds: List[Tuple[float, float]]) -> List[List[float]]:
        """Evolutionary generation with custom operators"""
        # Sort by fitness
        population.sort(key=lambda x: self._evaluate_individual(x))
        best_fitness = self._evaluate_individual(population[0])
        
        if best_fitness < self.best_score:
            self.best_score = best_fitness
            self.best_solution = deepcopy(population[0])
        
        # Select top 50% for breeding (elitism)
        elite_count = len(population) // 2
        elites = population[:elite_count]
        
        # Generate offspring through specialized crossover/mutation
        offspring = []
        for i in range(elite_count):
            parent1 = elites[i % len(elites)]
            parent2 = elites[(i + 1) % len(elites)]
            
            # Crossover: blend positions, combine rotations
            child = self._crossover(parent1, parent2)
            
            # Mutation: perturb with adaptive intensity
            child = self._mutate(child, bounds)
            offspring.append(child)
        
        # Fill remainder with new individuals
        new_individuals = self._generate_initial_population(len(population) - len(offspring))
        return elites + offspring + new_individuals
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Custom crossover operator for hexagon packing"""
        child = []
        # Cross over positions and rotations
        for i in range(0, 36, 3):  # Process groups of 3 (x,y,angle)
            if random.random() < 0.5:
                child.extend([parent1[i], parent1[i+1], parent1[i+2]])
            else:
                child.extend([parent2[i], parent2[i+1], parent2[i+2]])
        
        # Blend outer side length
        child.append((parent1[-1] + parent2[-1]) / 2)
        return child
    
    def _mutate(self, individual: List[float], bounds: List[Tuple[float, float]]) -> List[float]:
        """Adaptive mutation operator with symmetry preservation"""
        mutated = individual.copy()
        
        # Mutate positions and rotations
        for i in range(36):  # 12 hexagons * 3 parameters
            if random.random() < 0.2:  # 20% chance to mutate each parameter
                # Adaptive mutation strength based on parameter type
                if i % 3 == 0:  # x coordinate
                    strength = 0.1  # Small mutation for precise positioning
                elif i % 3 == 1:  # y coordinate  
                    strength = 0.1
                else:  # angle
                    strength = 2.0  # Larger mutation for rotation
                
                mutated[i] += random.uniform(-strength, strength)
                
                # Apply bounds
                mutated[i] = max(bounds[i][0], min(bounds[i][1], mutated[i]))
        
        # Mutate outer side length
        if random.random() < 0.1:  # 10% chance
            mutated[-1] *= random.uniform(0.95, 1.05)  # Small adjustment
            mutated[-1] = max(bounds[-1][0], min(bounds[-1][1], mutated[-1]))
        
        return mutated
    
    def _progressive_optimize(self, bounds: List[Tuple[float, float]], 
                            max_generations: int = 50) -> List[float]:
        """Progressive optimization with increasing resolution"""
        # Generation 1: Coarse resolution
        pop_size = 30
        population = self._generate_initial_population(pop_size)
        
        for gen in range(max_generations):
            population = self._evolve_generation(population, bounds)
            
            # Gradually increase precision in later generations
            if gen > max_generations // 2:
                # Increase mutation rates for fine-tuning
                pass
        
        # Final optimization using DE for fine-tuning
        if self.best_solution is not None:
            def objective_func(params):
                return self._evaluate_individual(params)
            
            try:
                # Fine tune with differential evolution
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=42,
                    maxiter=50,
                    popsize=15,
                    mutation=(0.8, 1),
                    recombination=0.9,
                    tol=1e-6,
                    workers=1,
                    init=[self.best_solution]
                )
                
                return result.x
            except:
                return self.best_solution
        else:
            # Fallback to best from population
            population.sort(key=lambda x: self._evaluate_individual(x))
            return population[0]
    
    def optimize(self, bounds: List[Tuple[float, float]]) -> List[float]:
        """Main optimization routine with progressive approach"""
        # Run progressive optimization
        final_solution = self._progressive_optimize(bounds)
        return final_solution

def hexagon_packing_12() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Define bounds for optimization
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    bounds.append((1, 20))  # Outer side length bound
    
    try:
        # Initialize optimizer
        optimizer = HexagonPackingOptimizer()
        
        # Run optimization
        best_params = optimizer.optimize(bounds)
        
        # Extract configuration
        inner_hex_data = []
        idx = 0
        for i in range(12):
            inner_hex_data.append([
                best_params[idx], 
                best_params[idx+1], 
                best_params[idx+2]
            ])
            idx += 3
            
        outer_side_length = best_params[-1]
        
        # Store results
        inner_hex_data = np.array(inner_hex_data)
        outer_hex_data = np.array([0, 0, 0])
        
        # Calculate metrics
        inv_outer_hex_side_length = 1.0 / outer_side_length
        benchmark_ratio = inv_outer_hex_side_length / 0.2537
        
        print(f"inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to previous solution
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_side_length = 8.0

        # Calculate fallback metrics
        inv_outer_hex_side_length, benchmark_ratio = 1.0/outer_side_length, 1.0/outer_side_length/0.2537
        print(f"Fallback - inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"Fallback - benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")

    # Ensure all computations completed within time limit
    elapsed_time = time.time() - start_time
    if elapsed_time > 175:  # Leave buffer
        warnings.warn("Warning: Time limit approaching")
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END