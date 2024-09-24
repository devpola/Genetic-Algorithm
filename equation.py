"""
Find the parameters(weights) that maximize equation shown below:
Y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6
inputs values (x1,x2,x3,x4,x5,x6) = (4,-2,3.5,-11,-4.7)
"""
import numpy as np

POPULATION_SIZE = 8
GENOME_LENGTH = 6
PARENTS_SIZE = 4
GENERATION = 4

def init_population(population_size, genome_length):
    return np.random.uniform(
        low=-4.0,
        high=4.0,
        size=(population_size, genome_length)
    )


def fitness(population, equation_inputs):
    """Calculates the sum of products between each input and its corresponding weight"""
    fitness = np.sum(population * equation_inputs, axis=1)
    return fitness


def select_parents(population, fitness_values, parents_size):
    """
    Select the best individuals(chromosome) in the current generation as parents
    for producing the offspring of the next generation
    """
    parents = np.empty((parents_size, population.shape[1]))

    for parent_idx in range(parents_size):
        max_fitness_idx = np.where(fitness_values == np.max(fitness_values))
        max_fitness_idx = max_fitness_idx[0][0]

        parents[parent_idx, :] = population[max_fitness_idx, :]
        fitness_values[max_fitness_idx] = -99999999999

    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]

        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offspring


def mutation(offspring_crossover):
    """Change a single gene in each offspring randomly"""
    for idx in range(offspring_crossover.shape[0]):
        random_value = np.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] += random_value

    return offspring_crossover


def main():
    # Inputs of the equation
    equation_inputs = [4, -2, 3.5, 5, -11, -4.7]

    # Create the initial population
    population = init_population(POPULATION_SIZE, GENOME_LENGTH)

    for generation in range(GENERATION):
        # Measure the fitness of each chromosome in the population
        fitness_values = fitness(equation_inputs, population)

        # Select the best parents in the population for mating
        parents = select_parents(population, fitness_values, PARENTS_SIZE)

        # Generate next generation using crossover
        offspring_size = (POPULATION_SIZE - PARENTS_SIZE, GENOME_LENGTH)
        offspring_crossover = crossover(parents, offspring_size)

        # Add some variations to the offspring using mutation
        offspring_mutation = mutation(offspring_crossover)

        # Create the new population based on the parents and offspring
        population[0:PARENTS_SIZE, :] = parents
        population[PARENTS_SIZE:, :] = offspring_mutation

        # The best result in the current iteration.
        print("Best result : ", np.max(np.sum(population*equation_inputs, axis=1)))

    # Get the best solution after iterating finishing all generations.
    # At first, the fitness is calculated for each solution in the final generation.
    fitness_values = fitness(equation_inputs, population)
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = np.where(fitness_values == np.max(fitness_values))

    print("Best solution : ", population[best_match_idx, :])
    print("Best solution fitness : ", fitness_values[best_match_idx])


if __name__ == '__main__':
    main()
