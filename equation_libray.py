"""
Find the parameters(weights) that satisfy equation shown below:
Y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6
inputs values (x1,x2,x3,x4,x5,x6) = (4,-2,3.5,-11,-4.7)
desired_output = 44
Genetic Algorithm Libray인 PyGAD를 활용한 예시 
"""
import pygad
import numpy as np
import math

GENERATION = 50     # num of generations
PARENTS_SIZE = 4    # num of solutions to be selected as parents in the mating pool
POPULATION_SIZE = 8 # num of solutions in the population; num of chromosomes
GENOME_LENGTH = 6

function_inputs = [4, -2, 3.5, 5, -11, -4.7]
desired_output = 44

init_range_low = -2
init_range_high = 5

parent_selection_type = "sss"
keep_parents = 1
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10


def fitness_func(ga_instance, solution, solution_idx):
    output = np.sum(solution * function_inputs)
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness


def main():
    ga_instance = pygad.GA(
                    num_generations=GENERATION,
                    num_parents_mating=PARENTS_SIZE,
                    sol_per_pop=POPULATION_SIZE,
                    num_genes=GENOME_LENGTH,
                    init_range_low=init_range_low,
                    init_range_high=init_range_high,
                    parent_selection_type=parent_selection_type,
                    keep_parents=keep_parents,
                    crossover_type=crossover_type,
                    mutation_num_genes=1,
                    mutation_type=mutation_type,
                    mutation_percent_genes=mutation_percent_genes,
                    fitness_func=fitness_func
                )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    prediction = np.sum(np.array(function_inputs)*solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
    

if __name__ == '__main__':
    main()