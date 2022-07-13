from random import random
from typing import List
from datatypes import Genome, Population, Matrix, Parents


def generate_population(population_size: int, genome_size: int) -> Population:
    values_list: List[int] = list(range(1, genome_size + 1))

    generation: Population = [random.sample(values_list, genome_size)
                              for _ in range(population_size)]

    return generation


def fitness(genome: Genome, distances: Matrix[int]) -> int:
    distance_summation: int = 0

    starting_location: int = genome[0]
    current_location: int = starting_location

    route: List[int] = genome[1:]

    for next_location in route:
        distance_summation += distances[current_location - 1][next_location - 1]
        current_location = next_location

    distance_summation += distances[current_location - 1][next_location - 1]

    return distance_summation


def select_parents(population: Population, number_parents: int,
                   fitness_function, distances: Matrix[int]) -> Parents:
    population = sorted(population, key=lambda genome: fitness_function(
        genome, distances))

    parents = []

    for index in range(0, number_parents, 2):
        start = index
        finish = index + 2
        parents_couple = tuple(population[start:finish])
        parents.append(parents_couple)

    return parents


def single_point_crossover(parent_1: Genome, parent_2: Genome,
                           crossover_rate: float) -> Genome:

    if len(parent_1) != len(parent_2):
        raise ValueError("GENOMES MUST BE OF THE SAME LENGTH")

    genome_length: int = len(parent_1)
    if genome_length == 2:
        return [parent_1[0], parent_2[1]]

    if random.random() > crossover_rate:
        return None

    crossover_index_1 = random.randrange(0, len(parent_1) - 1)
    crossover_index_2 = random.randrange(0, len(parent_1) - 1)

    starting_index = min(crossover_index_1, crossover_index_2)
    finishing_index = max(crossover_index_1, crossover_index_2)

    partial_offspring: Genome = parent_1[starting_index:finishing_index]
    remaining: Genome = [gene for gene in parent_2
                         if gene not in partial_offspring]

    offspring: Genome = partial_offspring + remaining

    return offspring


def mutate(genome: Genome, number_mutations: int, mutation_rate: float) -> Genome:
    for _ in range(number_mutations):
        if random.random() > mutation_rate:
            continue

        genome_length = len(genome)
        mutation_point_1 = random.randrange(genome_length)
        mutation_point_2 = random.randrange(genome_length)

        genome[mutation_point_1], genome[mutation_point_2] = \
            genome[mutation_point_2], genome[mutation_point_1]

    return genome
