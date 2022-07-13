from os import path
import time
import calendar

# TYPING
from typing import List
from datatypes import Matrix, GeneticSolution, GreedySolution

# THIRD PARTY
import utils

# GENETIC ALGORITHM
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.functions import *

# GREEDY ALGORITHM
from greedy.greedy import Greedy


def running_ga_instance(
        distances: Matrix[int],
        optimum: List[int]) -> GeneticSolution:
    number_generations: int = 300
    population_size: int = 500
    crossover_rate: float = 0.3
    mutation_rate: float = 0.1
    number_parents: int = 200
    number_mutations: int = 6
    verbose: bool = True
    minimum_fitness: int = fitness(optimum[:-1], distances)

    solver: GeneticAlgorithm = GeneticAlgorithm(
        distances, optimum, selection_function=select_parents,
        crossover_function=single_point_crossover, mutation_function=mutate,
        fitness_function=fitness,
        generation_function=generate_population)

    solution, profit, history = solver(
        number_generations, population_size, crossover_rate, mutation_rate,
        number_parents, number_mutations, minimum_fitness, verbose)

    return solution, profit, history


def running_greedy_instance(distances: Matrix[int]) -> GreedySolution:
    greedy = Greedy(distances)

    solution, profit = greedy()

    return solution, profit


def main():
    distances_file_path: str = path.join("dataset", "distances.txt")
    optimum_file_path: str = path.join("dataset", "optimum.txt")
    xy_file_path: str = path.join("dataset", "xy.txt")

    distances, optimum, xy = utils.get_dataset(
        distances_file_path, optimum_file_path, xy_file_path)

    # solution, _, history = running_ga_instance(distances, optimum)
    solution, profit = running_greedy_instance(distances)

    gmt = time.gmtime()
    timestamp = calendar.timegm(gmt)

    gif_directory: str = path.join("gifs", f"{timestamp}")

    # utils.plot_best_mean_fitness(history, gif_directory)

    # title: str = "Best Route Found Throughout the Generations"
    title: str = "Greedy Route"
    utils.plot_route(solution, xy, gif_directory, title)

    # utils.create_best_route_gif(history["best_solution"], xy, gif_directory)

    # input_directory: str = path.join(gif_directory, "images")
    # output_directory: str = path.join(gif_directory, "animation.gif")
    # utils.create_gif(output_directory, input_directory)


if __name__ == "__main__":
    main()
