from typing import Optional
from datatypes import Genome, History, Matrix, Parents, Population, GeneticSolution


class GeneticAlgorithm:
    def __init__(
            self, distances: Matrix[int],
            optimum: Genome, selection_function, crossover_function,
            mutation_function, fitness_function, generation_function) -> None:
        self.distances: Matrix[int] = distances
        self.optimum: Genome = optimum
        self.selection_function = selection_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.fitness_function = fitness_function
        self.generation_function = generation_function

    def __call__(
            self, number_generations: int, population_size: int,
            crossover_rate: float, mutation_rate: float, number_parents: int,
            number_mutations: int, minimum_fitness: int, verbose: bool) -> GeneticSolution:
        history: History = {
            "best_fitness": [],
            "mean_fitness": [],
            "best_solution": []}

        genome_size: int = len(self.distances)
        population: Population = self.generation_function(
            population_size, genome_size)

        for generation in range(number_generations):
            population = sorted(
                population, key=lambda genome: self.fitness_function(
                    genome, self.distances))

            fitnesses = [
                self.fitness_function(genome, self.distances)
                for genome in population]
            best_fitness = min(fitnesses)
            mean_fitness = sum(fitnesses) / population_size

            history["best_fitness"].append(best_fitness)
            history["mean_fitness"].append(mean_fitness)
            history["best_solution"].append(population[0])

            if verbose:
                print(
                    f"[{generation + 1}/{number_generations}]: best fitness is {best_fitness} and mean fitness is {mean_fitness}")

            if self.fitness_function(
                    population[0],
                    self.distances) <= minimum_fitness:
                break

            parents: Parents = self.selection_function(
                population, number_parents, self.fitness_function, self.distances)

            offsprings: Population = []

            for parent_1, parent_2 in parents:
                offspring_1: Genome = self.crossover_function(
                    parent_1, parent_2, crossover_rate)

                offspring_2: Genome = self.crossover_function(
                    parent_1, parent_2, crossover_rate)

                if offspring_1 is not None:
                    offspring_1 = self.mutation_function(
                        offspring_1, number_mutations, mutation_rate)

                    offsprings += [offspring_1]

                if offspring_2 is not None:
                    offspring_2 = self.mutation_function(
                        offspring_2, number_mutations, mutation_rate)

                    offsprings += [offspring_2]

            population = sorted(population + offsprings,
                                key=lambda genome: self.fitness_function(
                                    genome, self.distances))[:population_size]

        solution_genome: Genome = population[0]
        profit: int = self.fitness_function(solution_genome, self.distances)

        solution: GeneticSolution = solution_genome, profit, history

        return solution
