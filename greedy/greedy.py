from typing import List
from datatypes import GreedySolution, Matrix


class Greedy:
    def __init__(self, distances: Matrix[int]) -> None:
        self.distances: Matrix[int] = distances

        number_cities: int = len(distances)
        self.cities: List[int] = list(range(1, number_cities + 1))

    def __call__(self) -> GreedySolution:
        unvisited: List[int] = self.cities[1:]
        visited: List[int] = self.cities[0:1]

        distance_summation: int = 0

        while unvisited:
            current_location: int = visited[-1]

            destinations = list(enumerate(self.distances[current_location - 1]))

            potential_locations = list(
                filter(
                    lambda
                    destination: (destination[0] + 1)
                    not in visited, destinations))

            index, distance = min(potential_locations,
                                  key=lambda destination: destination[1])

            distance_summation += distance

            city: int = index + 1
            visited.append(city)

            city_index: int = unvisited.index(city)
            unvisited.pop(city_index)

        return visited, distance_summation
