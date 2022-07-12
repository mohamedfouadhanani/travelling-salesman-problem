from typing import List
import matplotlib.pyplot as plt
from os import path
import os
import imageio
import shutil

Genome = List[int]
Population = List[Genome]


def get_dataset(distances_file_path: str, optimum_file_path: str,
                xy_file_path: str):
    distances: List[List[int]] = []
    with open(distances_file_path, "r") as file:
        file_content_lines: List[str] = file.readlines()
        for file_content_line in file_content_lines:
            file_content_line = file_content_line.replace("\n", "")
            file_content_line_list: List[str] = file_content_line.split(" ")
            file_content_line_list_values: List[int] = [
                int(value) for value in file_content_line_list]

            distances.append(file_content_line_list_values)

    with open(optimum_file_path, "r") as file:
        file_content = file.readline()
        file_content_list = file_content.split(" ")
        optimum = [int(value) for value in file_content_list]

    xy = []
    with open(xy_file_path, "r") as file:
        file_content_lines = file.readlines()
        for file_content_line in file_content_lines:
            file_content_line = file_content_line.replace("\n", "")
            file_content_line_list: List[str] = file_content_line.split(" ")
            file_content_line_list_values: List[int] = [
                float(value) for value in file_content_line_list]

            xy.append(file_content_line_list_values)

    return distances, optimum, xy


def plot_best_mean_fitness(history):
    X = list(range(len(history["best_fitness"])))

    plt.plot(X, history["best_fitness"], color="blue", label="Best fitness")
    plt.plot(X, history["mean_fitness"], color="orange", label="Mean fitness")
    plt.title("Best & Mean fitness")
    plt.xlabel("Generations")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()


def plot_route(route: List[int], xy, output_directory: str):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    route.append(route[0])

    X = []
    Y = []

    for location in route:
        X.append(xy[location - 1][0])
        Y.append(xy[location - 1][1])

    plt.plot(X, Y, color="red")
    plt.scatter(X, Y, color="red")
    plt.title("Best route found throughout the generations")
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    full_path = path.join(output_directory, "ideal_route.png")
    plt.savefig(full_path)


def create_best_route_gif(best_routes, xy, directory_path: str):
    images_directory: str = path.join(directory_path, "images")

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if not os.path.exists(images_directory):
        os.makedirs(images_directory)

    number_generations: int = len(best_routes)

    for index, route in enumerate(best_routes):
        print(f"[{index + 1}/{number_generations}]: processing...")
        X = []
        Y = []

        circular_route = route + route[:1]

        for location in circular_route:
            X.append(xy[location - 1][0])
            Y.append(xy[location - 1][1])

        colour = "red" if index == 0 else "blue"

        plt.plot(X, Y, color=colour)
        plt.scatter(X, Y, color=colour)
        plt.title("Evolution of the Best Route throughout the generations")
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")

        full_path: str = path.join(images_directory, f"{index}.png")

        plt.savefig(full_path)
        plt.clf()
        print(f"[{index + 1}/{number_generations}]: done processing...")

        X.clear()
        Y.clear()


def create_gif(output_directory: str, input_directory: str):
    filenames = [
        filename for filename in os.listdir(input_directory)
        if os.path.isfile(path.join(input_directory, filename))]

    filenames = sorted(filenames, key=lambda
                       filename: int(filename.split(".")[0]))

    number_files: int = len(filenames)

    with imageio.get_writer(output_directory, mode='I', duration=0.1) as writer:
        for index, filename in enumerate(filenames):
            print(f"[{index + 1}/{number_files}]: animation processing...")
            full_path: str = path.join(input_directory, filename)
            image = imageio.imread(full_path)
            writer.append_data(image)
            print(f"[{index + 1}/{number_files}]: done animation processing...")

    shutil.rmtree(input_directory)
