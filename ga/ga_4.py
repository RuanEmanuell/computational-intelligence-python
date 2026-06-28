# Trabalho 11 - Caixeiro viajante
# Resolve o problema do caixeiro viajante com AG multipopulacional e threads

import random
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

CITIES_FILE = "data/cities.xlsx"
UNREACHABLE_DISTANCE = 999999
SEED = 42

TOTAL_POPULATION = 240
NUM_ISLANDS = 4
GENERATIONS = 500
MUTATION_RATE = 0.20
TOURNAMENT_SIZE = 5
ELITE_SIZE = 2

MIGRATION_INTERVAL = 50
MIGRANTS_PER_ISLAND = 2


def load_distance_dataframe():
    local_path = Path(__file__).resolve().parent / CITIES_FILE
    if local_path.exists():
        return pd.read_excel(local_path, index_col=0)

    # Backward compatible fallback if the user keeps the file in /data.
    return pd.read_excel("/data/cities.xlsx", index_col=0)


distance_df = load_distance_dataframe()

cities = list(distance_df.index)
NUM_CITIES = len(cities)

for city in cities:
    distance_df.loc[city, city] = 0

initial_matrix = distance_df.fillna(UNREACHABLE_DISTANCE).to_numpy().astype(float)

distance_matrix = initial_matrix.copy()
for k in range(NUM_CITIES):
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j]:
                distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]


def calculate_route_distance(route):
    total_distance = 0
    for i in range(NUM_CITIES - 1):
        origin = route[i]
        destination = route[i + 1]
        total_distance += distance_matrix[origin][destination]

    # Fecha o ciclo do caixeiro voltando para a cidade inicial.
    total_distance += distance_matrix[route[-1]][route[0]]
    return total_distance


def create_individual(rng):
    route = list(range(NUM_CITIES))
    rng.shuffle(route)
    return route


def tournament_selection(population, population_distances, rng, k=4):
    selected = rng.sample(list(zip(population, population_distances)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]


def ordered_crossover(parent1, parent2, rng):
    point1, point2 = sorted(rng.sample(range(NUM_CITIES), 2))
    child = [None] * NUM_CITIES
    child[point1 : point2 + 1] = parent1[point1 : point2 + 1]

    child_pos = (point2 + 1) % NUM_CITIES
    parent2_pos = (point2 + 1) % NUM_CITIES

    while None in child:
        gene = parent2[parent2_pos]
        if gene not in child:
            child[child_pos] = gene
            child_pos = (child_pos + 1) % NUM_CITIES
        parent2_pos = (parent2_pos + 1) % NUM_CITIES

    return child


def swap_mutation(route, rng, mutation_rate=0.05):
    if rng.random() < mutation_rate:
        idx1, idx2 = rng.sample(range(NUM_CITIES), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route


def initialize_islands(total_population, num_islands, seed):
    island_size = total_population // num_islands
    islands = []
    island_rngs = []

    for island_id in range(num_islands):
        rng = random.Random(seed + island_id)
        population = [create_individual(rng) for _ in range(island_size)]
        islands.append(population)
        island_rngs.append(rng)

    return islands, island_rngs


def evolve_island(
    island_population,
    rng,
    steps,
    mutation_rate,
    tournament_size,
    elite_size,
):
    population = [[gene for gene in route] for route in island_population]

    best_route = None
    best_distance = float("inf")

    for _ in range(steps):
        distances = [calculate_route_distance(individual) for individual in population]

        current_best_distance = min(distances)
        current_best_route = population[distances.index(current_best_distance)].copy()
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = current_best_route

        ranked = sorted(zip(population, distances), key=lambda x: x[1])
        new_population = [route.copy() for route, _ in ranked[:elite_size]]

        while len(new_population) < len(population):
            parent1 = tournament_selection(population, distances, rng, k=tournament_size)
            parent2 = tournament_selection(population, distances, rng, k=tournament_size)

            child = ordered_crossover(parent1, parent2, rng)
            child = swap_mutation(child, rng, mutation_rate)
            new_population.append(child)

        population = new_population

    return population, best_route, best_distance


def migrate_islands(islands):
    ranked_islands = []
    for population in islands:
        distances = [calculate_route_distance(individual) for individual in population]
        ranked = sorted(zip(population, distances), key=lambda x: x[1])
        ranked_islands.append(ranked)

    migrants_from_island = []
    for ranked in ranked_islands:
        migrants = [route.copy() for route, _ in ranked[:MIGRANTS_PER_ISLAND]]
        migrants_from_island.append(migrants)

    for island_idx in range(len(islands)):
        target_idx = (island_idx + 1) % len(islands)

        worst_positions = sorted(
            range(len(ranked_islands[target_idx])),
            key=lambda i: ranked_islands[target_idx][i][1],
            reverse=True,
        )[:MIGRANTS_PER_ISLAND]

        for replacement_pos, migrant in zip(worst_positions, migrants_from_island[island_idx]):
            route_to_replace = ranked_islands[target_idx][replacement_pos][0]
            real_index = islands[target_idx].index(route_to_replace)
            islands[target_idx][real_index] = migrant


def run_multipop_genetic_algorithm(
    total_population=TOTAL_POPULATION,
    num_islands=NUM_ISLANDS,
    generations=GENERATIONS,
    mutation_rate=MUTATION_RATE,
    tournament_size=TOURNAMENT_SIZE,
):
    islands, island_rngs = initialize_islands(total_population, num_islands, SEED)

    best_distance_ever = float("inf")
    best_route_ever = None

    completed_generations = 0
    total_epochs = (generations + MIGRATION_INTERVAL - 1) // MIGRATION_INTERVAL

    for epoch in range(total_epochs):
        steps = min(MIGRATION_INTERVAL, generations - completed_generations)

        with ThreadPoolExecutor(max_workers=num_islands) as executor:
            futures = [
                executor.submit(
                    evolve_island,
                    islands[i],
                    island_rngs[i],
                    steps,
                    mutation_rate,
                    tournament_size,
                    ELITE_SIZE,
                )
                for i in range(num_islands)
            ]

            results = [future.result() for future in futures]

        for island_idx, (new_population, island_best_route, island_best_distance) in enumerate(results):
            islands[island_idx] = new_population

            if island_best_distance < best_distance_ever:
                best_distance_ever = island_best_distance
                best_route_ever = island_best_route.copy()

        completed_generations += steps

        if completed_generations < generations:
            migrate_islands(islands)

        if completed_generations % 100 == 0 or completed_generations == steps:
            print(
                f"Geração {completed_generations:03d} | "
                f"Melhor distância linear: {best_distance_ever:.0f} km"
            )

    return best_route_ever, best_distance_ever


if __name__ == "__main__":
    print("Iniciando o AG Multipopulacional com Threads...")
    print(f"Ilhas: {NUM_ISLANDS} | População Total: {TOTAL_POPULATION} | Migração: a cada {MIGRATION_INTERVAL} gerações")

    best_route_indices, best_distance = run_multipop_genetic_algorithm()
    route_names = [cities[idx] for idx in best_route_indices]
    route_names_cycle = route_names + [route_names[0]]

    print("\n" + "=" * 60)
    print("RESULTADO FINAL (DISTÂNCIA REAL)")
    print("=" * 60)
    print(f"Distância Total Otimizada: {best_distance:.0f} km")
    print("\nSequência de Cidades (ciclo fechado):")
    print(" -> ".join(route_names_cycle))
    print("=" * 60)