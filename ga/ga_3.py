# Trabalho 10 - AG com horarios
# Monta uma grade escolar usando cromossomo em matriz (horarios x turmas)

import random


NUMERO_DE_PROFESSORES = 29
NUMERO_DE_TURMAS = 3
AULAS_POR_DIA = 5
DIAS_DA_SEMANA = 2
TOTAL_DE_AULAS = AULAS_POR_DIA * DIAS_DA_SEMANA

TAMANHO_POPULACAO = 100
CICLOS = 3000
TAXA_CRUZAMENTO = 0.85
TAXA_MUTACAO = 0.10
TAMANHO_TORNEIO = 3
ELITE_SIZE = 2

# Pesos da aptidao
PENALIDADE_CHOQUE = 8.0
BONUS_REPETICAO_ATE_2 = 3.0
PENALIDADE_REPETICAO_EXCESSIVA = 4.0
PENALIDADE_DESBALANCEAMENTO = 1.0


def create_random_schedule(total_aulas, num_turmas, num_professores):
    return [
        [random.randint(1, num_professores) for _ in range(num_turmas)]
        for _ in range(total_aulas)
    ]


def create_initial_population(population_size, total_aulas, num_turmas, num_professores):
    return [
        create_random_schedule(total_aulas, num_turmas, num_professores)
        for _ in range(population_size)
    ]


def fitness(schedule):
    score = 0.0

    # Penaliza choque: mesmo professor em duas turmas no mesmo horario.
    for horario in range(TOTAL_DE_AULAS):
        usage = {}
        for turma in range(NUMERO_DE_TURMAS):
            professor = schedule[horario][turma]
            usage[professor] = usage.get(professor, 0) + 1

        for count in usage.values():
            if count > 1:
                score -= PENALIDADE_CHOQUE * (count - 1)

    # Bonus para repeticoes curtas (ate 2 seguidas) por turma.
    for turma in range(NUMERO_DE_TURMAS):
        current_professor = schedule[0][turma]
        run_length = 1

        for horario in range(1, TOTAL_DE_AULAS):
            professor = schedule[horario][turma]
            if professor == current_professor:
                run_length += 1
            else:
                if run_length == 2:
                    score += BONUS_REPETICAO_ATE_2
                elif run_length > 2:
                    score -= PENALIDADE_REPETICAO_EXCESSIVA * (run_length - 2)
                current_professor = professor
                run_length = 1

        if run_length == 2:
            score += BONUS_REPETICAO_ATE_2
        elif run_length > 2:
            score -= PENALIDADE_REPETICAO_EXCESSIVA * (run_length - 2)

    # Balanceamento de carga entre professores.
    professor_load = {prof: 0 for prof in range(1, NUMERO_DE_PROFESSORES + 1)}
    for horario in schedule:
        for professor in horario:
            professor_load[professor] += 1

    expected_load = (NUMERO_DE_TURMAS * TOTAL_DE_AULAS) / float(NUMERO_DE_PROFESSORES)
    for load in professor_load.values():
        score -= PENALIDADE_DESBALANCEAMENTO * abs(load - expected_load)

    return score


def selection_tournament(population):
    def tournament():
        candidates = random.sample(population, min(TAMANHO_TORNEIO, len(population)))
        return max(candidates, key=fitness)

    return tournament(), tournament()


def crossover_row_swap(parent1, parent2, crossover_prob):
    num_rows = len(parent1)
    if random.random() > crossover_prob or num_rows < 2:
        return [row[:] for row in parent1], [row[:] for row in parent2]

    cut = random.randint(1, num_rows - 1)

    child1 = [row[:] for row in parent1[:cut]] + [row[:] for row in parent2[cut:]]
    child2 = [row[:] for row in parent2[:cut]] + [row[:] for row in parent1[cut:]]

    return child1, child2


def mutate(schedule, mutation_prob):
    mutated = [row[:] for row in schedule]
    if random.random() > mutation_prob:
        return mutated

    row = random.randint(0, len(mutated) - 1)
    col = random.randint(0, len(mutated[0]) - 1)

    current = mutated[row][col]
    new_prof = random.randint(1, NUMERO_DE_PROFESSORES)
    while NUMERO_DE_PROFESSORES > 1 and new_prof == current:
        new_prof = random.randint(1, NUMERO_DE_PROFESSORES)

    mutated[row][col] = new_prof
    return mutated


def print_schedule(schedule):
    print("\nGrade (horarios x turmas):")
    for idx, row in enumerate(schedule, start=1):
        dia = ((idx - 1) // AULAS_POR_DIA) + 1
        aula = ((idx - 1) % AULAS_POR_DIA) + 1
        row_str = " ".join(f"T{turma + 1}:P{prof:02d}" for turma, prof in enumerate(row))
        print(f"Dia {dia} Aula {aula}: {row_str}")


def genetic_algorithm():
    print("\n" + "=" * 60)
    print("INICIANDO AG PARA GRADE DE PROFESSORES")
    print("=" * 60)
    print(f"Professores: {NUMERO_DE_PROFESSORES}")
    print(f"Turmas: {NUMERO_DE_TURMAS}")
    print(f"Aulas por dia: {AULAS_POR_DIA}")
    print(f"Dias da semana: {DIAS_DA_SEMANA}")
    print(f"Total de aulas: {TOTAL_DE_AULAS}")
    print(f"Populacao: {TAMANHO_POPULACAO}")
    print(f"Ciclos: {CICLOS}")
    print(f"Taxa de mutacao: {TAXA_MUTACAO}")
    print(f"Taxa de cruzamento: {TAXA_CRUZAMENTO}")
    print("Selecao: Torneio")
    print("Crossover: troca de linhas entre matrizes")
    print("=" * 60 + "\n")

    population = create_initial_population(
        TAMANHO_POPULACAO,
        TOTAL_DE_AULAS,
        NUMERO_DE_TURMAS,
        NUMERO_DE_PROFESSORES,
    )

    best_schedule = max(population, key=fitness)
    best_fitness = fitness(best_schedule)

    for generation in range(CICLOS):
        new_population = []

        elite = sorted(population, key=fitness, reverse=True)[:ELITE_SIZE]
        new_population.extend([[row[:] for row in chromosome] for chromosome in elite])

        while len(new_population) < TAMANHO_POPULACAO:
            parent1, parent2 = selection_tournament(population)

            child1, child2 = crossover_row_swap(
                parent1,
                parent2,
                TAXA_CRUZAMENTO,
            )

            child1 = mutate(child1, TAXA_MUTACAO)
            child2 = mutate(child2, TAXA_MUTACAO)

            new_population.append(child1)
            if len(new_population) < TAMANHO_POPULACAO:
                new_population.append(child2)

        population = new_population[:TAMANHO_POPULACAO]

        current_best = max(population, key=fitness)
        current_best_fitness = fitness(current_best)

        if current_best_fitness > best_fitness:
            best_schedule = [row[:] for row in current_best]
            best_fitness = current_best_fitness

        if (generation + 1) % 10 == 0 or generation == 0:
            print(
                f"Geracao {generation + 1:3d} | "
                f"Melhor aptidao: {best_fitness:9.4f}"
            )

    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS")
    print("=" * 60)
    print(f"Melhor aptidao encontrada: {best_fitness:.4f}")
    print_schedule(best_schedule)

    return best_schedule, best_fitness


if __name__ == "__main__":
    random.seed(42)
    best_schedule, best_fitness = genetic_algorithm()
