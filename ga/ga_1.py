# Trabalho 08 - AG com função matemática	
# Encontra o mínimo de f(x) = -|x*sin(sqrt(|x|))| no intervalo [0, 512]

import random
import math

POPULATION_SIZE = 30
CHROMOSOME_BITS = 10
GENERATIONS = 100
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.8
ELITE_SIZE = 2

def binary_to_decimal(binary_string):
    decimal = int(binary_string, 2)
    return (decimal / 1023.0) * 512.0

def objective_function(x):
    return -abs(x * math.sin(math.sqrt(abs(x))))

def fitness(chromosome):
    x = binary_to_decimal(chromosome)
    return objective_function(x)

def create_random_chromosome():
    return ''.join(str(random.randint(0, 1)) for _ in range(CHROMOSOME_BITS))

def create_initial_population():
    return [create_random_chromosome() for _ in range(POPULATION_SIZE)]

def selection(population):
    fitness_values = [fitness(chromosome) for chromosome in population]
    
    min_fitness = min(fitness_values)
    max_fitness = max(fitness_values)
    
    if max_fitness == min_fitness:
        probabilities = [1 / POPULATION_SIZE] * POPULATION_SIZE
    else:
        inverted_fitness = [max_fitness - f for f in fitness_values]
        total_fitness = sum(inverted_fitness)
        probabilities = [f / total_fitness for f in inverted_fitness]
    
    parent1 = random.choices(population, weights=probabilities, k=1)[0]
    parent2 = random.choices(population, weights=probabilities, k=1)[0]
    
    return parent1, parent2

def crossover(parent1, parent2):
    if random.random() > CROSSOVER_PROBABILITY:
        return parent1, parent2
    
    point = random.randint(1, CHROMOSOME_BITS - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    
    return child1, child2

def mutate(chromosome):
    if random.random() > MUTATION_PROBABILITY:
        return chromosome
    
    chromosome_list = list(chromosome)
    mutation_point = random.randint(0, CHROMOSOME_BITS - 1)
    chromosome_list[mutation_point] = '0' if chromosome_list[mutation_point] == '1' else '1'
    
    return ''.join(chromosome_list)

def genetic_algorithm():
    
    print("=" * 60)
    print("ALGORITMO GENÉTICO - MINIMIZAÇÃO DE FUNÇÃO")
    print("=" * 60)
    print(f"Objetivo: minimizar f(x) = -|x*sin(sqrt(|x|))| em [0, 512]")
    print(f"Tamanho da população: {POPULATION_SIZE}")
    print(f"Bits por cromossomo: {CHROMOSOME_BITS}")
    print(f"Gerações: {GENERATIONS}")
    print(f"Probabilidade de mutação: {MUTATION_PROBABILITY}")
    print(f"Probabilidade de crossover: {CROSSOVER_PROBABILITY}")
    print("=" * 60)
    print()
    
    population = create_initial_population()
    best_chromosome = min(population, key=fitness)
    best_fitness = fitness(best_chromosome)
    
    fitness_history = [best_fitness]
    
    for generation in range(GENERATIONS):
        new_population = []
        
        elite = sorted(population, key=fitness)[:ELITE_SIZE]
        new_population.extend(elite)
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = selection(population)
            
            child1, child2 = crossover(parent1, parent2)
            
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
        
        population = new_population[:POPULATION_SIZE]
        
        current_best = min(population, key=fitness)
        current_best_fitness = fitness(current_best)
        
        if current_best_fitness < best_fitness:
            best_chromosome = current_best
            best_fitness = current_best_fitness
        
        fitness_history.append(best_fitness)
        
        if (generation + 1) % 10 == 0 or generation == 0:
            x_best = binary_to_decimal(best_chromosome)
            print(f"Geração {generation + 1:3d} | Melhor x: {x_best:7.3f} | Melhor f(x): {best_fitness:9.6f}")
    
    print()
    print("=" * 60)
    print("RESULTADOS FINAIS")
    print("=" * 60)
    
    x_best = binary_to_decimal(best_chromosome)
    print(f"Melhor cromossomo (binário): {best_chromosome}")
    print(f"Melhor valor de x: {x_best:.6f}")
    print(f"Melhor valor de f(x): {best_fitness:.6f}")
    print(f"Mínimo encontrado: {best_fitness:.6f}")
    
    return best_chromosome, x_best, best_fitness

if __name__ == "__main__":
    random.seed(42)
    best_chromosome, x_best, best_fitness = genetic_algorithm()
