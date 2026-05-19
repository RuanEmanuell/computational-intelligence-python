# MLP - REPRODUÇÃO DO EXEMPLO DOS SLIDES

import math
import random

def activation_func(x):
    return math.tanh(x)

def activation_derivative(x):
    return 1.0 - math.tanh(x) ** 2

# Dados do problema (com bias = 1 incluído na primeira posição)
X = [
    [1,  1,  0.5, -1],
    [1,  0,  0.5,  1],
    [1,  1, -0.5, -1]
]

T = [
    [ 1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1]
]

LEARNING_RATE = 0.01
NUM_INPUTS = 3
NUM_HIDDEN = 2
NUM_OUTPUTS = 3
MAX_EPOCHS = 5000
ERROR_THRESHOLD = 0.001

random.seed(42)

# Inicialização dos pesos entre -0.5 e 0.5
W_hidden = [[random.uniform(-0.5, 0.5) for _ in range(NUM_HIDDEN)] for _ in range(NUM_INPUTS + 1)]
W_output = [[random.uniform(-0.5, 0.5) for _ in range(NUM_OUTPUTS)] for _ in range(NUM_HIDDEN + 1)]

print("A treinar a MLP...")

for epoch in range(1, MAX_EPOCHS + 1):
    total_quadratic_error = 0.0
    
    for sample_idx in range(len(X)):
        x_sample = X[sample_idx]
        t_sample = T[sample_idx]
        
        # FASE FORWARD
        net_hidden = [0.0] * NUM_HIDDEN
        z_hidden = [1.0] * (NUM_HIDDEN + 1)
        
        for j in range(NUM_HIDDEN):
            soma = W_hidden[0][j] * x_sample[0]
            for i in range(1, NUM_INPUTS + 1):
                soma += W_hidden[i][j] * x_sample[i]
            net_hidden[j] = soma
            z_hidden[j + 1] = activation_func(soma)
            
        net_output = [0.0] * NUM_OUTPUTS
        y_output = [0.0] * NUM_OUTPUTS
        
        for k in range(NUM_OUTPUTS):
            soma = W_output[0][k] * z_hidden[0]
            for j in range(1, NUM_HIDDEN + 1):
                soma += W_output[j][k] * z_hidden[j]
            net_output[k] = soma
            y_output[k] = activation_func(soma)
            
        for k in range(NUM_OUTPUTS):
            total_quadratic_error += 0.5 * ((t_sample[k] - y_output[k]) ** 2)
            
        # FASE BACKWARD
        delta_output = [0.0] * NUM_OUTPUTS
        for k in range(NUM_OUTPUTS):
            erro_k = t_sample[k] - y_output[k]
            delta_output[k] = erro_k * activation_derivative(net_output[k])
            
        delta_hidden = [0.0] * NUM_HIDDEN
        for j in range(NUM_HIDDEN):
            soma_deltas_pesos = 0.0
            for k in range(NUM_OUTPUTS):
                soma_deltas_pesos += delta_output[k] * W_output[j + 1][k]
            delta_hidden[j] = soma_deltas_pesos * activation_derivative(net_hidden[j])
            
        for k in range(NUM_OUTPUTS):
            W_output[0][k] += LEARNING_RATE * delta_output[k] * z_hidden[0]
            for j in range(1, NUM_HIDDEN + 1):
                W_output[j][k] += LEARNING_RATE * delta_output[k] * z_hidden[j]
                
        for j in range(NUM_HIDDEN):
            W_hidden[0][j] += LEARNING_RATE * delta_hidden[j] * x_sample[0]
            for i in range(1, NUM_INPUTS + 1):
                W_hidden[i][j] += LEARNING_RATE * delta_hidden[j] * x_sample[i]

    if epoch % 500 == 0 or epoch == 1:
        print(f"Ciclo {epoch} -> Erro Quadrático Total: {total_quadratic_error:.6f}")
        
    if total_quadratic_error <= ERROR_THRESHOLD:
        print(f"\nTreino concluído no ciclo {epoch}! Erro final: {total_quadratic_error:.6f}")
        break
else:
    print(f"\nTreino terminado ao atingir o limite de {MAX_EPOCHS} ciclos.")

# TESTE DA REDE 
print("\n--- TESTE APÓS TREINO ---")
for sample_idx in range(len(X)):
    x_sample = X[sample_idx]
    
    z_hidden = [1.0] * (NUM_HIDDEN + 1)
    for j in range(NUM_HIDDEN):
        soma = W_hidden[0][j] * x_sample[0]
        for i in range(1, NUM_INPUTS + 1):
            soma += W_hidden[i][j] * x_sample[i]
        z_hidden[j + 1] = activation_func(soma)
        
    y_output = [0.0] * NUM_OUTPUTS
    for k in range(NUM_OUTPUTS):
        soma = W_output[0][k] * z_hidden[0]
        for j in range(1, NUM_HIDDEN + 1):
            soma += W_output[j][k] * z_hidden[j]
        y_output[k] = activation_func(soma)
        
    resultado_classes = [1 if y > 0 else -1 for y in y_output]
    print(f"Padrão {sample_idx + 1} (Entradas: {x_sample[1:]}):")
    print(f"  -> Saídas Brutas: {[round(y, 4) for y in y_output]}")
    print(f"  -> Alvo Esperado: {T[sample_idx]}")
    print(f"  -> Classificação: {resultado_classes}")