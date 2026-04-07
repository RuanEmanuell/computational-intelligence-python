# Perceptron - Letters A and B (7x7)

def flatten(matrix):
    vector = []
    
    for i in range(7):
        for j in range(7):
            vector.append(matrix[i][j])
    
    return vector


def train_perceptron(name, inputs, targets):
    # 49 entries in each letter (7x7)
    n = len(inputs[0])  
    
    weights = [0] * n
    bias = 0
    
    eta = 1.0
    epochs = 20
    
    print(f"\n ==== LETTERS {name} - PERCEPTRON ===== \n")
    
    for j in range(epochs):
        total_error = 0
        
        for i in range(len(inputs)):
            x = inputs[i]
            y = targets[i]
            
            amount = 0
            
            for k in range(n):
                amount = amount + (x[k] * weights[k])
            
            amount = amount + bias
            
            prediction = 0
            
            if amount >= 0:
                prediction = 1
            else:
                prediction = -1
            
            error = y - prediction
            
            if error != 0:
                for k in range(n):
                    weights[k] = weights[k] + eta * error * x[k]
                
                bias = bias + eta * error
                total_error = total_error + 1
        
        if total_error == 0:
            break
    
    
    print("Final weights:", weights)
    print("Final bias:", bias)
        
    print("\n Test:")

    for i in range(len(inputs)):
        amount = 0
        
        for k in range(n):
            amount = amount + (inputs[i][k] * weights[k])
        
        amount = amount + bias
        
        prediction = 0
        
        if amount >= 0:
            prediction = 1
        else:
            prediction = -1
        
        print(f"Letter {labels[i]} | Target: {targets[i]} | Prediction: {prediction}")


# 7x7 Letters (1 = pixel on, -1 = pixel off)

A = [
 [-1,  1,  1,  1,  1,  1, -1],
 [ 1, -1, -1, -1, -1, -1,  1],
 [ 1, -1, -1, -1, -1, -1,  1],
 [ 1,  1,  1,  1,  1,  1,  1],
 [ 1, -1, -1, -1, -1, -1,  1],
 [ 1, -1, -1, -1, -1, -1,  1],
 [ 1, -1, -1, -1, -1, -1,  1],
]

B = [
 [ 1,  1,  1,  1,  1, -1, -1],
 [ 1, -1, -1, -1, -1,  1, -1],
 [ 1, -1, -1, -1, -1,  1, -1],
 [ 1,  1,  1,  1,  1, -1, -1],
 [ 1, -1, -1, -1, -1,  1, -1],
 [ 1, -1, -1, -1, -1,  1, -1],
 [ 1,  1,  1,  1,  1, -1, -1],
]


inputs = [
    flatten(A),
    flatten(B)
]

targets = [1, -1]

labels = ["A", "B"]

train_perceptron("A vs B", inputs, targets)