# Perceptron 2 - All logic gates tested

def train_perceptron(name, inputs, targets):
    w1 = 0
    w2 = 0
    bias = 0
    
    eta = 1.0
    epochs = 10
    
    print(f"\n ==== PORTA {name} - PERCEPTRON ===== \n")
    
    for j in range(epochs):
        total_error = 0
        
        for i in range(4):
            x1 = inputs[i][0]
            x2 = inputs[i][1]
            y = targets[i]
            
            amount = x1 * w1 + x2 * w2 + bias 
            prediction = 0
            
            if amount >= 0: 
                prediction = 1
            else: 
                prediction = -1
            
            error = y - prediction
            
            if error != 0:
                w1 = w1 + eta * error * x1 
                w2 = w2 + eta * error * x2 
                bias = bias + eta * error 
                total_error = total_error + 1
                
        if total_error == 0:
            break
    
            
    print("Updated weights: ", w1, w2, bias)
        
    print("\n Test:")

    for i in range (4):
        amount = (inputs[i][0] * w1) + (inputs[i][1] * w2) + bias
        
        prediction = 0
            
        if amount > 0: 
            prediction = 1
        else: 
            prediction = -1
        
        print(f"Input: [{inputs[i][0]}, {inputs[i][1]}] | Target: {targets[i]} | Prediction: {prediction}")

inputs = [
    [1,   1],
    [1,  -1],
    [-1,  1],
    [-1, -1]
]

# Logic gates
AND  = [ 1, -1, -1, -1]
OR   = [ 1,  1,  1, -1] 
NAND = [-1,  1,  1,  1]
NOR  = [-1, -1, -1,  1]   

train_perceptron("AND",  inputs, AND )
train_perceptron("OR",   inputs, OR  )
train_perceptron("NAND", inputs, NAND)
train_perceptron("NOR",  inputs, NOR )
