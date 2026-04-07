# Hebb 2 - All logic gates tested

def train_hebb(name, inputs, targets):
    w1 = 0
    w2 = 0
    bias = 0
    
    print(f"\n ==== LOGIC GATE {name} ===== \n")
    
    for i in range (4):
        x1 = inputs[i][0]
        x2 = inputs[i][1]
        y = targets[i]
        
        w1 = w1 + (x1 * y)
        w2 = w2 + (x2 * y)
        bias = bias + y 
        
        print("Updated weights: ", w1, w2, bias)
        
    print("\n Test:")

    for i in range (4):
        amount = (inputs[i][0] * w1) + (inputs[i][1] * w2) + bias
        prediction = 0
        
        if amount >= 0: 
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

print("Starting training - Hebb Rule (All logical gates)")

# Logic gates
AND  = [ 1, -1, -1, -1]
OR   = [ 1,  1,  1, -1] 
NAND = [-1,  1,  1,  1]
NOR  = [-1, -1, -1,  1]   

train_hebb("AND",  inputs, AND )
train_hebb("OR",   inputs, OR  )
train_hebb("NAND", inputs, NAND)
train_hebb("NOR",  inputs, NOR )
