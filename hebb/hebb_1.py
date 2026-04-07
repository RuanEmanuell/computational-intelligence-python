# Hebb 1 - Base algorithm

inputs = [
    [1,   1],
    [1,  -1],
    [-1,  1],
    [-1, -1]
]

targets = [1, -1, -1, -1]

w1 = 0
w2 = 0
bias = 0

print("Starting training - Hebb Rule")

for i in range (4):
    x1 = inputs[i][0]
    x2 = inputs[i][1]
    y = targets[i]
    
    w1 = w1 + (x1 * y)
    w2 = w2 + (x2 * y)
    bias = bias + y 
    
    print("Updated weights: ", w1, w2, bias)
    
print("\n Final test")

for i in range (4):
    amount = (inputs[i][0] * w1) + (inputs[i][1] * w2) + bias
    prediction = 0
    
    if amount >= 0: 
        prediction = 1
    else: 
        prediction = -1
    
    print(f"Input: [{inputs[i][0]}, {inputs[i][1]}] | Target: {targets[i]} | Prediction: {prediction}")