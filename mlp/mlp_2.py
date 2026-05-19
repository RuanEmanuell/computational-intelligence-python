# MLP - Letters (7x7)

import tkinter as tk
import numpy as np

GRID_SIZE = 7

dataset = {

"A":[
0,1,1,1,1,1,0,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,1,1,1,1,1,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1
],

"B":[
1,1,1,1,1,0,0,
1,0,0,0,0,1,0,
1,0,0,0,0,1,0,
1,1,1,1,1,0,0,
1,0,0,0,0,1,0,
1,0,0,0,0,1,0,
1,1,1,1,1,0,0
],

"C":[
0,1,1,1,1,1,1,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
0,1,1,1,1,1,1
],

"D":[
1,1,1,1,1,0,0,
1,0,0,0,0,1,0,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,1,0,
1,1,1,1,1,0,0
],

"E":[
1,1,1,1,1,1,1,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,1,1,1,1,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,1,1,1,1,1,1
],

"F":[
1,1,1,1,1,1,1,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,1,1,1,1,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0
],

"G":[
0,1,1,1,1,1,1,
1,0,0,0,0,0,0,
1,0,0,1,1,1,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
0,1,1,1,1,1,1
],

"H":[
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,1,1,1,1,1,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1
],

"I":[
1,1,1,1,1,1,1,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0,
1,1,1,1,1,1,1
],

"J":[
0,0,0,1,1,1,1,
0,0,0,0,0,1,0,
0,0,0,0,0,1,0,
0,0,0,0,0,1,0,
1,0,0,0,0,1,0,
1,0,0,0,1,0,0,
0,1,1,1,0,0,0
],

"K":[
1,0,0,0,0,0,1,
1,0,0,0,1,0,0,
1,0,0,1,0,0,0,
1,1,1,0,0,0,0,
1,0,0,1,0,0,0,
1,0,0,0,1,0,0,
1,0,0,0,0,1,0
],

"L":[
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,1,1,1,1,1,1
],

"M":[
1,0,0,0,0,0,1,
1,1,0,0,0,1,1,
1,0,1,0,1,0,1,
1,0,0,1,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1
],

"N":[
1,0,0,0,0,0,1,
1,1,0,0,0,0,1,
1,0,1,0,0,0,1,
1,0,0,1,0,0,1,
1,0,0,0,1,0,1,
1,0,0,0,0,1,1,
1,0,0,0,0,0,1
],

"O":[
0,1,1,1,1,1,0,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
0,1,1,1,1,1,0
],

"P":[
1,1,1,1,1,0,0,
1,0,0,0,0,1,0,
1,0,0,0,0,1,0,
1,1,1,1,1,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0
],

"Q":[
0,1,1,1,1,1,0,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,1,0,0,1,
1,0,0,0,1,0,1,
0,1,1,1,1,1,1
],

"R":[
1,1,1,1,1,0,0,
1,0,0,0,0,1,0,
1,0,0,0,0,1,0,
1,1,1,1,1,0,0,
1,0,0,1,0,0,0,
1,0,0,0,1,0,0,
1,0,0,0,0,1,0
],

"S":[
0,1,1,1,1,1,1,
1,0,0,0,0,0,0,
1,0,0,0,0,0,0,
0,1,1,1,1,1,0,
0,0,0,0,0,0,1,
0,0,0,0,0,0,1,
1,1,1,1,1,1,0
],

"T":[
1,1,1,1,1,1,1,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0
],

"U":[
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
0,1,1,1,1,1,0
],

"V":[
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
0,1,0,0,0,1,0,
0,1,0,0,0,1,0,
0,0,1,1,1,0,0
],

"W":[
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,0,0,0,0,1,
1,0,1,0,1,0,1,
1,0,1,0,1,0,1,
1,1,0,0,0,1,1,
1,0,0,0,0,0,1
],

"X":[
1,0,0,0,0,0,1,
0,1,0,0,0,1,0,
0,0,1,0,1,0,0,
0,0,0,1,0,0,0,
0,0,1,0,1,0,0,
0,1,0,0,0,1,0,
1,0,0,0,0,0,1
],

"Y":[
1,0,0,0,0,0,1,
0,1,0,0,0,1,0,
0,0,1,0,1,0,0,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0,
0,0,0,1,0,0,0
],

"Z":[
1,1,1,1,1,1,1,
0,0,0,0,0,1,0,
0,0,0,0,1,0,0,
0,0,0,1,0,0,0,
0,0,1,0,0,0,0,
0,1,0,0,0,0,0,
1,1,1,1,1,1,1
]

}

LETRAS = list(dataset.keys())

for k in dataset:
    dataset[k] = [1 if v == 1 else -1 for v in dataset[k]]

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr

        self.W_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.b_hidden = np.random.uniform(-0.5, 0.5, hidden_size)
        
        self.W_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        self.b_output = np.random.uniform(-0.5, 0.5, output_size)

    def activation(self, x):
        # STangente Hiperbólica
        return np.tanh(x)

    def activation_derivative(self, x):
        # Derivada da Tangente Hiperbólica
        return 1.0 - np.tanh(x) ** 2

    def forward(self, x):
        # Propagação do sinal pelas camadas intermediária e de saída
        self.net_hidden = np.dot(self.W_hidden, x) + self.b_hidden
        self.z_hidden = self.activation(self.net_hidden)
        
        self.net_output = np.dot(self.W_output, self.z_hidden) + self.b_output
        self.y_output = self.activation(self.net_output)
        return self.y_output

    def train(self, X, Y, epochs=1500):
        print("Treinando a Rede MLP (Backpropagation)...")
        for epoch in range(epochs):
            total_error = 0
            for x, target in zip(X, Y):
                # Fase Forward
                y = self.forward(x)
                
                total_error += 0.5 * np.sum((target - y) ** 2)
                
                # Fase Backward
                error_output = target - y
                delta_output = error_output * self.activation_derivative(self.net_output)
                
                error_hidden = np.dot(self.W_output.T, delta_output)
                delta_hidden = error_hidden * self.activation_derivative(self.net_hidden)
                
                self.W_output += self.lr * np.outer(delta_output, self.z_hidden)
                self.b_output += self.lr * delta_output
                
                self.W_hidden += self.lr * np.outer(delta_hidden, x)
                self.b_hidden += self.lr * delta_hidden
            
            if epoch % 300 == 0:
                print(f"Época {epoch} -> Erro Quadrático Total: {total_error:.4f}")
                
            if total_error <= 0.01:
                print(f"Treino concluído com sucesso na época {epoch}!")
                break

X = []
Y = []

for i, letra in enumerate(LETRAS):
    X.append(dataset[letra])

    target = [-1] * len(LETRAS)
    target[i] = 1
    Y.append(target)

X = np.array(X)
Y = np.array(Y)

model = MLP(input_size=49, hidden_size=28, output_size=len(LETRAS), lr=0.01)
model.train(X, Y)

# Interface
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MLP A-Z (7x7)")

        self.grid_vars = []

        frame = tk.Frame(root)
        frame.pack()

        for i in range(GRID_SIZE):
            row = []
            for j in range(GRID_SIZE):
                var = tk.IntVar()
                cb = tk.Checkbutton(frame, variable=var)
                cb.grid(row=i, column=j)
                row.append(var)
            self.grid_vars.append(row)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Reconhecer", command=self.reconhecer).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Limpar", command=self.limpar).pack(side=tk.LEFT)

        self.label = tk.Label(root, text="Desenhe uma letra")
        self.label.pack()

    def get_input(self):
        x = np.array([var.get() for row in self.grid_vars for var in row])
        return np.where(x == 1, 1, -1)  

    def reconhecer(self):
        x = self.get_input()
        
        y_output = model.forward(x)
        
        distancias = []
        for target_letra in Y:
            dist = np.sqrt(np.sum((y_output - target_letra) ** 2))
            distancias.append(dist)
            
        letra = LETRAS[np.argmin(distancias)]
        self.label.config(text=f"Letra: {letra}")

    def limpar(self):
        for row in self.grid_vars:
            for var in row:
                var.set(0)

root = tk.Tk()
app = App(root)
root.mainloop()