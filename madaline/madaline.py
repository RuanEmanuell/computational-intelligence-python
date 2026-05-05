# Madaline - Letters (7x7)

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

class Madaline:
    def __init__(self, input_size, output_size, lr=0.1):
        self.lr = lr
        self.W = np.random.uniform(-0.5, 0.5, (output_size, input_size))
        self.b = np.random.uniform(-0.5, 0.5, output_size)

    def activation(self, x):
        return np.where(x >= 0, 1, -1)

    def predict(self, x):
        u = np.dot(self.W, x) + self.b
        return self.activation(u)

    def train(self, X, Y, epochs=500):
        for _ in range(epochs):
            for x, target in zip(X, Y):
                u = np.dot(self.W, x) + self.b
                y = self.activation(u)

                for i in range(len(target)):
                    if y[i] != target[i]:
                        self.W[i] += self.lr * target[i] * x
                        self.b[i] += self.lr * target[i]

# Preparar dados
X = []
Y = []

for i, letra in enumerate(LETRAS):
    X.append(dataset[letra])

    target = [-1] * len(LETRAS)
    target[i] = 1
    Y.append(target)

X = np.array(X)
Y = np.array(Y)

model = Madaline(input_size=49, output_size=len(LETRAS))
model.train(X, Y)

# Interface
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Madaline A-Z")

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

        scores = np.dot(model.W, x) + model.b
        letra = LETRAS[np.argmax(scores)]

        self.label.config(text=f"Letra: {letra}")

    def limpar(self):
        for row in self.grid_vars:
            for var in row:
                var.set(0)

# Rodar o app

root = tk.Tk()
app = App(root)
root.mainloop()