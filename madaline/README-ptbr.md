# Madaline

# Veja a versão em Inglês <a href="README.md">aqui</a>

## O que é uma Madaline?
A **Madaline (Multiple ADAptive LInear NEuron)** é um modelo de rede neural artificial utilizado para reconhecimento de padrões e classificação multiclasse.

Ela é uma evolução do Perceptron, utilizando múltiplos neurônios para classificar mais de duas classes.

Ela aprende ajustando os pesos com base no erro entre a saída prevista e a saída esperada:
- Se a previsão estiver correta, nenhum ajuste é feito
- Se estiver errada, os pesos são atualizados para reduzir o erro

## Implementação
Este projeto implementa uma rede neural Madaline para reconhecer letras de **A a Z**.

A implementação inclui:
- Reconhecimento de padrões usando **matrizes 7x7**
- Conversão de pixels em valores numéricos (`1` e `-1`)
- Classificação multiclasse (26 letras)
- Treinamento baseado em correção de erro

## Como funciona?
- Cada letra é representada como uma **matriz 7x7 (49 entradas)**
- Os valores são convertidos para:
  - `1` → pixel ativo  
  - `-1` → pixel inativo  
- A rede possui:
  - **49 entradas**
  - **26 saídas (uma para cada letra)**

Durante o treinamento:
- A letra correta recebe **+1**
- As demais recebem **-1**
- Os pesos são ajustados apenas quando há erro

Após o treinamento, a rede prevê a letra com base no maior valor de ativação.

## Apresentação
Uma apresentação explicando o conceito e a implementação também está incluída:

<img width="1920" height="1080" alt="1" src="https://github.com/user-attachments/assets/8d96b318-5705-4b73-bd95-8f497d157dcd" />
<img width="1920" height="1080" alt="2" src="https://github.com/user-attachments/assets/859434af-7e30-4548-835c-54a81a49a9cc" />
<img width="1920" height="1080" alt="3" src="https://github.com/user-attachments/assets/6c2f8b95-c4d3-49d8-84d8-1c57784de96a" />
<img width="1920" height="1080" alt="4" src="https://github.com/user-attachments/assets/a667f9ce-1f9e-4b76-ae9c-5fba43e8848a" />
<img width="1920" height="1080" alt="5" src="https://github.com/user-attachments/assets/aa6c7725-ea7a-4f1c-9601-d81a7b0f4841" />