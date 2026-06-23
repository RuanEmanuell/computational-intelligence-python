# Investimentos com MLP

# Veja a versão em Inglês <a href="README.md">aqui</a>

## O que é este desafio?
Esta pasta contém uma implementação de MultiLayer Perceptron (MLP) para aprender a dinâmica de uma ação brasileira e prever o preço de fechamento do dia seguinte.

A ideia segue um cenário de série temporal:
- Usar dados mais antigos para treino
- Usar os 7 dias mais recentes para validação

## Implementação
Este projeto suporta dois modos de dados:
- **Modo CSV local**: carrega um CSV em `investiments/data/`
- **Modo fallback**: baixa dados automaticamente com `yfinance` (ticker padrão: `PETR4.SA`)

## Apresentação
Uma apresentação explicando o conceito e a implementação também está incluída:

<img width="1920" height="1080" alt="Inglês" src="https://github.com/user-attachments/assets/ab0dcb52-9eea-46be-b16c-90a119741622" />
<img width="1920" height="1080" alt="Inglês (2)" src="https://github.com/user-attachments/assets/7f558026-f160-4d53-b33e-ddf71c33f8b3" />
<img width="1920" height="1080" alt="Inglês (3)" src="https://github.com/user-attachments/assets/370aa442-c2d2-41c5-9fe5-c1d2a8e16f52" />