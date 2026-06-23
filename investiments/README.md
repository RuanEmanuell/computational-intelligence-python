# Investments with MLP

# See the Portuguese version <a href="README-ptbr.md">here</a>

## What is this challenge?
This folder contains a MultiLayer Perceptron (MLP) implementation to learn the behavior of a Brazilian stock and predict the next day closing price.

The idea follows a time-series setting:
- Use older data for training
- Use the most recent 7 trading days for validation

## Implementation
This project supports two data modes:
- **Local CSV mode**: load a CSV from `investiments/data/`
- **Fallback mode**: automatically download data with `yfinance` (default ticker: `PETR4.SA`)

## Presentation
A presentation explaining the concept and implementation is also included:

<img width="1920" height="1080" alt="1" src="https://github.com/user-attachments/assets/fafee24a-1e55-495f-b9d6-194d8a5bf7a1" />
<img width="1920" height="1080" alt="2" src="https://github.com/user-attachments/assets/5e992870-d19b-4b06-9f47-d3f3c14763ab" />
<img width="1920" height="1080" alt="3" src="https://github.com/user-attachments/assets/fa08abb6-f738-4b8c-aea1-b3bd463ec68d" />
