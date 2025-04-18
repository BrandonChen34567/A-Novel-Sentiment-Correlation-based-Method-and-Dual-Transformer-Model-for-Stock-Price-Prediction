# A Novel Sentiment-Correlation-based Method and Dual Transformer Model for Stock Price Prediction

Welcome to the GitHub repository for the research paper "A Novel Sentiment-Correlation-based Method and Dual Transformer Model for Stock Price Prediction." This repository contains sample Python code to demonstrate the methodologies proposed in the paper. The provided code aims to guide researchers and practitioners in implementing and understanding the model components discussed in the paper.

## Overview
This research focuses on a proposed Dual Transformer architecture for predicting stock prices, leveraging sentiment analysis. The key contributions include:

1. **Sentiment-Correlation Analysis**: A novel method to assess the correlation between sentiment polarity of related companies and stock price movements of target company.
2. **Dual Transformer Model**: A two-step Transformer-based framework:
    - **Enhancement Transformer (Encoder)**: Processes sentiment polarity scores of multiple related companies.
    - **Forecast Transformer (Encoder + Decoder)**: Combines historical stock prices and outputs from Enhancement Transformer to predict future stock prices.

The proposed methodology outperforms traditional models such as Informer and LSTM in predictive accuracy.


## Repository Contents
- **Sample Code**: Example implementations of the Dual Transformer architecture and sentiment-correlation analysis.

## Data Preparation
- The news data is collected using the News Feed and News Sentiment data API.
- The stock data is collected using the yfinance Python library.

## Running the Code
The sample code is designed to be run in Google Colab. Follow these steps:
1. Open the Colab notebook
2. Upload your dataset to the Colab environment or link it via Google Drive.
3. Execute the Cells

## Requirements
1. **Python**: 3.10.12
2. **Pytorch**: 2.5.1
3. **Pandas**: 2.2.2
4. **NumPy**: 1.26.4

## Citation

@article{chen2025novel,
  title={A Novel Sentiment Correlation-based Method with Dual Transformer Model for Stock Price Prediction},
  author={Chen, Qizhao and Kawashima, Hiroaki},
  year={2025},
  journal={arXiv preprint arXiv:XXXX.XXXXX}
}


