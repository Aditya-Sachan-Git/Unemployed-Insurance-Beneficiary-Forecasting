# Unemployed Insurance Beneficiary Forecasting

A predictive analytics solution for forecasting unemployment insurance beneficiary counts, built as part of the Artificial Intelligence course from SmartBridge.

## 🚀 Introduction

"Unemployed Insurance Beneficiary Forecasting" is a predictive analytics project aimed at anticipating the number of individuals who will apply for and receive unemployment insurance benefits over a given period. By analyzing historical data, economic indicators, demographic trends, and labor market dynamics, this project seeks to provide accurate forecasts to government agencies, policymakers, and insurance providers, enabling them to allocate resources effectively and plan for future demand. The project leverages advanced time series analytics and machine learning to forecast the demand for unemployment insurance. It is designed to help government agencies, policymakers, and insurance providers with actionable insights for efficient resource allocation and planning.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Credits](#credits)

## Project Overview

- **Objective:** Predict the future number of unemployment insurance beneficiaries using historical data.
- **Problem Solved:** Enables data-driven budgeting, proactive resource allocation, and improved policy response to labor market changes.
- **Pipeline:** Data collection ➔ preprocessing ➔ feature engineering ➔ modeling ➔ evaluation ➔ web app deployment.

## Features

- Accurate time series forecasting using diverse models (ARIMA, SARIMA, VAR, AutoReg, Prophet, LSTM).
- Rigorous data cleaning and feature engineering for high-quality insights.
- Modular workflow for easy updates and extension.
- Web application for interactive, real-time forecasting by stakeholders.

## Tech Stack

| Technology | Purpose |
|----------------|---------------------------------------------------|
| **Python** | Core programming, data analysis |
| pandas, numpy | Data manipulation, numerical computations |
| matplotlib, seaborn, plotly | Static and interactive data visualization |
| statsmodels | Statistical modeling: ARIMA, SARIMA, VAR, AutoReg |
| Prophet | Trend and seasonality-based time series modeling |
| TensorFlow, Keras | Deep learning, LSTM sequence modeling |
| scikit-learn | Evaluation metrics, preprocessing (LabelEncoder) |
| Google Colab | Cloud-based notebook development |
| Flask | Web application deployment |
| warnings | Clean output management |

## Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/unemployed-insurance-forecasting.git
cd unemployed-insurance-forecasting
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the data from the official source or use the provided sample dataset.

## Usage

1. **Data Preprocessing:**
- All preprocessing scripts are in the Python Notebook.

2. **Model Training & Evaluation:**
- Use notebook scripts to train and validate your preferred model.

3. **Web Application:**
- Run the Flask app for the interactive dashboard:
```bash
python insurance.py
```
- Access `http://localhost:5000` in your browser.

## Results

| Model | Test MSE | Test MAE | Test R2 |
|----------|-----------------|------------|-------------|
| ARIMA | 102,763,733.35 | 5,691.37 | -8.18e-05 |
| SARIMA | 103,545,015.68 | 5,862.50 | N/A |
| AutoReg | 102,771,796.73 | 5,862.50 | N/A |
| Prophet | 57,301,995.56 | 3,522.24 | -0.1636 |

- **Prophet** achieved the best performance and was selected for deployment.
- The app allows users to input custom parameters and generate forecasts on demand.

## Screenshots

<img width="1920" height="1080" alt="Screenshot (389)" src="https://github.com/user-attachments/assets/87b64a9b-fe4e-4627-9797-c3dbb242e6f6" />

## Future Enhancements

- Integrate additional economic indicators (e.g., unemployment rate, GDP).
- Incorporate advanced deep learning models (LSTM, GRU).
- Develop more interactive dashboards with advanced visualizations.
- Automate data updates for real-time forecasting.
- Expand coverage to support multiple states or regions.

## Credits

- Immense gratitude to course mentors and teammates for critical guidance and hands-on collaboration throughout the project.

Feel free to raise issues or reach out for discussion, improvements, or collaboration on data science, time series forecasting, and public-sector analytics!
