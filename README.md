# 📈 S&P 500 Price Prediction using Machine Learning

This project was developed as part of a Computer Science group project. It implements and evaluates multiple machine learning and deep learning models to predict future S&P 500 stock prices and assess their real-world performance.

---

## 🚀 Overview

The system provides a complete pipeline for financial time series forecasting:

- Predicts future S&P 500 prices (default: **01-01-2026 → 01-05-2027**)  
- Runs multiple ML/DL models on the same dataset  
- Evaluates predictions using standard regression metrics  
- Simulates trading performance via backtesting  
- Outputs results, metrics, and visualisations for comparison  

---

## ⚙️ How It Works

1. Input historical data is stored in the `Data/` folder  
2. `main.py` runs the full pipeline:
   - Loads and preprocesses data  
   - Trains and runs all models  
   - Generates predictions  
   - Evaluates performance using metrics  
   - Runs backtesting to simulate real-world trading outcomes  
3. Results are saved in the `Newest Results/` folder  

---

## 🧠 Models

### 🔧 Developed in this Project
- ANN (Artificial Neural Network)  
- CNN-LSTM  
- DTR (Decision Tree Regressor)  
- GRU (Gated Recurrent Unit)  
- KNN  
- KNN-PM  
- Seer (and variants)  

### 🤝 Contributions
- Random Forest — *Lysandra*  
- LSTM — *Avin*  
- Linear Regression — *Prisha*  
- SVR — *Safiya*  

---

## 📊 Evaluation Methods

Each model is evaluated using:

- **RMSE** (Root Mean Squared Error)  
- **MAE** (Mean Absolute Error)  
- **R² Score**  
- **Directional Accuracy**  

Additionally, predictions are tested using a **backtesting framework** to assess how they would perform in a realistic trading scenario.

---

## ▶️ Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt


## 🙏 Acknowledgements
This readme file was assisted by AI tools (ChatGPT) to help with structuring and formatting.