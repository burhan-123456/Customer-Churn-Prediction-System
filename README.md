# 📊 Customer Churn Prediction using LightGBM

## 🚀 Overview
Customer churn prediction is an important problem for businesses to identify customers who are likely to stop using their services.  
This project uses **LightGBM** along with proper data preprocessing techniques like **One-Hot Encoding** and **SMOTE** to improve model performance.

The goal is to build an efficient model that helps in customer retention strategies.

---

## 🎯 Objectives
- Predict customer churn accurately  
- Handle categorical data using encoding techniques  
- Solve class imbalance using SMOTE  
- Build a robust model using LightGBM  

---

## 🧠 Model Used
**LightGBM (Light Gradient Boosting Machine)**  
- Fast and efficient boosting algorithm  
- Works well with structured/tabular data  
- Provides high accuracy with optimized performance  

---

## 📈 Model Performance
- **Accuracy:** 85%  

---

## 🗂️ Dataset Features
- Customer demographics (Gender, Senior Citizen, etc.)  
- Account details (Tenure, Contract Type)  
- Services used (Internet Service, Streaming, etc.)  
- Billing information (Payment Method, Monthly Charges)  

---

## ⚙️ Technologies Used
- Python  
- Pandas & NumPy  
- Scikit-learn  
- LightGBM  
- Imbalanced-learn (SMOTE)  

---

## 🔄 Workflow
1. Data Collection  
2. Data Cleaning  
3. Handling Missing Values  
4. **Encoding Categorical Variables (One-Hot Encoding)**  
5. **Handling Imbalanced Data using SMOTE**  
6. Train-Test Split  
7. Model Training using LightGBM  
8. Model Evaluation  
9. Prediction  

---

## 🔍 Techniques Used

### 🔹 One-Hot Encoding
- Converts categorical variables into numerical format  
- Creates binary columns for each category  
- Helps ML models understand non-numeric data  

### 🔹 SMOTE (Synthetic Minority Oversampling Technique)
- Used to handle class imbalance  
- Generates synthetic samples for minority class  
- Improves model performance on churn prediction  

---

## 📊 Key Insights
- Customers with **short-term contracts** are more likely to churn  
- High **monthly charges** increase churn probability  
- Class imbalance significantly affects predictions (handled using SMOTE)  

---

## ▶️ How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/churn-prediction.git

# Navigate to project folder
cd Customer-Churn-Prediction-System

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
