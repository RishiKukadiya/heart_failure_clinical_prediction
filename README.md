# Heart Failure Clinical Prediction using ANN 🫀

This project predicts the likelihood of death events in heart failure patients using an Artificial Neural Network (ANN). It is designed to assist in early identification of high-risk patients using clinical data.

## 🔧 Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- TensorFlow / Keras
- Scikit-learn

## 📊 Dataset
The dataset contains medical records of 299 patients with the following features:
- Age, Anaemia, Creatinine Phosphokinase, Diabetes, Ejection Fraction, High Blood Pressure, Platelets, Serum Creatinine, Serum Sodium, Sex, Smoking, and DEATH_EVENT (target).

## 🔍 Project Workflow

### 1. Data Preparation
- Handled missing values (if any)
- Exploratory Data Analysis (EDA) using visualizations
- Normalized numerical features
- Handled class imbalance using class weights

### 2. Model Building
- Built an ANN with:
  - Input layer
  - 2 Hidden layers with ReLU activation
  - Output layer with Sigmoid activation
- Used binary cross-entropy loss & Adam optimizer

### 3. Model Evaluation
- Evaluated using Accuracy, Precision, Recall, and F1-score
- Confusion Matrix for visual performance
- Trained with and without class weights to compare improvements

### 4. Hyperparameter Tuning
- Experimented with different learning rates and hidden layer sizes
- Compared validation accuracies for optimal tuning

## 📈 Results

- **Best Accuracy Achieved**: ~58% (baseline ANN, improved with class weighting)
- **Model improved** with class weighting & threshold tuning for better recall of critical class (DEATH_EVENT = 1)

## 📁 Folder Structure
