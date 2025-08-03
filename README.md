# Diabetes Predictor using Machine Learning  
A simple and effective machine learning model that predicts the likelihood of diabetes in individuals based on medical attributes. This project utilizes the PIMA Indians Diabetes dataset and is perfect for beginners interested in applying ML to healthcare problems.

## 📊 Dataset
The model is trained on the PIMA Indians Diabetes Dataset, which includes the following features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Target variable: Outcome (1 = Diabetic, 0 = Non-diabetic)

## 🔍 Features
- Data preprocessing and visualization
- Model training using scikit-learn
- Accuracy evaluation and metrics
- Predictive tool using trained model
- Simple command-line interface (CLI) or optional web UI (if applicable)

## 🛠️ Tech Stack
- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn

## 🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/krishnakantt/Diabetes-Predictor-using-ML.git
cd diabetes-predictor-ml
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt

## 📈 Model Performance
The model is evaluated using accuracy, precision, recall, F1-score, and confusion matrix. Results may vary slightly depending on random seed and split.

## 🧪 Example
  #Sample input  
  [6, 148, 72, 35, 0, 33.6, 0.627, 50] 
  
  #Predicted output  
  [1]  
  The person is diabetic

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
