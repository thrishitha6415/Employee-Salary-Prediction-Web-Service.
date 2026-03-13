# 💰 Salary Prediction Using Machine Learning

This project predicts the **salary of a person based on their experience** using a Machine Learning model.
The model is trained using Python and deployed using a **Flask web application** where users can enter their experience and get a predicted salary.

---

## 🚀 Project Overview

The goal of this project is to build a simple **machine learning web application** that:

* Trains a regression model on salary data
* Saves the trained model using `pickle`
* Uses **Flask** to create a web interface
* Allows users to input their **years of experience**
* Predicts the **expected salary**

---

## 📂 Project Structure

```
Salary_Prediction_Thrishitha/
│
├── source_code/
│   ├── app.py                 # Flask web application
│   ├── train_model.py         # Script to train the ML model
│   ├── salary_model.pkl       # Trained machine learning model
│   ├── scaler.pkl             # Scaler used for preprocessing
│   ├── requirements.txt       # Required Python libraries
│   │
│   └── templates/
│       └── index.html         # Frontend web page
│
├── Project_Report_Thrishitha.pdf
```

---

## ⚙️ Technologies Used

* Python
* Machine Learning
* Scikit-learn
* Flask
* HTML
* Pickle

---

## 📊 How the Model Works

1. Dataset containing **experience vs salary** is used.
2. Data is **preprocessed and scaled**.
3. A **regression model** is trained.
4. The trained model is saved using **pickle (.pkl)**.
5. Flask loads the model and predicts salary from user input.

---

## 🖥️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/salary-prediction.git
cd salary-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask Application

```bash
python app.py
```

### 4️⃣ Open in Browser

```
http://127.0.0.1:5000
```

Enter your **years of experience** and the system will predict your salary.

---

## 📷 Project Demo

You can add screenshots of the web interface here.

Example:

```
Home Page
Salary Prediction Result
```

---

## 📄 Project Report

Detailed explanation of the project is available in:

`Project_Report_Thrishitha.pdf`

---

## 🎯 Future Improvements

* Add more features (education, job role, location)
* Use advanced ML models
* Improve UI design
* Deploy the project online (Render / Heroku)

---

## 👩‍💻 Author

**Thrishitha**

BSc Computer Science Student
Machine Learning Enthusiast
