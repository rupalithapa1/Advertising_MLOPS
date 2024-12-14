# Advertising_MLOPS

Project Title: Advertising Sales Prediction Using Linear Regression.<br>

Project Overview: This project demonstrates a complete workflow for building, evaluating, and deploying a machine learning model to predict sales based on advertising budgets across TV, radio, and newspaper channels. The project integrates machine learning with MLOps practices to ensure scalability and reproducibility.
Project Overview
The primary objective of this project is to predict product sales based on advertising budgets. The workflow starts with Exploratory Data Analysis (EDA), followed by data preprocessing, model training, evaluation, and deployment using MLOps principles.

Workflow
1. Exploratory Data Analysis (EDA)
Explored the relationships between advertising budgets (TV, radio, newspaper) and sales.
Identified missing values and patterns in the data.
Visualized key insights using scatter plots, correlation heatmaps, and histograms.

2. Data Preprocessing
Handled missing values using SimpleImputer (mean/median strategy).
Standardized numerical features with StandardScaler to ensure uniform scaling.
Split the dataset into training and testing sets for evaluation.

3. Model Development
Implemented a Linear Regression model to predict sales.
Used cross-validation with KFold to validate model performance across multiple folds.
Tuned hyperparameters with GridSearchCV to optimize model accuracy and performance.

4. Model Evaluation
Evaluated the model using metrics such as:
R² Score: To measure the proportion of variance explained.
RMSE (Root Mean Squared Error): To determine average prediction errors.

5.Observations:
Advertising on TV and radio had a significant impact on sales.
Newspaper budgets showed less predictive power.

6.MLOps Workflow:
Created a robust pipeline to automate the following tasks:

Data preprocessing.<br>
Model training and evaluation.<br>
Version control for code and datasets.<br>

Deployed the model using:
API Frameworks: Created REST API endpoints using Flask or FastAPI.
Containerization: Deployed the application in Docker for scalability.
Provided user-friendly interfaces to interact with the model for predictions.

MLOps Tools:
Docker (Containerization)
Flask/FastAPI (API Deployment)
Git (Version Control)

Project Highlights
📊 Insights from EDA: Clear visualizations and data-driven insights.
🧠 Model Accuracy: Optimized regression model with robust evaluation metrics.
🚀 Deployment: Fully operational model with scalable deployment.
