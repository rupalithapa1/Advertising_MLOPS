# Advertising_MLOPS

Project Title: Advertising Sales Prediction Using Linear Regression.<br>

Project Overview: This project demonstrates a complete workflow for building, evaluating, and deploying a machine learning model to predict sales based on advertising budgets across TV, radio, and newspaper channels. The project integrates machine learning with MLOps practices to ensure scalability and reproducibility.

<br>Project Overview<br>
The primary objective of this project is to predict product sales based on advertising budgets. The workflow starts with Exploratory Data Analysis (EDA), followed by data preprocessing, model training, evaluation, and deployment using MLOps principles.<br>

**Workflow**<br>

1. Exploratory Data Analysis (EDA)<br>
Explored the relationships between advertising budgets (TV, radio, newspaper) and sales.<br>
Identified missing values and patterns in the data.<br>
Visualized key insights using scatter plots, correlation heatmaps, and histograms.<br>

2. Data Preprocessing<br>
Handled missing values using SimpleImputer (mean/median strategy).<br>
Standardized numerical features with StandardScaler to ensure uniform scaling.<br>
Split the dataset into training and testing sets for evaluation.<br>

3. Model Development<br>
Implemented a Linear Regression model to predict sales.<br>
Used cross-validation with KFold to validate model performance across multiple folds.<br>
Tuned hyperparameters with GridSearchCV to optimize model accuracy and performance.<br>

4. Model Evaluation<br>
Evaluated the model using metrics such as:
R² Score: To measure the proportion of variance explained.<br>
RMSE (Root Mean Squared Error): To determine average prediction errors.<br>

Observations:<br>
Advertising on TV and radio had a significant impact on sales.<br>
Newspaper budgets showed less predictive power.<br>

MLOps Workflow:<br>
Created a robust pipeline to automate the following tasks:<br>
Data preprocessing.<br>
Model training and evaluation.<br>
Version control for code and datasets.<br>

Deployed the model using:<br>
API Frameworks: Created REST API endpoints using Flask or FastAPI.<br>
Containerization: Deployed the application in Docker for scalability.<br>
Provided user-friendly interfaces to interact with the model for predictions.<br>

MLOps Tools:<br>
Docker (Containerization)<br>
Flask/FastAPI (API Deployment)<br>
Git (Version Control)<br>

Project Highlights:<br>
📊 Insights from EDA: Clear visualizations and data-driven insights.<br>
🧠 Model Accuracy: Optimized regression model with robust evaluation metrics.<br>
🚀 Deployment: Fully operational model with scalable deployment.<br>

Feel free to explore the project and contribute!
**Repository Link: Advertising_MLOPS
**
