# Loan_Prediction_using_keras_ANN

### **Project Description: Predictive Model for Loan Approval**

### **Overview:**
This project aims to develop a predictive model for loan approval based on various input parameters such as 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', and 'Total_Income', along with categorical variables including 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', and 'Property_Area'. After experimenting with multiple classification models like Logistic Regression, Decision Tree Classifier, Random Forest Classifier, XGBoost, Gaussian Naive Bayes, K-Nearest Neighbors, LightGBM, and CatBoost, the project ultimately selected an Artificial Neural Network (ANN) model. Model selection was based on evaluation metrics derived from a validation set, including accuracy, precision, recall, confusion matrix, classification report, ROC-AUC curve, and AUC score.

### **Project Workflow:**

### **Data Collection:**
The dataset containing loan application information, including various parameters like loan amount, loan term, credit history, total income, and categorical attributes such as gender, marital status, dependents, education, self-employment status, and property area, was obtained from a reliable source.

### **Data Cleaning:**
Missing values and outliers were handled to ensure data integrity and reliability. Techniques such as imputation and outlier detection were employed during this stage.

### **Data Visualization:**
Exploratory data analysis (EDA) was conducted using Python libraries like Matplotlib and Seaborn to visualize the relationships between different features and the target variable (loan approval status). This step helped in understanding the data distribution and identifying potential patterns.

### **Feature Engineering:**
Feature engineering techniques were applied to enhance the predictive power of the model. This involved creating new features, transforming existing ones, and encoding categorical variables using methods like one-hot encoding. Numerical features were scaled to ensure uniformity in their impact on the model.

### **Feature Selection:**
Feature selection was performed to identify the most relevant attributes for predicting loan approval. Techniques such as Recursive Feature Elimination (RFE) or feature importance from tree-based models were utilized to select the most informative features.

### **Model Training:**
Multiple classification models including Logistic Regression, Decision Tree Classifier, Random Forest Classifier, XGBoost, Gaussian Naive Bayes, K-Nearest Neighbors, LightGBM, and CatBoost were trained on the preprocessed data. The performance of each model was evaluated using cross-validation techniques.

### **Model Hyper-parameter Tuning:**
Hyperparameters of the selected model (ANN) were fine-tuned using techniques like Grid Search or Randomized Search to optimize its performance.

### **Model Evaluation:**
The trained ANN model was evaluated using the validation set. The accuracy achieved by the model was 75%, along with other relevant metrics such as precision, recall, confusion matrix, classification report, ROC-AUC curve, and AUC score. These metrics provided insights into the model's performance and its ability to predict loan approval accurately.

Overall, this project demonstrates the development of a robust predictive model for loan approval, leveraging various classification techniques and evaluation metrics to ensure its effectiveness in real-world applications.