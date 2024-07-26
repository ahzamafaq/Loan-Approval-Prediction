# Loan-Approval-Prediction
The goal of this project is to build a predictive model that can accurately determine the likelihood of loan approval based on various borrower attributes. By analyzing historical loan application data, the model aims to predict whether a loan application will be approved or denied.
1. Introduction:
The goal of this project is to predict whether a loan application will be approved based on various features related to the applicant's financial status and history.

2. Data Preprocessing:
Loading Data:
Use pandas to load the dataset.
Check for missing values and handle them appropriately.
Feature Engineering:
Convert categorical variables into numerical values using techniques like one-hot encoding or label encoding.
Create new features if needed (e.g., extracting the year from a date).
Scaling:
Standardize or normalize numerical features using StandardScaler from sklearn.

4. Exploratory Data Analysis (EDA):
Use visualization libraries such as matplotlib and seaborn to understand the distribution of features.
Plot correlations between features and the target variable to identify potential predictors.

5. Model Building:
Splitting Data:
Split the dataset into training and testing sets using train_test_split.
Model Selection:
Choose a machine learning model such as Neural Networks.
Initialize and configure the model.
Training the Model:
Train the model using the training data.
Use techniques like cross-validation to optimize the model.

6. Evaluation:
Evaluate the model's performance on the test set using metrics such as accuracy, precision, recall, F1-score.
Generate classification reports and confusion matrices to understand the performance.

7. Prediction:
Use the trained model to make predictions on new or unseen data.
Evaluate the model's predictions and analyze any misclassifications.

Key Code Snippets and Explanations
Loading Data:
import pandas as pd
df = pd.read_csv('lc_data.csv')

Handling Missing Values:
df.fillna(df.mean(), inplace=True)

Converting Categorical Variables:
df = pd.get_dummies(df, columns=['categorical_column'], drop_first=True)

Scaling Features:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

Splitting Data:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Model Training:
model = Sequential()

model.add(Dense(78, activation = 'relu'))
model.add(Dropout(0.15))

model.add(Dense(35, activation = 'relu'))
model.add(Dropout(0.15))

model.add(Dense(18, activation = 'relu'))
model.add(Dropout(0.15))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X_train, y_train)

Evaluation:
from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
