#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import dependencies
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


# Read the CSV file from the Resources folder into a Pandas DataFrame
path = "outputs/emails_df.csv"
email_df = pd.read_csv(path)

# Review the DataFrame
email_df.head()


# In[ ]:


# Drop the email name column
email_df = email_df.drop(columns=email_df.columns[0])


# # Logistic Regression

# In[ ]:


# Separate the data into labels and features
# Separate the y variable, the labels
y = email_df['Spam Indicator']

# Separate the X variable, the features
X = email_df.drop(columns=['Spam Indicator'])


# In[ ]:


# Import the train_test_learn module
from sklearn.model_selection import train_test_split

# Split the data using train_test_split
# Assign a random_state of 1 to the function
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)


# In[ ]:


# Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
classifier = LogisticRegression(solver='lbfgs', max_iter=200, random_state=1)

# Fit the model using training data
classifier.fit(X_train, y_train)


# In[ ]:


# Make a prediction using the testing data
predictions = classifier.predict(X_test)
results = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)


# In[ ]:


# Print the balanced_accuracy score of the model
balanced_accuracy_score(y_test, predictions)


# In[ ]:


# Generate a confusion matrix for the model
confusion_matrix(y_test, predictions)


# In[ ]:


# Print the classification report for the model
print(classification_report(y_test, predictions))


# In[ ]:


# Import the RandomOverSampler module form imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

# Instantiate the random oversampler model
# # Assign a random_state parameter of 1 to the model
model = RandomOverSampler(random_state=1)

# Fit the original training data to the random_oversampler model
X_over, y_over = model.fit_resample(X_train, y_train)


# In[ ]:


# Count the distinct values of the resampled labels data
y_over.nunique()


# In[ ]:


# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
classifier = LogisticRegression(solver='lbfgs', max_iter=200, random_state=1)

# Fit the model using the resampled training data
classifier.fit(X_over, y_over)

# Make a prediction using the testing data
predictions = classifier.predict(X_test)
results = pd.DataFrame({"Predictions": predictions, "Actual": y_test}).reset_index(drop=True)


# In[ ]:


# Print the balanced_accuracy score of the model
balanced_accuracy_score(y_test, predictions)


# In[ ]:


# Generate a confusion matrix for the model
confusion_matrix(y_test, predictions)


# In[ ]:


# Print the classification report for the model
print(classification_report(y_test, predictions))


# # Support Vector Machine (SVM)

# In[ ]:


# Get the target variables.
target = email_df["Spam Indicator"]
target_names = ["ham", "spam"]


# In[ ]:


# Get the features.
data = email_df.drop(columns=['Spam Indicator'])
feature_names = data.columns


# In[ ]:


# Split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=1)


# In[ ]:


# Support vector machine linear classifier
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, y_train)


# In[ ]:


# Model Accuracy
print('Test Acc: %.3f' % model.score(X_test, y_test))


# In[ ]:


# Calculate the classification report
from sklearn.metrics import classification_report
predictions = model.predict(X_test)
print(classification_report(y_test, predictions,
                            target_names=target_names))


# # Random Forest

# In[ ]:


# Separate the data into labels and features
# Separate the y variable, the labels
y = email_df['Spam Indicator']

# Separate the X variable, the features
X = email_df.drop(columns=['Spam Indicator'])


# In[ ]:


# Import the train_test_learn module
from sklearn.model_selection import train_test_split

# Split the data using train_test_split
# Assign a random_state of 1 to the function
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)


# In[ ]:


# Create the StandardScaler instance
scaler = StandardScaler()


# In[ ]:


# Fit the Standard Scaler with the training data
X_scaler = scaler.fit(X_train)


# In[ ]:


# Scale the training data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# In[ ]:


# Create the random forest classifier instance
rf_model = RandomForestClassifier(n_estimators=500, random_state=78)


# In[ ]:


# Fit the model and use .ravel()on the "y_train" data.
rf_model = rf_model.fit(X_train_scaled, y_train.ravel())


# In[ ]:


# Making predictions using the testing data
predictions = rf_model.predict(X_test_scaled)


# In[ ]:


# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
)

# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)


# In[ ]:


# Displaying results
print("Confusion Matrix")
display(cm_df)
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, predictions))


# In[ ]:


# Get the feature importance array
importances = rf_model.feature_importances_
# List the top 10 most important features
importances_sorted = sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
importances_sorted[:10]

