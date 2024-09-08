# Email Scraper and Machine Learning Spam Project

## Overview

The purpose of this project was to create a dataset of the most common words used in a collection of emails and using it to train multiple machine learning models to attempt to identify Spam emails based on their word usage.

### Installation

Below are the dependencies used for this project.

<img src="images\scraper_dependencies.png" alt="Scraper Dependencies" align="" width="400"/></br>
<img src="images\models_dependencies.png" alt="Models Dependencies" align="" width="600"/>

### Saving Emails

Emails need to be saved in the "emails" folder and labeled based on whether they are determined to be Spam or Ham (not Spam). Any email determined to be Ham should simply be saved as an html file ending with .html. Any Spam email should be saved as an html file ending in -spam.html (more on this later).

## Files

### Email Scraper

The first part of the project was to create a program that would pull individual email files from a folder and scrape them to pull out each individual word. Then, a running total was kept for each word across all emails in a Pandas DataFrame. Finally, the top 5000 words were separated and the DataFrame was saved as a CSV file.

#### Scraping Emails:

A loop was created to scrape each email saved in the "emails" folder. Email files were only scraped if they ended in .html. A running total was kept for each individual word in each email file.

#### Word Extraction:

A function was created using Beautiful Soup and the html5lib parser to extract the words from the email files. The files first had to be checked to make sure they contained a body tag because I only wanted to pull the words out from the body of the emails. Any email not containing a body tag was skipped.

Then, the NLTK (Natural Language Toolkit) was used to pick out the individual words which were then included only if they contained characters in the alphabet using .isalpha. This was done to try and exclude any non-word tags or other references that might be found within the body tags.

#### Filtering:

A list was created to target certain words such as names and usernames to be omitted from the words included in the DataFrame. Any words longer than 20 characters were also excluded since words longer than 20 characters tended to be non-word tags or other references not likely to provide value to the dataset.

#### Word Ranking:

The DataFrame was then pared down to the 5000 most common words to limit the model to the words most likely to affect the machine learning models.

#### Spam Indicator:

A column was added at the end of the dataset to contain the Spam Indicator. Each field in the Spam Indicator column was initially set to zero. Then, any email file which ended in -spam.html was changed to a 1 to specifically identify the email as Spam.

<img src="images\spam_indicator.png" alt="Spam Indicator" align="" width="600"/>

### Machine Learning Models

The second part of the project was to create multiple machine learning models to predict whether an email was Spam or Ham based on the words that appear in the email.

#### Model Selection:

There were three different classification machine learning models used in this part of the project including Logistic Regression, Support Vector Machines (SVM), and Random Forest. These models were chosen due to their value in performing binary classification. Logistic Regression was chosen because of it's computational efficiency and it's clear and easily interpretable results. SVM was included because it is less prone to overfitting. And Random Forest was included because it can provide a breakdown and understanding of feature importance.

#### Model Training:

Below are the specifics on how each model was trained:

Logistic Regression - The columns in the CSV were first separated into labels (Spam Indicator) and features (the word columns) and splitting the data into training and testing sets using train_test_split. Then, using the Logistic Regression solver lbfgs, the model is fitted using the training data. Finally, predictions are made using the training data and comparing those predictions to the labels (Spam Indicator).

SVM - I first prepared the dataset for training by separating the target variable (what I am trying to predict) from the features (the input data that will be used to make predictions). I then divided the data into training and testing sets to train and evaluate the model performance. The SVM classifier was trained with a linear kernel since the data was linearly separable.

Random Forest - The first step was to separate the dataset into the labels (target variable) and features (the word columns). Then the data was split into training and testing sets and the features were normalized using standard scaling. The Random Forest classifier was then trained on the scaled training data and predictions were made which were evaluated for performance.

#### Model Evaluation:

Below is how the performance for each model are evaluated:

Logistic Regression - One indication of the performance of a Logistic Regression model is the balanced accuracy score which can be used to evaluate the performance of a classification model, especially in the presence of class imbalance (when one class is more frequent than the other - many more Ham emails than Spam). Another indication of the performance of this model is the confusion matrix which is a 2x2 matrix where the elements correspond to the counts of true positives, true negatives, false positives, and false negatives. Finally, a classification report can be printed to provide detailed metrics that help to understand the model's performance on the test data in terms of precision, recall, F-1 score, and support for each class.

SVM - One indication of the performance of an SVM model is the accuracy which is the proportion of correct predictions (both true positives and true negatives) over the total number of instances. While useful, accuracy can be misleading for imbalanced datasets where one class is more frequent than the other (e.g., if there are far more Ham emails than Spam). Another indication of the model performance is the classification report which evaluates model performance by providing key metrics such as precision, recall, and F1-score for each class, helping to assess how well the model distinguishes between classes (e.g., Spam and Ham). These metrics highlight the model’s ability to correctly identify true positives, minimize false positives, and balance accuracy across all classes.

Random Forest - One measure of performance for a Random Forest model is a confusion matrix which provides a detailed breakdown of the model's performance by showing counts of true positives, true negatives, false positives, and false negatives. Another measure of model performance is the accuracy score which measures the proportion of correct predictions. Another valuable measure for a Random Forest model is the feature importance analysis which highlights the top features (in this model, words) influencing the model's predictions.

#### Findings:

Below is are the specific findings for each model:

Logistic Regression - The balance accuracy score for the Logistic Regression model was 0.975 which means that the model has a balanced accuracy of 97.5% and indicates that, on average, the model correctly classifies 97.5% of both the Spam and Ham emails in the test dataset. The output from the confusion matrix showed there were 38 true negatives, 2 false positives, 0 false negatives, and 35 true positives which is a good indication of how few mistakes the model made in it's predictions. The classification report shows that the model has a high overall accuracy of 97%, performing well on both Ham and Spam email detection. The model correctly identifies 95% of Ham emails and 100% of Spam emails, with a strong balance between precision and recall for both classes, as reflected by F1-scores of 0.97. This means the model is both effective at catching Spam and minimizing false positives.

<img src="images\confusion_matrix.png" alt="Spam Indicator" align="" width="500"/>

<img src="images\classification_report.png" alt="Spam Indicator" align="" width="500"/>

SVM - The SVM model achieved a test accuracy of 97.3%, indicating that it correctly classified 97.3% of the emails. For the Ham class (non-spam), the model achieved 98% precision and recall, while for the Spam class, it achieved 97% precision and recall. The overall F1-score for both classes is 0.97 or higher, reflecting a good balance between precision and recall. These results show that the model performs well in distinguishing between Spam and Ham emails with high accuracy and reliability.

<img src="images\model_accuracy.png" alt="Spam Indicator" align="" width="500"/>

<img src="images\svm_classification_report.png" alt="Spam Indicator" align="" width="500"/>

Random Forest - The Random Forest model achieved an accuracy of 97.3%, indicating that it correctly classified nearly all emails. The confusion matrix reveals that the model correctly identified 35 out of 35 Spam emails and 38 out of 40 Ham emails, with only 2 Ham emails incorrectly labeled as Spam. The classification report shows high precision and recall for both classes, with an F1-score of 0.97, reflecting a well-balanced performance. Additionally, feature importance analysis highlights that terms like 'From', 'https', and common words like 'a' and 'our' are among the top features influencing the model's predictions.

<img src="images\rf_confusion_matrix.png" alt="Spam Indicator" align="" width="500"/>

<img src="images\feature_importances.png" alt="Spam Indicator" align="" width="500"/>

## Conclusion

### Results

The results of this project show that using the dataset created the trained models were largely accurate at determining whether an email was Spam or Ham based on the makeup of the words in the email.

### Limitations and Possible Improvements

- I only used full, real words which would have excluded typos that can be very helpful in identifying Spam emails. Other iterations of this program could try to exclude non-word tags while allowing common typos.
- The email dataset was limited to 300 emails. A larger sample of emails would likely be much more efficient and reliable in training the machine learning models.
- Certain very common and less differentiating words like "a", "the", "is" and "and" could be excluded to focus the models on words that might be more informative for classification.

### Contributor

Brian Kath
