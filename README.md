# Machine Learning and Email Scraper Project

## Overview

The purpose of this project was to create a dataset of the most common words used in a collection of emails and using it to train multiple machine learning models to attempt to identify spam emails based on their word usage.

### Installation

Below are the dependencies used for this project.

<img src="images\scraper_dependencies.png" alt="Image Name" align="" width="400"/></br>
<img src="images\models_dependencies.png" alt="Image Name" align="" width="600"/>

### Saving Emails

Emails need to be saved in the "emails" folder and labeled based on whether they are determined to be Spam or Ham (not Spam). Any email determined to be Ham should simply be saved as an html file ending with .html. Any Spam email should be saved as an html file ending in -spam.html (more on this later).

## Files

### Email Scraper

The first part of the project was to create a program that would pull individual email files from a folder and scrape them to pull out each individual word. Then, a running total was kept for each word across all emails in a Pandas DataFrame. Finally, the top 5000 words were separated and the DataFrame was saved as a CSV file.

#### Scraping Emails:

A loop was created to scrape each email saved in the "emails" folder. Email files were only scraped if they ended in .html. A running total was kept for each individual word in each email file.

#### Word Extraction:

A function was created using Beautiful Soup and the html5lib parser to extract the words from the email files. The files first had to be checked to make sure they contained a body tag because I only wanted to pull the words out from the body of the emails. Any email not containing a body tagged was skipped.

Then, the NLTK (Natural Language Toolkit) was used to pick out the individual words which were then only included if they only contained characters in the alphabet using .isalpha. This was done to try and exclude any non-word tags or other references that might be found within the body tag.

#### Filtering:

Detail the process of filtering out certain chosen words and words with more than 20 letters.

#### Word Ranking:

Describe how words are ranked by frequency and the top 5000 words are selected.

#### Spam Indicator:

Explain how a spam indicator is added based on the presence of a certain ending in the file name.

Include a sample code snippet demonstrating the email scraping process.

### Machine Learning Models

Explain the process of building and using machine learning models to predict spam emails, including the following steps:

#### Data Preparation:

Describe how the word makeup of each email and the spam indicator are used as features.

#### Model Selection:

Explain why Logistic Regression, Support Vector Machine, and Random Forest were chosen as the models.

#### Model Training:

Detail how each model is trained using the prepared data.

#### Model Evaluation:

Explain how the performance of each model is evaluated using metrics such as accuracy, precision, recall, and F1 score.

#### Findings:

Provide insights into the findings of each model, including any strengths or weaknesses observed.

Include sample code snippets for training and evaluating each machine learning model.

## Conclusion

### Results

Include screenshots of results from the machine learning models, such as confusion matrices or classification reports, to illustrate their performance.

### Limitations

- Only used full, real words which would have excluded typos that can be very helpful in identifying Spam emails.

### Next Steps

Summarize the key findings of the project and discuss any future improvements or extensions that could be made.

### Contributors

Brian Kath
