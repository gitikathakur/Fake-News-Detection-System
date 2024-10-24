# Fake News Detection System

## Project Overview
The **Fake News Detection System** is a machine learning-based solution aimed at classifying news articles as *real* or *fake*. The system analyzes the text of news articles using NLP (Natural Language Processing) techniques to determine their authenticity. This project is implemented in Python and runs on Google Colab for ease of use.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
  
## Installation

### Prerequisites:
Before running this project, ensure that you have the following:
- A Google account for using **Google Colab**.
- Google Drive access for storing the dataset.

### Steps to Install:
1. **Clone the Repository:**
   Open Google Colab and run the following command to clone the repository:
   ```bash
   !git clone https://github.com/gitikathakur/Fake-News-Detection-System.git
2. **Install the Required Libraries:**
   After cloning the repository, install the dependencies using the command below:
   ```bash
   !pip install -r Fake-News-Detection-System/requirements.txt
3. **Mount Google Drive**
   ```bash
   from google.colab import drive
   drive.mount('/content/drive')
4. **Download Dataset**
   Download Dataset: Download the dataset from this [Google Drive link](https://drive.google.com/drive/folders/1df7OgEEkbT3459p1kzKd3D5FU-0ShJmp) and store it in your Google Drive.
5. **Open the Project Notebook**
   Open the Notebook: Open the notebook `Fake_News_Detection_System.ipynb` in the Colab environment and follow the instructions within the notebook for executing the code and running the project.

You can upload the notebook directly to Colab from Google Drive, or use [Google Colab](https://colab.research.google.com/) to manually upload the file.
## Dataset
The Fake News Detection System uses a dataset of news articles labeled as real or fake for model training and evaluation.
## Dataset Contents:
- `train.csv`: Used for training the model, contains labeled news articles.
- `test.csv`: Used for evaluating the model's performance on unseen data.
## Dataset Location:
You can access the dataset at this [Google Drive link](https://drive.google.com/drive/folders/1df7OgEEkbT3459p1kzKd3D5FU-0ShJmp).
## Usage:
1. To run the project on Google Colab:
Load the Dataset: After mounting your Google Drive, ensure the dataset files (`train.csv`, `test.csv`) are in your Drive and accessible to the notebook.
2. Preprocess the Data: The system processes the news articles by performing the following:
Tokenization (splitting text into words).
Removing stopwords and punctuation.
Converting text into numerical format using TF-IDF Vectorization.
3. **Train the Model**: The system uses machine learning algorithms like Logistic Regression and Support Vector Machines (SVM) to classify the news. Execute the training section in the notebook to build the model.
4. **Test the Model:** Run the prediction function on the test dataset to evaluate the model's performance.   
## Evaluation Metrics
The performance of the model is evaluated using the following metrics:

- Accuracy: Measures the percentage of correct predictions.
- Precision: The proportion of true positive predictions out of all positive predictions.
- Recall: The proportion of true positive predictions out of all actual positives.
- F1-Score: Harmonic mean of precision and recall for better overall evaluation.
- Confusion Matrix: Provides a detailed breakdown of true positives, true negatives, false 
  positives, and false negatives.
