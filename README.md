
# McDonald's Restaurant Review Topic Segmentation

This project aims to extract categories of issues from McDonald's restaurant reviews by leveraging NLP techniques such as **BERT**, **KMeans clustering**, and **Word2Vec**. The project focuses on identifying negative reviews and segmenting them into distinct topics, which are useful for identifying common issues like slow service or food quality.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Code Structure](#code-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Customer reviews can provide critical insights into restaurant performance. This project focuses on analyzing McDonald's restaurant reviews, identifying negative feedback, and segmenting the reviews into meaningful categories (e.g., slow service, food quality, etc.). The ultimate goal is to use this information to improve customer satisfaction.

## Dataset
The dataset consists of McDonald's restaurant reviews provided in an Excel file. These reviews include both positive and negative feedback on various aspects of the restaurant experience. The data is preprocessed, analyzed for sentiment, and clustered into topics.

## Methodology

### 1. Pre-processing Reviews with Pandas & NLTK
- **Tokenization**: The text reviews are tokenized using NLTK’s `word_tokenize`.
- **Stopword Removal**: Common English stopwords and McDonald's-specific terms like "mcdonald", "mcds" are removed.
- **Text Normalization**: Reviews are converted to lowercase, and non-alphabetical characters are removed.

### 2. Identify Negative Reviews with Sentiment Analysis
- **Sentiment Classification**: The `transformers` library is used to perform sentiment analysis on the reviews (based on BERT). Reviews are classified as positive, neutral, or negative.
- **Focus on Negative Reviews**: Only negative reviews are analyzed further to identify key issues.

### 3. Obtain Bag of Words & Count Term Frequencies and Significance with TF-IDF
- **Bag of Words (BoW)**: A matrix of token counts is created, where rows represent reviews and columns represent word frequencies.
- **TF-IDF**: The Term Frequency-Inverse Document Frequency (TF-IDF) is computed to assess the significance of each term in the reviews across the dataset.

### 4. Find Topics with LDA Model and Summarize with LLM
- **Latent Dirichlet Allocation (LDA)**: LDA is used to extract meaningful topics from the review text, allowing the grouping of reviews by themes like food quality, service speed, and cleanliness.
- **Summarization with Large Language Models (LLMs)**: After the topics are identified, LLMs are used to generate readable summaries of each topic for easy interpretation.

### 5. Map Issues to Restaurant Reviews
Three different methods are used to map issues to reviews:

#### Method 1: Apply BERT to Label Pre-processed Reviews
- BERT is used to classify the reviews into predefined categories (e.g., service, ambiance, food) based on their semantic content.

#### Method 2: Apply Word2Vec to Calculate Cosine Distance Between Predefined Labels and Reviews
- **Word2Vec** embeddings are generated from the review text. The cosine distance is calculated between the embeddings of predefined labels (e.g., "slow service") and the review texts to assign relevant categories.

#### Method 3: Cluster Reviews with KMeans & Label with Human Understanding (Best Method)
- **KMeans Clustering**: Reviews are clustered into distinct groups using KMeans based on their textual content.
- **Manual Labeling**: After clustering, human understanding is used to assign categories to each cluster, as this method provided the best results in identifying key topics.

## Code Structure
- `Topic_Segmentation_Restaurant_Review.ipynb`: The main notebook where the entire process is implemented.
  - **Text Preprocessing**: Includes tokenization, stopword removal, and text normalization using NLTK and Pandas.
  - **Sentiment Analysis**: Using BERT models to classify review sentiments.
  - **BoW and TF-IDF**: Implementations for term frequency analysis and TF-IDF scoring.
  - **Topic Modeling with LDA**: Using LDA for discovering topics and grouping reviews.
  - **Issue Mapping**: Mapping issues to reviews with BERT, Word2Vec, and KMeans.

## Installation
To run the notebook locally, you will need Python 3 and the following libraries:

```bash
pip install -r requirements.txt
```

### Requirements
- pandas
- numpy
- scikit-learn
- nltk
- gensim (for Word2Vec and LDA)
- transformers (for BERT)
- torch (for BERT)
- vaderSentiment (for sentiment analysis)
- matplotlib
- openai (for LLM-based summarization)

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/joyhwang41/mcdonalds-review-topic-segmentation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mcdonalds-review-topic-segmentation
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Topic_Segmentation_Restaurant_Review.ipynb
   ```
4. Run the notebook cells to preprocess the data, perform sentiment analysis, extract topics, and map issues using the described methods.

## Results
- **Sentiment Classification**: Reviews are classified as positive, neutral, or negative.
- **Topic Extraction**: Common themes from the reviews are identified using LDA.
- **Issue Mapping**: Reviews are mapped to specific restaurant issues using BERT, Word2Vec, and KMeans.

## Future Work
- **Model Fine-tuning**: Experiment with hyperparameters and model tuning to improve clustering and topic accuracy.
- **Real-time Analysis**: Implement real-time monitoring of reviews to provide live feedback to restaurant management.
- **Sentiment Analysis Improvement**: Explore using deep learning-based sentiment models for more accurate classification.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
