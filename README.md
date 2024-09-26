# Urdu Sarcasm Detection App

## Introduction
The **Urdu Sarcasm Detection App** is an AI-powered web application designed to analyze Urdu text for sarcasm. With the growing importance of sentiment analysis in understanding user feedback and content, this project aims to tackle the challenge of detecting sarcasm, which is often nuanced and context-dependent. By providing users with a tool to identify sarcastic comments, this application can help improve communication and enhance user engagement across various platforms.

## Installation
To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saadsohail05/urdu-sarcasm-detection.git
   cd urdu-sarcasm-detection
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download additional resources**:
   Ensure you have the following files in the project directory:
   - `urdu_sentiment_model.pkl` (the trained model)
   - `urdu_sentiment_tfidf.pkl` (the TF-IDF vectorizer)
   - `stopwords-ur.txt` (list of Urdu stopwords)
   - `Dictionary_final.csv` (Urdu lemma dictionary)

## Usage
To run the application:

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Enter an Urdu sentence in the input box and click the submit button to analyze the text for sarcasm.

### Example
```plaintext
User Input: "تم تو بہت اچھے ہو"
Prediction: Non-Sarcastic
```

## Features
- **Sarcasm Detection**: Analyze Urdu text for sarcasm with a trained machine learning model.
- **Preprocessing Pipeline**: Includes text normalization, emoji removal, stopwords filtering, and lemmatization.
- **User-Friendly Interface**: Modern and engaging web interface powered by Streamlit.
- **Visual Feedback**: Displays predictions with color-coded feedback for better user experience.

## Data
The project uses a variety of datasets:
- **Stopwords**: A comprehensive list of Urdu stopwords sourced from `stopwords-ur.txt` and `urdu_stopwords.csv`.
- **Lemma Dictionary**: The `Dictionary_final.csv` file contains Urdu words and their respective lemmas used for text normalization.

## Methodology
The application employs several techniques:
- **Text Preprocessing**: Cleans and prepares text for analysis through functions that remove numbers, emojis, hashtags, mentions, URLs, and stopwords.
- **Machine Learning Model**: A Gaussian Naive Bayes classifier trained on Urdu text data to predict sarcasm.
- **TF-IDF Vectorization**: Converts processed text into a vectorized format for model input.

## Results
The model has shown promising results in detecting sarcasm in Urdu text. The accuracy of the model can be evaluated using metrics such as confusion matrix and classification report.

## Conclusion
The Urdu Sarcasm Detection App successfully addresses the challenge of identifying sarcasm in Urdu text, providing users with valuable insights into the nature of their content. This application can be utilized in various domains, including social media monitoring, customer feedback analysis, and more.

## Future Work
Potential improvements and extensions for the project include:
- Expanding the dataset for better model accuracy.
- Integrating additional languages for sarcasm detection.
- Enhancing the user interface with more interactive features.
- Implementing real-time sentiment analysis in social media platforms.

## Contributing
Contributions are welcome! If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make your changes and commit them.
4. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
