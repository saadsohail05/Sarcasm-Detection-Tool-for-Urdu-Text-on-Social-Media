# 🌟 Urdu Sarcasm Detection App

## 📖 Introduction
The **Urdu Sarcasm Detection App** is an AI-powered web application that expertly analyzes Urdu text for sarcasm. In today's digital landscape, where sentiment analysis is crucial for understanding user feedback and content, this project addresses the complex challenge of sarcasm detection—often nuanced and context-dependent. By equipping users with a tool to identify sarcastic comments, this application enhances communication and boosts user engagement across various platforms.

---

## ⚙️ Installation
Follow these steps to set up the project:

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

---

## 🚀 Usage
To run the application:

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Enter an Urdu sentence in the input box and click the **Submit** button to analyze the text for sarcasm.

### 💡 Example
```plaintext
User Input: "تم تو بہت اچھے ہو"
Prediction: Non-Sarcastic
```

---

## 🌈 Features
- **Sarcasm Detection**: Analyze Urdu text for sarcasm using a trained machine learning model.
- **Preprocessing Pipeline**: Includes text normalization, emoji removal, stopwords filtering, and lemmatization.
- **User-Friendly Interface**: A modern and engaging web interface powered by Streamlit.
- **Visual Feedback**: Displays predictions with color-coded feedback for enhanced user experience.

---

## 📊 Data
The project utilizes a variety of datasets:
- **Stopwords**: A comprehensive list of Urdu stopwords sourced from `stopwords-ur.txt` and `urdu_stopwords.csv`.
- **Lemma Dictionary**: The `Dictionary_final.csv` contains Urdu words and their respective lemmas used for text normalization.

---

## 🛠️ Methodology
The application employs several techniques:
- **Text Preprocessing**: Cleans and prepares text for analysis through functions that remove numbers, emojis, hashtags, mentions, URLs, and stopwords.
- **Machine Learning Model**: A Gaussian Naive Bayes classifier trained on Urdu text data to predict sarcasm.
- **TF-IDF Vectorization**: Converts processed text into a vectorized format for model input.

---

## 📈 Results
The model has demonstrated promising results in detecting sarcasm in Urdu text. The accuracy of the model can be evaluated using metrics such as the confusion matrix and classification report.

---

## 🎯 Conclusion
The Urdu Sarcasm Detection App effectively tackles the challenge of identifying sarcasm in Urdu text, offering users valuable insights into their content's nature. This application has applications in various domains, including social media monitoring, customer feedback analysis, and beyond.

---

## 🚀 Future Work
Potential improvements and extensions for the project include:
- Expanding the dataset for improved model accuracy.
- Integrating additional languages for sarcasm detection.
- Enhancing the user interface with more interactive features.
- Implementing real-time sentiment analysis in social media platforms.

---

## 🤝 Contributing
Contributions are welcome! If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make your changes and commit them.
4. Submit a pull request with a detailed description of your changes.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.