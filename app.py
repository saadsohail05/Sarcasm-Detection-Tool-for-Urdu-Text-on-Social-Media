import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import urduhack
from urduhack.preprocessing import normalize_whitespace, remove_punctuation, remove_accents
import demoji
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from sklearn.model_selection import train_test_split
from nltk import ngrams
from collections import Counter 
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import joblib

# 1. Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="Urdu Sarcasm Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for centering content and styling
st.markdown("""
    <style>
    /* Center the title, subheader, and input box */
    .centered-content {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }

    /* Style the buttons */
    .stButton > button {
        background-color: #00aaff;
        color: white;
        border-radius: 10px;
        width: 200px;
        margin: 20px auto;
        display: block;
    }

    /* Style for title and headers */
    h1, h2, h3 {
        text-align: center;
        color: #00aaff;
    }

    /* Style for body background */
    body {
        background-color: #f0f2f6;
    }

    /* Progress bar style */
    .stProgress > div > div > div {
        background-color: #00aaff;
    }

    </style>
""", unsafe_allow_html=True)

# 3. Title and description with centered layout
st.markdown('<div class="centered-content">', unsafe_allow_html=True)
st.title("🧠 Urdu Sarcasm Detection")
st.subheader("Analyze Urdu text for sarcasm with AI-powered detection")
st.markdown('</div>', unsafe_allow_html=True)

# 4. Loading Stopwords
with open("stopwords-ur.txt", 'r', encoding='utf-8') as file:
    stopwords_from_file = file.read().splitlines()

stopdf = pd.read_csv("urdu_stopwords.csv", encoding='utf-8')
stopwords_from_csv = stopdf["stopword"].tolist()

# Additional stopwords
stop_words = set("""
آ آئی آئیں آئے آتا آتی آتے آداب آدھ آدھا آدھی آدھے آس آمدید آنا آنسہ آنی آنے
آپ آگے آہ آہا آیا اب ابھی ابے اتوار ارب اربویں ارے اس اسکا اسکی اسکے اسی اسے اف
افوہ الاول البتہ الثانی الحرام السلام الف المکرم ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور
اوپر اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکبر اکثر اگر اگرچہ اگست اہاہا ایسا ایسی ایسے ایک بائیں
بار بارے بالکل باوجود باہر بج بجے بخیر برسات بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند
بڑی بھر بھریں بھی بہار بہت بہتر بیگم تاکہ تاہم تب تجھ تجھی تجھے ترا تری تلک تم تمام
تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تہائی تیرا تیری تیرے تین جا جاؤ
جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی
جیسا جیسوں جیسی جیسے جیٹھ حالانکہ حالاں حصہ حضرت خاطر خالی خدا خزاں خواہ خوب خود دائیں درمیان
دریں دو دوران دوسرا دوسروں دوسری دوشنبہ دوں دکھائیں دگنا دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی
دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی رکھنے رکھو رکھی رکھے رہ رہا
رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی سراسر سلام سمیت سوا
سوائے سکا سکتا سکتے سہ سہی سی سے شام شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور
علاوہ عین فروری فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو
لوجی لوگوں لگ لگا لگتا لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے
لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محترمی محض مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق
مطلق مل منٹ منٹوں مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک نما نو نومبر
نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وعلیکم وغیرہ ولے
وگرنہ وہ وہاں وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونا پونی پونے پھاگن
پھر پہ پہر پہلا پہلی پہلے پیر پیچھے چاہئے چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ
چوغنی چکی چکیں چکے چہارشنبہ چیت ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاتک کاش کب کبھی کدھر کر
کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کل کم کن کنہیں کو کوئی کون
کونسا کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے
کے گئی گئے گا گرما گرمی گنا گو گویا گھنٹا گھنٹوں گھنٹے گی گیا ہائیں ہائے ہاڑ ہاں ہر
ہرچند ہرگز ہزار ہفتہ ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا ہوبہو ہوتا ہوتی
ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
""".split())

final_stopwords = set(stopwords_from_file + stopwords_from_csv)
final_stopwords.update(stop_words) 

# 5. Loading Lemmatization Dictionary
urdu_dict = pd.read_csv("Dictionary_final.csv")
lemma_dict = pd.Series(urdu_dict.Lemma.values, index=urdu_dict.Word).to_dict()

# 6. Loading the trained model and vectorizer with caching
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('urdu_sentiment_model.pkl')
    tfidf = joblib.load('urdu_sentiment_tfidf.pkl')
    return model, tfidf

model, tfidf = load_model_and_vectorizer()

# 7. Preprocessing Functions
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["  
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  
        u"\U000024C2-\U0001F251"  
        u"\U0001F900-\U0001F9FF"  # additional emojis
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_hash_sign(text):
    return re.sub(r'#', '', text)  

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def remove_english_words(text):
    return re.sub(r'\b[a-zA-Z]+\b', '', text)

def remove_diacritics(text):
    urdu_diacritics  = ['ِ', 'ٰ', 'ُ', 'ٍ', 'ً', 'َ']
    for letter in text:
        if letter in urdu_diacritics:
            text = text.replace(letter, '')
    return text

def remove_stopwords(text, final_stopwords):
    new_text = []
    for word in text.split():
        if word not in final_stopwords:
            new_text.append(word)
    return " ".join(new_text)

def lemmatize_text(text, lemma_dict):
    words = text.split()  
    lemmatized_words = []
    
    for word in words:
        if word in lemma_dict:
            lemmatized_words.append(lemma_dict[word])
        else:
            lemmatized_words.append(word)
    return " ".join(lemmatized_words)

# 8. Prediction Function with Confidence Scores
def predict_sarcasm(user_input):
    # Preprocess the input text
    user_input = remove_numbers(user_input)
    user_input = remove_emoji(user_input)
    user_input = remove_hash_sign(user_input)
    user_input = remove_mentions(user_input)
    user_input = remove_url(user_input)
    user_input = remove_english_words(user_input)
    user_input = remove_diacritics(user_input)
    user_input = remove_punctuation(user_input)
    user_input = remove_stopwords(user_input, final_stopwords)
    user_input = lemmatize_text(user_input, lemma_dict)

    # Transform the text into the TF-IDF vector
    text_vector = tfidf.transform([user_input]).toarray()

    # Predict sarcasm probabilities
    prediction_proba = model.predict_proba(text_vector)[0]
    prob_non_sarcastic = prediction_proba[0]
    prob_sarcastic = prediction_proba[1]

    # Determine prediction based on higher probability
    prediction = "Sarcastic" if prob_sarcastic > prob_non_sarcastic else "Non-Sarcastic"

    return prediction, prob_sarcastic, prob_non_sarcastic

# 9. Sidebar: Example Sentences
st.sidebar.header("Example Sentences")

# Sarcastic Urdu Sentences
sarcastic_examples = [
    "شاباش! آج آپ نے کمال کر دیا",
    "واہ! تم نے تو پورا دن سونے کا ریکارڈ توڑ دیا، بہت محنتی ہو۔",
    "واقعی، تمہیں سب کچھ معلوم ہوتا ہے، تم تو انسائیکلوپیڈیا ہو",
    "کمال ہے، تم کبھی غلطی نہیں کرتے، دنیا میں تمہاری مثال نہیں!",
    "تمہاری رفتار تو ایسی ہے جیسے تم دوڑ میں حصہ لے رہے ہو!"
]

# Non-Sarcastic Urdu Sentences
non_sarcastic_examples = [
    "مجھے اردو کتابیں پڑھنا بہت پسند ہے",
    "آج موسم بہت خوبصورت ہے۔",
    "آپ کی مدد کے لیے شکریہ۔",
    "میں نے آج ایک نئی چیز سیکھی",
    "خوش آمدید، گھر میں تشریف لائیں۔"
]

# Display Sarcastic Examples
st.sidebar.subheader("Sarcastic Urdu Sentences")
sarcastic_choice = st.sidebar.selectbox("Choose a sarcastic example", sarcastic_examples)

# Display Non-Sarcastic Examples
st.sidebar.subheader("Non-Sarcastic Urdu Sentences")
non_sarcastic_choice = st.sidebar.selectbox("Choose a non-sarcastic example", non_sarcastic_examples)

# Button to use Sarcastic Example
if st.sidebar.button("Use Sarcastic Example"):
    st.session_state['user_input'] = sarcastic_choice

# Button to use Non-Sarcastic Example
if st.sidebar.button("Use Non-Sarcastic Example"):
    st.session_state['user_input'] = non_sarcastic_choice

# 10. Display selected example in the main text input
if 'user_input' in st.session_state:
    user_input_text = st.text_input("Your sentence here", value=st.session_state['user_input'], placeholder="Type in Urdu...", key="input")
else:
    user_input_text = st.text_input("Your sentence here", placeholder="Type in Urdu...", key="input")

# 11. Main Input and Prediction
if user_input_text:
    with st.spinner('Analyzing...'):
        prediction, prob_sarcastic, prob_non_sarcastic = predict_sarcasm(user_input_text)
    
    # Display Prediction
    st.success(f"**Prediction:** {prediction}")
    
    # Display Confidence Score as a Single Progress Bar
    st.markdown("### Confidence Score:")
    confidence_percentage = prob_sarcastic * 100 if prediction == "Sarcastic" else prob_non_sarcastic * 100
    confidence_label = "Sarcastic Confidence" if prediction == "Sarcastic" else "Non-Sarcastic Confidence"
    
    # Display the confidence bar with percentage
    st.markdown(f"**{confidence_label}:** {confidence_percentage:.2f}%")
    st.progress(float(confidence_percentage)/100)  # Convert to 0-1 scale
    
    # Visual feedback using color blocks
    if prediction == "Sarcastic":
        st.markdown(
            "<div style='background-color: #2c2c2c; padding: 20px; border-radius: 5px; color: #ffcccb; font-size: 18px; text-align: center;'>"
            "یہ جملہ طنزیہ ہے 😏"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color: #2c2c2c; padding: 20px; border-radius: 5px; color: #ccffcc; font-size: 18px; text-align: center;'>"
            "یہ جملہ غیر طنزیہ ہے 😊"
            "</div>",
            unsafe_allow_html=True
        )

# 12. Footer: About Section in English
st.sidebar.markdown("## About")
st.sidebar.markdown("""
This app utilizes a trained model to analyze Urdu sentences and determine whether they are sarcastic or non-sarcastic. The model processes the input text by cleaning and lemmatizing it, transforming it into a TF-IDF vector, and then predicting the sentiment. The confidence score indicates the likelihood of the sentence being sarcastic or non-sarcastic.
""")
