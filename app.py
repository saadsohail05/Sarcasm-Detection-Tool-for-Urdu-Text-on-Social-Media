import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import urduhack
from urduhack.preprocessing import normalize_whitespace
from urduhack.preprocessing import remove_punctuation
from urduhack.preprocessing import remove_accents
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

# Load the trained model and vectorizer
model = joblib.load('urdu_sentiment_model.pkl')
tfidf = joblib.load('urdu_sentiment_tfidf.pkl')

# Streamlit app setup with modern design

st.set_page_config(
    page_title="Sarcasm Detection App", 
    page_icon="🧠", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for centering content
st.markdown("""
    <style>
    /* Center the title, subheader, and input box */
    .centered-content {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }

    /* Center and style the buttons */
    .stButton > button {
        background-color: #00aaff;
        color: white;
        border-radius: 10px;
        width: 200px;
        margin: 20px auto;
        display: block;
    }

    /* Centering text input box */
    div[role='textbox'] {
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        text-align: center;
    }

    /* Style for title */
    h1, h2, h3 {
        text-align: center;
        color: #00aaff;
    }

    body {
        background-color: #f0f2f6;
    }

    </style>
""", unsafe_allow_html=True)

# Title and description with centered layout
st.markdown('<div class="centered-content">', unsafe_allow_html=True)
st.title("🧠 Urdu Sarcasm Detection")
st.subheader("Analyze Urdu text for sarcasm with AI-powered detection")
st.markdown('</div>', unsafe_allow_html=True)

# Making a list of stopwords
with open("stopwords-ur.txt", 'r', encoding='utf-8') as file:
        stopwords_from_file = file.read().splitlines()

stopdf=pd.read_csv("urdu_stopwords.csv",encoding='utf-8')
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
 چوگنی چکی چکیں چکے چہارشنبہ چیت ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاتک کاش کب کبھی کدھر کر
 کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کل کم کن کنہیں کو کوئی کون
 کونسا کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے
 کے گئی گئے گا گرما گرمی گنا گو گویا گھنٹا گھنٹوں گھنٹے گی گیا ہائیں ہائے ہاڑ ہاں ہر
 ہرچند ہرگز ہزار ہفتہ ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا ہوبہو ہوتا ہوتی
 ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں


""".split())

final_stopwords = set(stopwords_from_file + stopwords_from_csv)
final_stopwords.update(stop_words) 


urdu_dict=pd.read_csv("Dictionary_final.csv")
lemma_dict = pd.Series(urdu_dict.Lemma.values, index=urdu_dict.Word).to_dict()



# Function for preprocessing and predicting sarcasm
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

# Assuming stopwords are loaded correctly as per your original code
def remove_stopwords(text, final_stopwords):
    new_text = []
    for word in text.split():
        if word not in final_stopwords:
            new_text.append(word)
    return " ".join(new_text)

# Assuming lemmatization dictionary loaded correctly
def lemmatize_text(text, lemma_dict):
    words = text.split()  
    lemmatized_words = []
    
    for word in words:
        if word in lemma_dict:
            lemmatized_words.append(lemma_dict[word])
        else:
            lemmatized_words.append(word)
    return " ".join(lemmatized_words)

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

    # Predict sarcasm (1 for sarcastic, 0 for non-sarcastic)
    prediction = model.predict(text_vector)

    if prediction == 1:
        return "Sarcastic"
    else:
        return "Non-Sarcastic"

# Main input box
st.markdown("### Enter an Urdu sentence to analyze:")
user_input = st.text_input("Your sentence here", placeholder="Type in Urdu...")
if user_input:
    with st.spinner('Analyzing...'):
        prediction = predict_sarcasm(user_input)
    st.success(f"Prediction: {prediction}")

    # Visual feedback using color blocks
# Visual feedback using color blocks
    if prediction == "Sarcastic":
        st.markdown(
            "<div style='background-color: #2c2c2c; padding: 20px; border-radius: 5px; color: #ffcccb; font-size: 18px; text-align: center;'>"
            "This sentence is Sarcastic 😏"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color: #2c2c2c; padding: 20px; border-radius: 5px; color: #ccffcc; font-size: 18px; text-align: center;'>"
            "This sentence is Non-Sarcastic 😊"
            "</div>",
            unsafe_allow_html=True
        )


