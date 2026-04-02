import streamlit as st
import pickle
import re
import pymorphy3
from nltk.corpus import stopwords
nltk.download('stopwords')
@st.cache_resource
def load_models():
    with open('logreg_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    morph = pymorphy3.MorphAnalyzer()
    stop_words = set(stopwords.words('russian'))
    stop_words.update(['rt', 'http', 'https', 'co', 't', 'это'])
    return model, vectorizer, morph, stop_words

model, vectorizer, morph, stop_words = load_models()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'[^а-яё]', ' ', text)
    words = text.split()
    cleaned_words = [morph.parse(w)[0].normal_form for w in words if w not in stop_words]
    return " ".join(cleaned_words)

st.title("🎭 Анализ тональности твитов")
st.write("Введите любой текст, и модель определит, позитивный он или негативный!")

user_input = st.text_area("Ваш текст:", height=100)

if st.button("Проанализировать"):
    if user_input.strip() == "":
        st.warning("Пожалуйста, введите текст.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        st.subheader("Результат:")
        if prediction == 1:
            st.success(f"😊 Позитивный текст (Уверенность: {probability[1]:.1%})")
        else:
            st.error(f"😠 Негативный текст (Уверенность: {probability[0]:.1%})")

        with st.expander("Посмотреть, как модель видит текст после очистки:"):
            st.code(cleaned)
