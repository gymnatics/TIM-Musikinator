import streamlit as st
import numpy as np
from transformers import pipeline, AutoTokenizer,TFAutoModel
import joblib


classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

# NUM_CLASSES = 5
# tf_model = (TFAutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=NUM_CLASSES))
# set tokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tf_model = TFAutoModel.from_pretrained(model_ckpt)

def tokenize(batch):
    encoded_dict =  tokenizer(batch, padding="max_length", truncation=True, max_length = 128)
    input_ids = encoded_dict["input_ids"]
    attention_mask = encoded_dict["attention_mask"]
    return input_ids, attention_mask
def extract_hidden_states(batch):
    # First convert text to tokens
    inputs = tokenizer(batch["text"], padding="max_length",
                       truncation=True, return_tensors='tf', max_length = 128)
    # Extract last hidden states
    outputs = tf_model(inputs)
     # Return vector for [CLS] token
    return {"hidden_state": outputs.last_hidden_state[:,0].numpy()}

# Load the saved sklearn model
model_path_sklearn = "text_classification_model.pkl"
lr_model = joblib.load(model_path_sklearn)
def predict(input):
    X_predict = np.array(input["hidden_state"])
    predictions = lr_model.predict(X_predict)
    
    return predictions

def detect_sentiment_custom(sentence):
    extracted_features = extract_hidden_states({"text": sentence})
    predictions = predict(extracted_features)[0]
    labels = ['Sadness', 'Love', 'Joy', 'Anger', 'Fear']
    return labels[predictions]

def string_concat(string):
    return ("This will be your caption: " + detect_sentiment_custom(string))

def detect_sentiment(sentence):
    labels_to_remove = ['disgust', 'surprise', 'neutral']
    sentiments = classifier(sentence)[0]
    filtered_sentiments = [item for item in sentiments if item['label'] not in labels_to_remove]
    max_label = max(filtered_sentiments, key=lambda x: x['score'])['label']
    return (max_label)

st.set_page_config(page_title = "My Webpage", page_icon=":tada:", layout="wide")

# --- HEADER SECTION---
with st.container():
    st.subheader("Hi, We are TIM Musikinator :guitar:")
    st.title("A music recommendation app for your captions")
    st.write("Our goal is to recommend you music that would best suit your captions")

col1,col2 = st.columns([0.4,0.6])

with col1:
    with st.container():
        st.image("music.webp")
        caption = st.text_input("Caption",
                                placeholder = "Write your caption here")
        st.write("Press **Enter** to confirm your caption")
        if caption:
            # st.write("Detected Sentiment: " + detect_sentiment(caption))
            st.write("Detected Sentiment: " + detect_sentiment_custom(caption))

with col2:
    with st.container():
        if caption:
            st.subheader(string_concat(caption))

            st.write("By Artist name :microphone:")
            audio_file = open('7400.mp3', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg')

            with st.popover("Watch music video here"):
                VIDEO_URL = "https://www.youtube.com/watch?v=wd7JqXouRRA"
                st.video(VIDEO_URL)
        else:
            st.subheader("You will see your recommended songs here :point_down:")

    st.divider()

    with st.container():
        if caption:
            st.subheader(string_concat(caption))

            st.write("By Artist name :microphone:")
            audio_file = open('7400.mp3', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg')

            with st.popover("Watch music video here"):
                VIDEO_URL = "https://www.youtube.com/watch?v=wd7JqXouRRA"
                st.video(VIDEO_URL)