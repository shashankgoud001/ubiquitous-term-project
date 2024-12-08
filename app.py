import time, os
import logging
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE, AUDIO_FILES_DIR
import sound
import extractor
import joblib


emoji_mapping = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}




def main():
    title = "Emotion Detection From Audio"
    st.title(title)
    audio_file_path = 'frontend/output/recording/recorded.wav'
    flag = True
    image = Image.open(os.path.join(IMAGE_DIR, 'img1.jpg'))
    st.image(image, use_column_width=True)
    s = sound.Sound()
    # use_existing_file = st.checkbox("Select from existing audio files")
    available_files = os.listdir(AUDIO_FILES_DIR)
    selected_file = st.selectbox("Choose an audio file from the directory", available_files)
    # print(selected_file)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('⏺️ Record'):
            with st.spinner(f'Recording for {DURATION-1} seconds ....'):
                s.record()
            # st.success("Recording completed")
            flag = False
            output = 1
     
    with col2:
        if st.button('▶️ Play'):
            output = 2
           
    with col3:
        if st.button('🔍 Classify'):
            audio_file_path = 'frontend/output/recording/recorded.wav'
            if(flag is True):
                audio_file_path = 'frontend/output/recording/'+selected_file
            output = 3
           
    if 'output' in locals():
        if output==1:
            st.success("Recording completed")
        elif output==2:
            try:
                audio_file = open(WAVE_OUTPUT_FILE, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
            except:
                st.write("Please record sound first")
        elif output==3:
            # model = joblib.load('frontend/pretrained_models/RandomForestClassifier')
            model = joblib.load('frontend/pretrained_models/ExtraTreesClassifier')
            prediction = extractor.predict_emotion(audio_file_path,model)
            st.markdown(f"<h1 style='font-size:40px; text-align: center;'>{prediction}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='font-size:60px; text-align: center;'>{emoji_mapping[prediction]}</h1>", unsafe_allow_html=True)
        

if __name__ == '__main__':
    main()
