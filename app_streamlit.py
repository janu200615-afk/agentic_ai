# Minimal Streamlit UI for interacting with a local agent
import streamlit as st
import pandas as pd
import requests, base64, json
from utils import gen_synthetic_data, prepare_data
from model import SmallNet, load_state_dict_bytes, get_state_dict_bytes
import torch

SERVER = 'http://127.0.0.1:5000'
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

def transcribe_audio_file(uploaded_file):
    # Save uploaded file to temp, convert to WAV if needed, and transcribe using Google Web Speech API (requires internet)
    ext = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.'+ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    wav_path = tmp_path
    if ext != 'wav':
        # convert to wav via pydub (requires ffmpeg on system)
        sound = AudioSegment.from_file(tmp_path)
        wav_path = tmp_path + '.wav'
        sound.export(wav_path, format='wav')
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, language='en-IN')
    except Exception as e:
        text = f"(transcription failed: {e})"
    return text



# Extended crop images and localized names
EXTENDED_CROPS = [
    'maize','rice','cotton','wheat','sugarcane','pulses','millets','groundnut','sunflower','vegetables'
]
IMAGES = {k: f'images/{k}.jpg' for k in EXTENDED_CROPS}
LOCALIZED_NAMES = {
    'English': {
        'maize':'Maize','rice':'Rice','cotton':'Cotton','wheat':'Wheat','sugarcane':'Sugarcane','pulses':'Pulses','millets':'Millets','groundnut':'Groundnut','sunflower':'Sunflower','vegetables':'Vegetables'
    },
    'Hindi': {
        'maize':'मक्का','rice':'चावल','cotton':'कपास','wheat':'गेहूं','sugarcane':'गन्ना','pulses':'दाल','millets':'बाजरा','groundnut':'मूंगफली','sunflower':'सूरजमुखी','vegetables':'सब्ज़ियाँ'
    },
    'Telugu': {
        'maize':'మక్కా','rice':'అన్నం','cotton':'పత్తి','wheat':'గోదుమ','sugarcane':'చెక్క','pulses':'పప్పు','millets':'సజ్జ/సామ/కొర్ర','groundnut':'వేరుశనగ','sunflower':'సూర్యకాంతి','vegetables':'కూరగాయలు'
    }
}

from PIL import Image
import os

IMAGES = {
    'maize': 'images/maize.jpg',
    'rice': 'images/rice.jpg',
    'cotton': 'images/cotton.jpg'
}

st.set_page_config(page_title='Agentic AI Farmer', layout='centered')

st.title('Agentic AI — Farmer Assistant (Demo)')

st.sidebar.header('Controls')
use_sample = st.sidebar.checkbox('Use synthetic sample data', True)

# Provide quick sample inputs (JSON / CSV) for judges
st.sidebar.markdown('---')
if st.sidebar.button('Load sample JSON input'):
    import json
    with open('sample_inputs.json','r') as f:
        s = json.load(f)
    st.session_state['sample_loaded'] = s['samples']
    st.success('Sample JSON loaded into session. Use Get recommendation.')

if st.sidebar.button('Load sample CSV input'):
    import pandas as _pd
    s = _pd.read_csv('sample_inputs.csv')
    st.session_state['sample_loaded'] = s.to_dict(orient='records')
    st.success('Sample CSV loaded into session. Use Get recommendation.')

if use_sample:
    df = gen_synthetic_data(200, seed=1)
else:
    uploaded = st.sidebar.file_uploader('Upload CSV (columns: soil_ph, soil_moisture, temperature, rain_last_7d, day_of_year, previous_crop_index)', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.info('Upload a CSV or use the sample data.')
        st.stop()

st.write('Sample of data:')
st.dataframe(df.head())

if st.button('Get recommendation from local agent'):
    X_train, X_test, y_train, y_test = prepare_data(df)
    # load global model from server if available
    try:
        r = requests.get(SERVER + '/get_global', timeout=3)
        if r.status_code == 200:
            model_b64 = r.content.decode('utf-8')
            global_bytes = base64.b64decode(model_b64.encode('utf-8'))
            model = SmallNet()
            load_state_dict_bytes(model, global_bytes, map_location=torch.device('cpu'))
        else:
            model = SmallNet()
    except Exception as e:
        st.warning('Could not reach server, using local init model.')
        model = SmallNet()

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X_test.astype('float32')))
        preds = out.argmax(dim=1).numpy()

    # show simple aggregated distribution of predicted crop labels
    import numpy as np
    from utils import CROPS
    unique, counts = np.unique(preds, return_counts=True)
    dist = {CROPS[int(u)]: int(c) for u,c in zip(unique, counts)}
    st.subheader('Recommendation distribution (on test split)')
    st.write(dist)

    st.success('Agent produced crop recommendations. You can accept to simulate action.')
# show grid of crop images with localized captions
st.subheader('Recommended Crops with Images')
lang = st.selectbox('Language', ['English', 'Hindi', 'Telugu'], index=0)
names = LOCALIZED_NAMES.get(lang, LOCALIZED_NAMES['English'])
# create grid: 2 columns
cols = st.columns(2)
i = 0
for crop in EXTENDED_CROPS:
    col = cols[i % 2]
    with col:
        caption = names.get(crop, crop.capitalize())
        if crop in dist:
            score = dist[crop]
        else:
            score = 0
        st.caption(f"{caption} — Score: {score}")
        img_path = IMAGES.get(crop)
        if img_path and os.path.exists(img_path):
            st.image(img_path, use_column_width=True, caption=caption)
    i += 1

    

    st.subheader('Recommended Crops with Images')
    for crop, count in dist.items():
        st.write(f"{crop.capitalize()} ({count} votes)")
        if crop in IMAGES and os.path.exists(IMAGES[crop]):
            st.image(IMAGES[crop], caption=crop.capitalize(), use_column_width=True)


if st.button('Sync local agent (train & send update)'):
    st.info('Training local agent on sample data (short epoch) and sending update...')
    # here we simply call a local agent process via requests (for demo we call agent.py in-process)
    import subprocess, sys, shlex
    subprocess.Popen([sys.executable, 'agent.py', '--id', '99', '--simulate']).wait()
    st.success('Local agent trained and update sent (if server reachable).')
