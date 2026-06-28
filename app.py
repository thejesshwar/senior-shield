import streamlit as st
import torch
import os
import onnxruntime as ort
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import time
from transformers import XLMRobertaTokenizer
from lime.lime_text import LimeTextExplainer
from gtts import gTTS 
import matplotlib.pyplot as plt
from config import SCAM_THRESHOLD, MAX_LENGTH, MODEL_PATH, ONNX_MODEL

if os.name == 'nt':
    try:
        lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        os.add_dll_directory(lib_path)
    except Exception as e:
        pass

st.set_page_config(page_title="Senior Shield", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .big-font { font-size:22px !important; }
    .warning-text { font-size:26px !important; font-weight: bold; color: #FF4B4B; }
    .safe-text { font-size:26px !important; font-weight: bold; color: #00CC96; }
    .stTextArea textarea { font-size: 18px !important; }
    </style>
""", unsafe_allow_html=True)

ui_text = {
    "English": {
        "title": "Senior Shield",
        "subtitle": "Privacy-First Fraud Protection for Seniors",
        "tab1": "SMS / WhatsApp Shield",
        "tab2": "Call Audio Shield",
        "input_label": "Paste SMS / WhatsApp Message:",
        "placeholder": "e.g., Your SBI account is suspended",
        "scan_btn": "Scan Message",
        "scam_alert": "SCAM DETECTED",
        "safe_alert": "SAFE MESSAGE",
        "why": "Why is this suspicious?",
        "notify": "Notify Family",
        "audio_label": "Upload Call Recording",
        "analyze_audio": "Analyze Audio",
        "listening": "Listening & Transcribing...",
        "audio_safe": "This message is safe.",
        "audio_scam": "Warning. Scam detected. Do not click any links.",
        "family_alert": "Alert sent to Rahul via WhatsApp API"
    },
    "Hindi": {
        "title": "सीनियर शील्ड",
        "subtitle": "बुजुर्गों के लिए गोपनीयता-प्रथम सुरक्षा",
        "tab1": "एसएमएस / व्हाट्सएप सुरक्षा",
        "tab2": "कॉल रिकॉर्डिंग सुरक्षा",
        "input_label": "सन्देश यहाँ पेस्ट करें:",
        "placeholder": "जैसे: आपका बैंक खाता बंद हो गया है",
        "scan_btn": "स्कैन करें",
        "scam_alert": "धोखा पकड़ा गया!",
        "safe_alert": "सन्देश सुरक्षित है",
        "why": "यह खतरनाक क्यों है?",
        "notify": "परिवार को सूचित करें",
        "audio_label": "कॉल रिकॉर्डिंग अपलोड करें",
        "analyze_audio": "ऑडियो जांचें",
        "listening": "सुन रहा है और लिख रहा है...",
        "audio_safe": "यह संदेश सुरक्षित है।",
        "audio_scam": "सावधान। यह एक धोखा है। लिंक पर क्लिक न करें।",
        "family_alert": "राहुल को व्हाट्सएप पर अलर्ट भेजा गया"
    }
}

@st.cache_resource
def load_resources():
    print("Loading Optimized Resources")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.intra_op_num_threads = 1    
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_MODEL, sess_options=sess_options, providers=providers)
    
    dummy_input = tokenizer("Warmup", return_tensors="np", padding=True, truncation=True, max_length=MAX_LENGTH)
    dummy_feed = {
        session.get_inputs()[0].name: dummy_input['input_ids'].astype(np.int64),
        session.get_inputs()[1].name: dummy_input['attention_mask'].astype(np.int64)
    }
    session.run(None, dummy_feed)
    
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return tokenizer, session, whisper_model
@st.cache_resource
def load_explainer():
    return LimeTextExplainer(class_names=["Safe", "Scam"])

tokenizer, session, whisper_model = load_resources()
@st.fragment
def render_lime_explanation(text, t_ui_strings):
    """Renders the heavy LIME visual chart independently of the main UI thread."""
    with st.spinner(t_ui_strings['why']):
        st.subheader(t_ui_strings['why'])
        try:
            num_words = len(text.split())
            safe_features = min(5, num_words)
            
            if safe_features > 0:
                exp = explainer.explain_instance(text, predictor, num_features=safe_features, num_samples=100)
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)              
                plt.close(fig)
            else:
                st.info("Message is too short for detailed feature analysis.")
        except Exception as e:
            st.warning("Could not generate visual explanation for this specific text.")
def predictor(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="np")
    onnx_inputs = {
        session.get_inputs()[0].name: inputs['input_ids'].astype(np.int64),
        session.get_inputs()[1].name: inputs['attention_mask'].astype(np.int64)
    }
    logits = session.run(None, onnx_inputs)[0]
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = e_x / e_x.sum(axis=1, keepdims=True)
    return probs

explainer = load_explainer()

def play_voice_alert(alert_type, lang_code):
    if lang_code == 'en':
        file_name = "alert_scam_en.mp3" if alert_type == 'Scam' else "alert_safe_en.mp3"
    else:
        file_name = "alert_scam_hi.mp3" if alert_type == 'Scam' else "alert_safe_hi.mp3"
    if os.path.exists(file_name):
        try:
            st.audio(file_name, format='audio/mp3', start_time=0, autoplay=True)
        except Exception as e:
            pass

with st.sidebar:
    st.header("Settings / सेटिंग्स")
    lang_choice = st.radio("Language / भाषा", ["English", "Hindi"])
    t = ui_text[lang_choice]
    lang_code = 'hi' if lang_choice == "Hindi" else 'en'
    st.divider()
    st.header("Architecture Specs")
    st.markdown("""
    * **Model:** XLM - RoBERTa
    * **Inference:** ONNX Runtime
    * **Privacy:** Local Inference
    * **Latency:** < 100ms
    """)

st.title(f"{t['title']}")
st.markdown(f"### {t['subtitle']}")

if "msg_analysis" not in st.session_state: st.session_state.msg_analysis = None
if "audio_analysis" not in st.session_state: st.session_state.audio_analysis = None

tab1, tab2 = st.tabs([t['tab1'], t['tab2']])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_area(t['input_label'], height=150, placeholder=t['placeholder'])
    
    if st.button(t['scan_btn']):
        if user_input:
            start_time = time.time()
            probs = predictor([user_input])[0]
            latency = (time.time() - start_time) * 1000
            t_lower = user_input.lower()
            
            st.session_state.msg_analysis = {
                "text": user_input,
                "raw_scam_score": probs[1],
                "raw_safe_score": probs[0],
                "is_scam": 1 if probs[1] > SCAM_THRESHOLD else 0,
                "latency": latency
            }
        else:
            st.warning("Please enter text first.")

    if st.session_state.msg_analysis:
        data = st.session_state.msg_analysis
        if data["raw_scam_score"] > SCAM_THRESHOLD:
            st.markdown(f'<p class="warning-text">{t["scam_alert"]} ({data["raw_scam_score"]*100:.1f}%)</p>', unsafe_allow_html=True)
            st.caption(f"Inference Time: {data['latency']:.1f} ms")
            play_voice_alert('Scam', lang_code)
            render_lime_explanation(data["text"], t)     
            
            st.divider()
            c1, c2 = st.columns([3,1])
            with c1: st.error(f"**Action Required:** High Risk detected.")
            with c2:
                if st.button(t['notify'], key="notify_btn_sms"):
                    st.toast(t['family_alert'])
                    time.sleep(1)
                    st.balloons()
                    
        # 3. SAFE
        else:
            st.markdown(f'<p class="safe-text">{t["safe_alert"]} ({data["raw_safe_score"]*100:.1f}%)</p>', unsafe_allow_html=True)
            st.caption(f"Inference Time: {data['latency']:.1f} ms")
            play_voice_alert('Safe', lang_code)

with tab2:
    st.info(f"**Voice - to - Text:** {t['audio_label']}")
    uploaded_file = st.file_uploader(t['audio_label'], type=["wav", "mp3"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button(t['analyze_audio']):
            with st.status(t['listening'], expanded=True) as status:
                st.write("Extracting audio channels...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:    
                    st.write("Running Faster-Whisper transcription")
                    segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
                    text = " ".join([segment.text for segment in segments])
                    
                    st.write("Passing transcript to XLM-RoBERTa")
                    probs = predictor([text])[0]
                    st.session_state.audio_analysis = {"text": text, "raw_scam_score": probs[1]}
                except Exception as e:
                    st.error("Audio file could not be processed. Please try another file.")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    if st.session_state.audio_analysis:
        data = st.session_state.audio_analysis
        st.markdown("### Transcript / प्रतिलेख")
        st.markdown(f"> *\"{data['text']}\"*")
        if data["raw_scam_score"] > SCAM_THRESHOLD:
            st.markdown(f'<p class="warning-text">{t["scam_alert"]} ({data["raw_scam_score"]*100:.1f}%)</p>', unsafe_allow_html=True)
            play_voice_alert('Scam', lang_code)
            render_lime_explanation(data["text"], t)        
            st.divider()
            c1, c2 = st.columns([3,1])
            with c1: st.error(f"**Action Required:** High Risk detected.")
            with c2:
                if st.button(t['notify'], key="notify_btn_audio"):
                    st.toast(t['family_alert'])
                    time.sleep(1)
                    st.balloons()
        else:
            st.markdown(f'<p class="safe-text">{t["safe_alert"]} ({data["raw_safe_score"]*100:.1f}%)</p>', unsafe_allow_html=True)
            play_voice_alert('Safe', lang_code)