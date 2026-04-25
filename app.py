
import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os
import subprocess
import imageio_ffmpeg
from tensorflow.keras.models import load_model
from streamlit_mic_recorder import mic_recorder

# ✅ 1. تحميل الموديل
try:
    model = load_model('emotion_model_DL.keras')
    scaler = joblib.load('scaler_DL.pkl')
    le = joblib.load('label_encoder_DL.pkl')
except Exception as e:
    st.error(f"❌ خطأ في تحميل ملفات الموديل: {e}")
    st.stop()

# ✅ 2. تنسيق المشاعر
emotion_style = {
    'ang': ('😠 Angry',   '#FF4444'),
    'hap': ('😊 Happy',   '#FFD700'),
    'sad': ('😢 Sad',     '#4444FF'),
    'neu': ('😐 Neutral', '#888888'),
}

# ✅ 3. استخراج المميزات
def extract_features(audio, sr):
    try:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)

        pitches, _ = librosa.piptrack(y=audio, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))

        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(
            y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)

        return np.hstack([mfccs, chroma, mel, zcr, rms,
                          [pitch],
                          [spectral_centroid, spectral_bandwidth, spectral_rolloff],
                          contrast, tonnetz])
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

# ✅ 4. الواجهة
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤")

# ── اللوجو
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo.png", width=200)

st.markdown("<h1 style='text-align:center; color:#7C83FD;'>🎤 Speech Emotion Recognition</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray;'>Egyptian Arabic Dialect</h4>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("### 🎙️ Record your voice and detect your emotion!")

# ✅ 5. التسجيل
audio = mic_recorder(
    start_prompt="🔴 Start Recording",
    stop_prompt="⏹ Stop Recording",
    just_once=True,
    key="recorder"
)

if audio:
    st.audio(audio['bytes'], format='audio/wav')

    with st.spinner('⚙️ Analyzing emotion...'):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
                tmp_in.write(audio['bytes'])
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path.replace(".webm", ".wav")
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            subprocess.run([
                ffmpeg_exe, '-y',
                '-i', tmp_in_path,
                '-ar', '22050',
                '-ac', '1',
                tmp_out_path
            ], capture_output=True)

            y, sr = librosa.load(tmp_out_path, sr=22050)

            os.remove(tmp_in_path)
            os.remove(tmp_out_path)

            feat = extract_features(y, sr)

            if feat is not None:
                feat_scaled = scaler.transform([feat])
                pred_probs = model.predict(feat_scaled)
                pred_index = np.argmax(pred_probs, axis=1)[0]
                emotion = le.inverse_transform([pred_index])[0]

                label, color = emotion_style.get(emotion, (f'🎭 {emotion}', '#333333'))

                st.markdown(f"""
                <div style='text-align:center; padding:40px; border-radius:15px;
                            background-color:{color}22; border: 2px solid {color}; margin-top:20px;'>
                    <h1 style='color:{color}; font-size:60px;'>{label}</h1>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ تعذر استخراج المميزات، جربي التسجيل مرة أخرى!")

        except Exception as e:
            st.error(f"❌ حدث خطأ: {e}")
            import traceback
            st.code(traceback.format_exc())

# ── Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; line-height:1.8;'>
    <p>Graduation Project 2026</p>
    <p><b>Shahd DiefAllah</b></p>
    <p>Supervised by <b>Dr. Sameh El-Ansary</b></p>
    <p>Phonetics and Linguistics Department</p>
    <p>Faculty of Arts — Alexandria University</p>
</div>
""", unsafe_allow_html=True)