"""
Music Genre Classifier - Application Streamlit
Projet Final - Traitement du Signal Audio
Diego MELENDEZ et Christophe STEVENS - NEXA Digital School

Une app qui detecte le genre musical d'un extrait audio (.wav)
a l'aide de modeles entraines sur le dataset GTZAN (10 genres).
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib

# ------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
)

GENRES_FR = {
    'blues': 'Blues', 'classical': 'Classique', 'country': 'Country',
    'disco': 'Disco', 'hiphop': 'Hip-Hop', 'jazz': 'Jazz',
    'metal': 'Metal', 'pop': 'Pop', 'reggae': 'Reggae', 'rock': 'Rock',
}

FEATURE_NAMES = [
    'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean',
    'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var',
    'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean',
    'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
] + [f'mfcc{i}_{stat}' for i in range(1, 21) for stat in ('mean', 'var')]


# ------------------------------------------------------------
# Chargement des modeles (mis en cache)
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    svm = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    ann = None
    try:
        from tensorflow.keras.models import load_model
        ann = load_model('ann_model.keras')
    except Exception:
        ann = None  # si TensorFlow n'est pas disponible, l'app marche quand meme avec le SVM
    return svm, scaler, le, ann


# ------------------------------------------------------------
# Extraction des 58 features (meme ordre que features_3_sec.csv)
# ------------------------------------------------------------
def extract_features(y, sr):
    f = []
    f.append(len(y))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr); f += [chroma.mean(), chroma.var()]
    rms = librosa.feature.rms(y=y); f += [rms.mean(), rms.var()]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr); f += [cent.mean(), cent.var()]
    band = librosa.feature.spectral_bandwidth(y=y, sr=sr); f += [band.mean(), band.var()]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr); f += [roll.mean(), roll.var()]
    zcr = librosa.feature.zero_crossing_rate(y); f += [zcr.mean(), zcr.var()]
    y_harm, y_perc = librosa.effects.hpss(y)
    f += [y_harm.mean(), y_harm.var(), y_perc.mean(), y_perc.var()]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    f.append(float(np.atleast_1d(tempo)[0]))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        f += [mfcc[i].mean(), mfcc[i].var()]
    return np.array(f, dtype=float)


# ------------------------------------------------------------
# Interface
# ------------------------------------------------------------
st.title("🎵 Music Genre Classifier")
st.markdown(
    "Déposez un extrait audio (`.wav`) et le modèle prédit son **genre musical** "
    "parmi 10 catégories. Entraîné sur le dataset **GTZAN**."
)

svm, scaler, le, ann = load_models()

# Barre laterale : choix du modele
st.sidebar.header("⚙️ Paramètres")
modeles_dispo = ["SVM (rapide, 91.5%)"]
if ann is not None:
    modeles_dispo.append("Réseau de neurones (91.7%)")
choix_modele = st.sidebar.radio("Choisir le modèle :", modeles_dispo)

st.sidebar.markdown("---")
st.sidebar.caption(
    "ℹ️ Le modèle est entraîné sur GTZAN (extraits de 3 s). "
    "Il est plus fiable sur des extraits similaires ; un morceau très différent "
    "peut être moins bien classé."
)

# Zone d'upload
fichier = st.file_uploader("📁 Déposez un fichier .wav", type=['wav'])

if fichier is not None:
    # Lecture audio
    st.audio(fichier, format='audio/wav')

    with st.spinner("Analyse de l'audio en cours..."):
        y, sr = librosa.load(fichier, sr=22050)
        feats = extract_features(y, sr)
        feats_scaled = scaler.transform(feats.reshape(1, -1))

        # Prediction selon le modele choisi
        if choix_modele.startswith("SVM"):
            proba = svm.predict_proba(feats_scaled)[0]
        else:
            proba = ann.predict(feats_scaled, verbose=0)[0]

        idx = int(np.argmax(proba))
        genre_pred = le.classes_[idx]
        confiance = float(proba[idx])

    # Resultat principal
    st.success(f"### 🎯 Genre prédit : **{GENRES_FR.get(genre_pred, genre_pred)}**  ({confiance:.1%} de confiance)")

    col1, col2 = st.columns(2)

    # Graphe des probabilites
    with col1:
        st.subheader("Probabilités par genre")
        proba_df = pd.DataFrame({
            'Genre': [GENRES_FR.get(c, c) for c in le.classes_],
            'Probabilité': proba,
        }).sort_values('Probabilité', ascending=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(proba_df['Genre'], proba_df['Probabilité'], color='steelblue')
        ax.set_xlabel('Probabilité')
        ax.set_xlim(0, 1)
        st.pyplot(fig)

    # Spectrogramme Mel
    with col2:
        st.subheader("Mel-spectrogramme")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax2)
        fig2.colorbar(img, ax=ax2, format='%+2.0f dB')
        st.pyplot(fig2)

    # Features visibles (section depliable)
    with st.expander("🔬 Voir les features extraites"):
        st.markdown("Quelques descripteurs clés extraits de l'audio :")
        apercu = {
            'Tempo (BPM)': round(feats[17], 1),
            'Centroïde spectral (Hz)': round(feats[5], 1),
            'Rolloff (Hz)': round(feats[9], 1),
            'Largeur de bande (Hz)': round(feats[7], 1),
            'Zero Crossing Rate': round(feats[11], 4),
            'RMS (énergie)': round(feats[3], 4),
            'Chroma STFT': round(feats[1], 4),
        }
        st.table(pd.DataFrame(apercu.items(), columns=['Feature', 'Valeur']))

        st.markdown("**Les 58 features complètes :**")
        full_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Valeur': np.round(feats, 4)})
        st.dataframe(full_df, height=300, use_container_width=True)

else:
    st.info("👆 Déposez un fichier .wav pour commencer. "
            "Vous pouvez utiliser un extrait du dataset GTZAN ou n'importe quel autre clip audio.")

# Pied de page
st.markdown("---")
st.caption("Projet Final - Traitement du Signal Audio | MELENDEZ & STEVENS | NEXA Digital School")
