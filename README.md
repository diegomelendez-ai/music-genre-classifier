# 🎵 Music Genre Classifier

Application web qui détecte le **genre musical** d'un extrait audio (`.wav`) parmi 10 catégories, à l'aide de modèles de Machine Learning entraînés sur le dataset **GTZAN**.

**Projet Final – Traitement du Signal Audio**
Diego MELENDEZ & Christophe STEVENS – Mastère Data & IA, NEXA Digital School

---

## ✨ Fonctionnalités

- Dépôt d'un fichier audio `.wav` (lecteur intégré)
- Choix du modèle : **SVM** (rapide) ou **réseau de neurones**
- Prédiction du genre avec **niveau de confiance**
- **Graphique** des probabilités pour les 10 genres
- **Mel-spectrogramme** de l'extrait
- Tableau des **58 features** extraites (tempo, MFCC, centroïde spectral, etc.)

## 🎸 Genres reconnus

Blues · Classique · Country · Disco · Hip-Hop · Jazz · Metal · Pop · Reggae · Rock

## 🛠️ Stack technique

- **Python**, **Streamlit** (interface web)
- **librosa** (extraction des features audio)
- **scikit-learn** (SVM, Random Forest, KNN, Decision Tree)
- **TensorFlow / Keras** (réseau de neurones)

## 📊 Performances (sur le jeu de test)

| Modèle | Précision |
|--------|-----------|
| Réseau de neurones | 91.7 % |
| SVM | 91.5 % |
| KNN | 89.7 % |
| Random Forest | 86.8 % |
| Decision Tree | 65.7 % |

## 🚀 Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Fichiers nécessaires

L'application a besoin des modèles entraînés (exportés depuis le notebook) :

- `svm_model.joblib`
- `scaler.joblib`
- `label_encoder.joblib`
- `ann_model.keras`

## ⚠️ Note honnête sur les performances

Le modèle est entraîné sur GTZAN (extraits de 3 secondes). Il est plus fiable sur des extraits similaires à ce dataset. Un morceau très différent (autre style de production, autre époque) peut être moins bien classé. Les scores élevés bénéficient aussi du fait que des extraits d'une même chanson se retrouvent à l'entraînement et au test ; un découpage par chanson donnerait une mesure plus stricte de la généralisation.
