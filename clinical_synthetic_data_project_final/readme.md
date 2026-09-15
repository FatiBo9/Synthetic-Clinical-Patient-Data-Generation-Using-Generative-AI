# 🧬 Génération de Données Cliniques Synthétiques par IA Générative

## 📌 Présentation
Développement d'un système de génération de données cliniques synthétiques sous contraintes médicales strictes. Le projet génère des profils réalistes de patients (sains et atteints de pathologies telles que le diabète, l'hypertension, l'obésité ou l'hypercholestérolémie) tout en préservant la confidentialité des données. Une évaluation approfondie par analyse statistique (filtre de rejet, règles de plausibilité clinique) et modèles de Machine Learning (classification & régression avec Logistic Regression, Random Forest, MLP) garantit la fidélité et l'utilité du dataset produit.

## 🚀 Fonctionnalités
- Génération sous contraintes de distributions de paramètres médicaux réels via **Gaussian Copula** et **CTGAN**.
- Pipeline de validation clinique avec filtre de rejet sur les bornes physiologiques et la cohérence inter-variables.
- Analyse statistique comparative et vérification des motifs épidémiologiques.
- Évaluation ML (Cross-Validation 5-fold et test de transférabilité entre Copule et CTGAN).
- Interface graphique d'interaction et de démonstration sous Tkinter.

## 🛠️ Technologies & Outils
- **Langage :** Python
- **IA Générative & Data :** SDV (Synthetic Data Vault), CTGAN, Copulas, Pandas, NumPy
- **Machine Learning & Évaluation :** Scikit-learn, SciPy
- **Visualisation & UI :** Matplotlib, Seaborn, Tkinter

## ⚙️ Installation & Lancement

```bash
# 1. Cloner le dépôt
git clone [https://github.com/FatiBo9/G-n-ration-de-donn-es-cliniques-synth-tiques-de-patients--l-aide-de-l-IA-g-n-rative.git](https://github.com/FatiBo9/G-n-ration-de-donn-es-cliniques-synth-tiques-de-patients--l-aide-de-l-IA-g-n-rative.git)
cd G-n-ration-de-donn-es-cliniques-synth-tiques-de-patients--l-aide-de-l-IA-g-n-rative

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'interface graphique Tkinter
python app.py