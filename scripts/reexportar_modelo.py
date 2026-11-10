import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

print("🔄 Cargando dataset y embeddings [CLS]...")
df = pd.read_csv("data/processed/corpus_unificado_20k.csv")
X_cls = np.load("data/processed/X_cls_20k.npy")
y = df['etiqueta'].values

# Separar train y test (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y, test_size=0.30, random_state=SEED, stratify=y
)

print("📏 Ajustando StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("⚙️ Entrenando SVM (Kernel RBF, C=1.0)...")
clf = SVC(kernel='rbf', probability=True, random_state=SEED)
clf.fit(X_train_scaled, y_train)

# Guardar artefactos alineados
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(clf, "models/clasificador_phishing.joblib")

print("✅ ¡Artefactos 'scaler.joblib' y 'clasificador_phishing.joblib' re-exportados con éxito!")
