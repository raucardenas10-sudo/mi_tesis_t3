import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve

SEED = 42
np.random.seed(SEED)

print("🚀 Cargando datos deduplicados y matrices de embeddings de 20k...")
df = pd.read_csv("data/processed/corpus_unificado_20k.csv")
X_mean_all = np.load("data/processed/X_mean_20k.npy")
X_cls_all = np.load("data/processed/X_cls_20k.npy")

indices_es = df[df['idioma'] == 'es'].index.values
indices_en = df[df['idioma'] == 'en'].index.values

# Partición Estratificada 70/30 del corpus en español
idx_es_train, idx_es_test = train_test_split(
    indices_es, test_size=0.30, random_state=SEED, stratify=df.loc[indices_es, 'etiqueta']
)

y_train_cross = np.concatenate([df.loc[indices_en, 'etiqueta'].values, df.loc[idx_es_train, 'etiqueta'].values])
y_test_es = df.loc[idx_es_test, 'etiqueta'].values

# ==============================================================================
# FASE 1: COMPARATIVA DINÁMICA DE VECTORIZACIÓN
# ==============================================================================
print("\n" + "="*85)
print("🔬 FASE 1: COMPARATIVA DINÁMICA DE ESTRATEGIAS DE VECTORIZACIÓN")
print("="*85)

eval_fase1 = {}
for nombre_est, X_all in [("Mean Pooling", X_mean_all), ("Token [CLS]", X_cls_all)]:
    X_tr = np.vstack([X_all[indices_en], X_all[idx_es_train]])
    X_te = X_all[idx_es_test]
    
    scaler_temp = StandardScaler()
    X_tr_sc = scaler_temp.fit_transform(X_tr)
    X_te_sc = scaler_temp.transform(X_te)
    
    clf_temp = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    clf_temp.fit(X_tr_sc, y_train_cross)
    y_pred_temp = clf_temp.predict(X_te_sc)
    
    f1_t = f1_score(y_test_es, y_pred_temp)
    p_t = precision_score(y_test_es, y_pred_temp)
    r_t = recall_score(y_test_es, y_pred_temp)
    eval_fase1[nombre_est] = (f1_t, p_t, r_t)

mej_f1 = max(eval_fase1["Mean Pooling"][0], eval_fase1["Token [CLS]"][0])

for nombre_est in ["Mean Pooling", "Token [CLS]"]:
    f1_t, p_t, r_t = eval_fase1[nombre_est]
    estado_f1 = "SELECCIONADO" if f1_t == mej_f1 else "DESCARTADO"
    print(f"{nombre_est:<22} | {f1_t:<8.4f} | {p_t:<9.4f} | {r_t:<7.4f} | {estado_f1}")
print("="*85 + "\n")

# ==============================================================================
# FASE 2: BENCHMARK Y GRAFICACIÓN ROC-AUC
# ==============================================================================
X_train_cross = np.vstack([X_mean_all[indices_en], X_mean_all[idx_es_train]])
X_test_es = X_mean_all[idx_es_test]

scaler_cross = StandardScaler()
X_train_cross_sc = scaler_cross.fit_transform(X_train_cross)
X_test_es_sc = scaler_cross.transform(X_test_es)

modelos = {
    "SVM (RBF)": SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED),
    "Regresión Logística": LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=25, random_state=SEED, n_jobs=-1),
    "Gaussian Naive Bayes": GaussianNB()
}

resultados = []
probs_modelos = {}
mejor_f1_score = 0
mejor_modelo_obj = None
nombre_ganador = ""

print("⏳ Evaluando Benchmark de Clasificadores...")

plt.figure(figsize=(8, 6))

for nombre, clf in modelos.items():
    clf.fit(X_train_cross_sc, y_train_cross)
    
    start_time = time.time()
    y_pred = clf.predict(X_test_es_sc)
    end_time = time.time()
    
    latencia_ms = ((end_time - start_time) / len(X_test_es)) * 1000
    
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test_es_sc)[:, 1]
        auc = roc_auc_score(y_test_es, y_prob)
        fpr, tpr, _ = roc_curve(y_test_es, y_prob)
        plt.plot(fpr, tpr, label=f"{nombre} (AUC = {auc:.4f})")
    else:
        auc = 0.0
        
    f1 = f1_score(y_test_es, y_pred)
    prec = precision_score(y_test_es, y_pred)
    rec = recall_score(y_test_es, y_pred)
    
    cumple_f1 = f1 >= 0.90
    cumple_lat = latencia_ms <= 50.0
    
    resultados.append({
        "Modelo": nombre,
        "F1-Score": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC-AUC": auc,
        "Latencia (ms)": latencia_ms,
        "Cumple RNF": "SÍ" if (cumple_f1 and cumple_lat) else "NO"
    })
    
    if f1 > mejor_f1_score:
        mejor_f1_score = f1
        mejor_modelo_obj = clf
        nombre_ganador = nombre

# Configuración y guardado de la Curva ROC
plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio (AUC = 0.50)')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curvas ROC - Benchmark de Modelos Evaluados')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

os.makedirs("reports", exist_ok=True)
plt.savefig("reports/curva_roc_auc.png", dpi=300, bbox_inches='tight')
plt.close()

# Exportar Artefactos
os.makedirs("models", exist_ok=True)
joblib.dump(mejor_modelo_obj, "models/clasificador_phishing.joblib")
joblib.dump(scaler_cross, "models/scaler.joblib")

print("\n" + "="*85)
print("📊 FASE 2: CUADRO COMPARATIVO BENCHMARK DE MODELOS")
print("="*85)
for r in resultados:
    estado = " SELECCIONADO" if r["Modelo"] == nombre_ganador else (" CUMPLE" if r["Cumple RNF"] == "SÍ" else " DESCARTADO")
    print(f"{r['Modelo']:<22} | {r['F1-Score']:<8.4f} | {r['Precision']:<9.4f} | {r['Recall']:<7.4f} | {r['ROC-AUC']:<8.4f} | {r['Latencia (ms)']:<8.2f} | {estado}")

print("="*85)
print(f"\n🏆 MODELO GANADOR SELECCIONADO: {nombre_ganador}")
print(f"🖼️ Imagen de la Curva ROC guardada en: 'reports/curva_roc_auc.png'\n")
