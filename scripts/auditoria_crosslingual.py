import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

SEED = 42
np.random.seed(SEED)

print("=" * 95)
print("🌎 AUDITORÍA DE APORTE CROSS-LINGUAL Y LÍNEAS BASE (OBSERVACIÓN N.° 5)")
print("=" * 95)

# 1. CARGA DE DATOS Y EMBEDDINGS DEDUPLICADOS
df = pd.read_csv("data/processed/corpus_unificado_20k.csv")
X_cls_all = np.load("data/processed/X_cls_20k.npy")

indices_es = df[df['idioma'] == 'es'].index.values
indices_en = df[df['idioma'] == 'en'].index.values

# Unificar texto (asunto + cuerpo) para TF-IDF
df['texto_completo'] = df['asunto'].fillna('') + " " + df['cuerpo'].fillna('')

# Partición 70% Train (1,400 ES) y 30% Test (600 ES) en Español
idx_es_train, idx_es_test = train_test_split(
    indices_es, test_size=0.30, random_state=SEED, stratify=df.loc[indices_es, 'etiqueta']
)

y_train_en = df.loc[indices_en, 'etiqueta'].values
y_train_es = df.loc[idx_es_train, 'etiqueta'].values
y_train_cross = np.concatenate([y_train_en, y_train_es])
y_test_es = df.loc[idx_es_test, 'etiqueta'].values

textos_es_train = df.loc[idx_es_train, 'texto_completo'].values
textos_es_test = df.loc[idx_es_test, 'texto_completo'].values

# Preparar conjuntos de Embeddings XLM-R
X_tr_en = X_cls_all[indices_en]
X_tr_es = X_cls_all[idx_es_train]
X_te_es = X_cls_all[idx_es_test]

X_tr_cross = np.vstack([X_tr_en, X_tr_es])

# ------------------------------------------------------------------------------
# EXPERIMENTO 1: TF-IDF + Regresión Logística (Solo 1,400 ES)
# ------------------------------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000)
X_tr_tfidf = tfidf.fit_transform(textos_es_train)
X_te_tfidf = tfidf.transform(textos_es_test)

clf_tfidf = LogisticRegression(max_iter=1000, random_state=SEED)
clf_tfidf.fit(X_tr_tfidf, y_train_es)
y_pred_tfidf = clf_tfidf.predict(X_te_tfidf)
y_prob_tfidf = clf_tfidf.predict_proba(X_te_tfidf)[:, 1]

f1_tfidf = f1_score(y_test_es, y_pred_tfidf)
p_tfidf = precision_score(y_test_es, y_pred_tfidf)
r_tfidf = recall_score(y_test_es, y_pred_tfidf)
auc_tfidf = roc_auc_score(y_test_es, y_prob_tfidf)

# ------------------------------------------------------------------------------
# EXPERIMENTO 2: XLM-R + SVM (Solo 1,400 ES)
# ------------------------------------------------------------------------------
scaler_es = StandardScaler()
X_tr_es_sc = scaler_es.fit_transform(X_tr_es)
X_te_es_sc = scaler_es.transform(X_te_es)

clf_svm_es = SVC(kernel='rbf', probability=True, random_state=SEED)
clf_svm_es.fit(X_tr_es_sc, y_train_es)
y_pred_svm_es = clf_svm_es.predict(X_te_es_sc)
y_prob_svm_es = clf_svm_es.predict_proba(X_te_es_sc)[:, 1]

f1_svm_es = f1_score(y_test_es, y_pred_svm_es)
p_svm_es = precision_score(y_test_es, y_pred_svm_es)
r_svm_es = recall_score(y_test_es, y_pred_svm_es)
auc_svm_es = roc_auc_score(y_test_es, y_prob_svm_es)

# ------------------------------------------------------------------------------
# EXPERIMENTO 3: XLM-R + SVM Zero-Shot (Solo 18,000 EN -> Eval en 600 ES)
# ------------------------------------------------------------------------------
scaler_en = StandardScaler()
X_tr_en_sc = scaler_en.fit_transform(X_tr_en)
X_te_es_sc_en = scaler_en.transform(X_te_es)

clf_svm_zs = SVC(kernel='rbf', probability=True, random_state=SEED)
clf_svm_zs.fit(X_tr_en_sc, y_train_en)
y_pred_zs = clf_svm_zs.predict(X_te_es_sc_en)
y_prob_zs = clf_svm_zs.predict_proba(X_te_es_sc_en)[:, 1]

f1_zs = f1_score(y_test_es, y_pred_zs)
p_zs = precision_score(y_test_es, y_pred_zs)
r_zs = recall_score(y_test_es, y_pred_zs)
auc_zs = roc_auc_score(y_test_es, y_prob_zs)

# ------------------------------------------------------------------------------
# EXPERIMENTO 4: PROPUESTA TESIS (XLM-R + SVM Cross-Lingual Few-Shot: 18k EN + 1.4k ES)
# ------------------------------------------------------------------------------
scaler_cross = StandardScaler()
X_tr_cross_sc = scaler_cross.fit_transform(X_tr_cross)
X_te_es_sc_cross = scaler_cross.transform(X_te_es)

clf_svm_cross = SVC(kernel='rbf', probability=True, random_state=SEED)
clf_svm_cross.fit(X_tr_cross_sc, y_train_cross)
y_pred_cross = clf_svm_cross.predict(X_te_es_sc_cross)
y_prob_cross = clf_svm_cross.predict_proba(X_te_es_sc_cross)[:, 1]

f1_cross = f1_score(y_test_es, y_pred_cross)
p_cross = precision_score(y_test_es, y_pred_cross)
r_cross = recall_score(y_test_es, y_pred_cross)
auc_cross = roc_auc_score(y_test_es, y_prob_cross)

# ------------------------------------------------------------------------------
# IMPRESIÓN DE TABLA COMPARATIVA AUDITABLE
# ------------------------------------------------------------------------------
print("\n📊 CUADRO COMPARATIVO DE APORTE CROSS-LINGUAL Y LÍNEAS BASE")
print("=" * 95)
print(f"{'Estrategia / Modelo':<36} | {'Datos Entrenamiento':<20} | {'F1-Score':<8} | {'Precision':<9} | {'ROC-AUC':<8}")
print("-" * 95)
print(f"{'1. Zero-Shot (EN -> ES)':<36} | {'18 000 EN (0 ES)':<20} | {f1_zs:<8.4f} | {p_zs:<9.4f} | {auc_zs:<8.4f}")
print(f"{'2. Línea Base A: TF-IDF + Reg. Log.':<36} | {'1 400 ES (0 EN)':<20} | {f1_tfidf:<8.4f} | {p_tfidf:<9.4f} | {auc_tfidf:<8.4f}")
print(f"{'3. Línea Base B: XLM-R + SVM (Solo ES)':<36} | {'1 400 ES (0 EN)':<20} | {f1_svm_es:<8.4f} | {p_svm_es:<9.4f} | {auc_svm_es:<8.4f}")
print(f"{'4. Propuesta Tesis (Cross-Lingual)':<36} | {'18 000 EN + 1 400 ES':<20} | {f1_cross:<8.4f} | {p_cross:<9.4f} | {auc_cross:<8.4f}")
print("=" * 95)
