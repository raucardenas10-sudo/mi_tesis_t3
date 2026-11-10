import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Configuración según la Observación 7 de la tesis
SEEDS = [42, 123, 777, 999, 2024]

print("📊 INICIANDO AUDITORÍA DE CURVA K-SAMPLES (FEW-SHOT CROSS-LINGUAL)")
print("=" * 70)

# 1. Cargar embeddings y dataset
try:
    df_unificado = pd.read_csv("data/processed/corpus_unificado_20k.csv")
    X_all = np.load("data/processed/X_cls_20k.npy")
    y_all = df_unificado['etiqueta'].values
    idioma_all = df_unificado['idioma'].values
except Exception as e:
    print(f"❌ Error al cargar los archivos procesados: {e}")
    exit()

# Separar por idiomas
mask_en = (idioma_all == 'en')
mask_es = (idioma_all == 'es')

X_en, y_en = X_all[mask_en], y_all[mask_en]
X_es, y_es = X_all[mask_es], y_all[mask_es]

# Reservar un conjunto de prueba fijo en español (30% del total en español) para evaluar siempre
X_es_train_pool, X_es_test_fixed, y_es_train_pool, y_es_test_fixed = train_test_split(
    X_es, y_es, test_size=0.30, random_state=42, stratify=y_es
)

max_k_disponible = len(X_es_train_pool)
print(f"🔹 Dataset de Entrenamiento en Inglés: {X_en.shape[0]} muestras")
print(f"🔹 Pool disponible para muestreo K en Español: {max_k_disponible} muestras")
print(f"🔹 Conjunto de prueba FIJO en Español (Evaluación): {X_es_test_fixed.shape[0]} muestras\n")

# Valores de K solicitados por el jurado (adaptados al pool disponible)
K_VALUES_SOLICITADOS = [0, 50, 100, 250, 500, 1000, 1400]
K_VALUES = [k for k in K_VALUES_SOLICITADOS if k <= max_k_disponible]
if max_k_disponible not in K_VALUES:
    K_VALUES.append(max_k_disponible)

print(f"📋 Valores de K que se evaluarán en la curva: {K_VALUES}\n")

# Estructuras para acumular resultados
acc_results = {k: [] for k in K_VALUES}
f1_results = {k: [] for k in K_VALUES}

# Bucle principal de evaluación multisemilla
for k in K_VALUES:
    for seed in SEEDS:
        np.random.seed(seed)
        
        # 1. Base de entrenamiento con 18,000 en inglés
        X_train_k = list(X_en)
        y_train_k = list(y_en)
        
        # 2. Agregar k muestras en español si k > 0
        if k > 0:
            if k >= len(X_es_train_pool):
                X_train_k.extend(X_es_train_pool)
                y_train_k.extend(y_es_train_pool)
            else:
                X_es_k, _, y_es_k, _ = train_test_split(
                    X_es_train_pool, y_es_train_pool, 
                    train_size=k, 
                    stratify=y_es_train_pool, 
                    random_state=seed
                )
                X_train_k.extend(X_es_k)
                y_train_k.extend(y_es_k)
            
        X_train_k = np.array(X_train_k)
        y_train_k = np.array(y_train_k)
        
        # 3. Escalado y entrenamiento SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_k)
        
        clf = SVC(kernel='rbf', C=1.0, random_state=seed)
        clf.fit(X_train_scaled, y_train_k)
        
        # 4. Evaluación SIEMPRE sobre el conjunto de prueba FIJO en Español
        X_es_test_scaled = scaler.transform(X_es_test_fixed)
        y_pred_es = clf.predict(X_es_test_scaled)
        
        acc = accuracy_score(y_es_test_fixed, y_pred_es)
        f1 = f1_score(y_es_test_fixed, y_pred_es, average='weighted')
        
        acc_results[k].append(acc)
        f1_results[k].append(f1)

# Procesar métricas agregadas (Media ± Desviación Estándar)
summary_data = []

print("📌 RESUMEN DE CURVA FEW-SHOT CROSS-LINGUAL (5 SEMILLAS):")
print("-" * 75)
print(f"{'K-Samples':<10} | {'Accuracy (%)':<25} | {'F1-Score Weighted (%)':<25}")
print("-" * 75)

for k in K_VALUES:
    acc_mean = np.mean(acc_results[k]) * 100
    acc_std = np.std(acc_results[k]) * 100
    f1_mean = np.mean(f1_results[k]) * 100
    f1_std = np.std(f1_results[k]) * 100
    
    summary_data.append({
        'k_samples': k,
        'accuracy_mean': acc_mean,
        'accuracy_std': acc_std,
        'f1_mean': f1_mean,
        'f1_std': f1_std
    })
    
    print(f"{k:<10} | {acc_mean:6.2f}% ± {acc_std:4.2f}%          | {f1_mean:6.2f}% ± {f1_std:4.2f}%")

print("-" * 75)

# Guardar informe CSV
os.makedirs("reports", exist_ok=True)
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv("reports/resultados_curva_k.csv", index=False)
print("💾 Resultados exportados a 'reports/resultados_curva_k.csv'")

# Generar y guardar la gráfica oficial
plt.figure(figsize=(9, 5.5))
k_arr = [item['k_samples'] for item in summary_data]
f1_means = [item['f1_mean'] for item in summary_data]
f1_stds = [item['f1_std'] for item in summary_data]

plt.errorbar(k_arr, f1_means, yerr=f1_stds, fmt='-o', color='#1E3A8A', ecolor='#93C5FD', elinewidth=2, capsize=4, label='F1-Score (Español)')
plt.axhline(y=90.0, color='r', linestyle='--', label='Meta F1 ≥ 90%')
plt.title('Curva de Transferencia Few-Shot Cross-Lingual (XLM-RoBERTa + SVM)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Número de muestras en Español agregadas al entrenamiento (k)', fontsize=10)
plt.ylabel('F1-Score en Test Fijo en Español (%)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig("reports/curva_k_samples.png", dpi=300)
print("📈 Gráfica guardada en 'reports/curva_k_samples.png'")
