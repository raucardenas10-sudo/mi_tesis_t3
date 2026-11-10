import time
import os
import platform
import multiprocessing
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

SEED = 42
N_SAMPLES = 100

print("=" * 80)
print("⏱️ AUDITORÍA DE LATENCIA END-TO-END (TOKENIZACIÓN + XLM-R + ESCALADOR + SVM)")
print("=" * 80)

# 1. DETECCIÓN DE HARDWARE (NATIVO PYTHON)
cpu_info = platform.processor() or "Procesador Estándar"
cpu_count = multiprocessing.cpu_count()
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_info = torch.cuda.get_device_name(0) if device == "cuda" else "No (Ejecución en CPU)"

print(f"💻 Especificaciones del Hardware de Evaluación:")
print(f"   • Sistema Operativo: {platform.system()} {platform.release()}")
print(f"   • Procesador (CPU): {cpu_info} ({cpu_count} núcleos lógicos)")
print(f"   • Acelerador GPU:   {gpu_info}")
print("=" * 80)

# 2. CARGA DE MODELOS Y ARTEFACTOS
print("📦 Cargando modelo transformer XLM-RoBERTa y clasificadores guardados...")
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_transformer = AutoModel.from_pretrained(MODEL_NAME).to(device)
model_transformer.eval()

scaler = joblib.load("models/scaler.joblib")
svm_model = joblib.load("models/clasificador_phishing.joblib")

# 3. CARGA DE CORREOS DE PRUEBA
df = pd.read_csv("data/processed/corpus_unificado_20k.csv")
sample_emails = df.sample(n=N_SAMPLES, random_state=SEED)[['asunto', 'cuerpo']].values

print(f"🧪 Iniciando medición de latencia individual sobre {N_SAMPLES} correos de prueba...")

latencias_totales = []
latencias_tokenizacion = []
latencias_transformer = []
latencias_clasificador = []

# Warm-up (calentamiento de memoria para evitar sesgo inicial)
text_warm = str(sample_emails[0][0]) + " " + str(sample_emails[0][1])
inputs_w = tokenizer(text_warm, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
with torch.no_grad():
    out_w = model_transformer(**inputs_w)
    emb_w = out_w.last_hidden_state[:, 0, :].cpu().numpy()
    _ = svm_model.predict(scaler.transform(emb_w))

# 4. MEDICIÓN INDIVIDUAL END-TO-END
for i, (asunto, cuerpo) in enumerate(sample_emails):
    texto_completo = f"{asunto} {cuerpo}".strip()
    
    t0 = time.perf_counter()
    
    # Paso A: Tokenización
    t_tok_start = time.perf_counter()
    inputs = tokenizer(texto_completo, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
    t_tok_end = time.perf_counter()
    
    # Paso B: XLM-RoBERTa (Extracción de Embedding Token [CLS])
    t_tf_start = time.perf_counter()
    with torch.no_grad():
        outputs = model_transformer(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    t_tf_end = time.perf_counter()
    
    # Paso C: Escalador + Inferencia SVM RBF
    t_clf_start = time.perf_counter()
    emb_scaled = scaler.transform(cls_embedding)
    prediccion = svm_model.predict(emb_scaled)
    t_clf_end = time.perf_counter()
    
    t1 = time.perf_counter()
    
    # Cálculo de tiempos en milisegundos (ms)
    time_total = (t1 - t0) * 1000
    time_tok = (t_tok_end - t_tok_start) * 1000
    time_tf = (t_tf_end - t_tf_start) * 1000
    time_clf = (t_clf_end - t_clf_start) * 1000
    
    latencias_totales.append(time_total)
    latencias_tokenizacion.append(time_tok)
    latencias_transformer.append(time_tf)
    latencias_clasificador.append(time_clf)

# 5. CÁLCULO DE ESTADÍSTICAS METROLÓGICAS
mediana_total = np.median(latencias_totales)
p95_total = np.percentile(latencias_totales, 95)
promedio_total = np.mean(latencias_totales)

print("\n" + "=" * 80)
print("📊 RESULTADOS DE LATENCIA END-TO-END POR CORREO (100 MUESTRAS INDIVIDUALES)")
print("=" * 80)
print(f" • Tiempo Promedio Total:      {promedio_total:.2f} ms")
print(f" • MEDIANA (P50) Total:        {mediana_total:.2f} ms")
print(f" • PERCENTIL 95 (P95) Total:   {p95_total:.2f} ms")
print("-" * 80)
print(f" Desglose por Etapa del Pipeline (Mediana):")
print(f"   1. Tokenización (Tokenizer):              {np.median(latencias_tokenizacion):.2f} ms")
print(f"   2. Extracción XLM-RoBERTa ([CLS]):        {np.median(latencias_transformer):.2f} ms")
print(f"   3. Clasificación SVM RBF + Scaler:       {np.median(latencias_clasificador):.2f} ms")
print("=" * 80)
