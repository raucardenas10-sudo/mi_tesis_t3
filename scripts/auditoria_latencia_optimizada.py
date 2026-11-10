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
MAX_LEN_OPTIMIZADO = 128

print("=" * 80)
print("⚡ AUDITORÍA DE LATENCIA OPTIMIZADA (OPCIÓN A: INT8 QUANTIZATION + MAX_LEN 128)")
print("=" * 80)

# 1. CARGA Y CUANTIZACIÓN DEL MODELO
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_transformer = AutoModel.from_pretrained(MODEL_NAME)
model_transformer.eval()

# Aplicar Cuantización INT8 para CPU
print("🔧 Aplicando Cuantización Dinámica INT8 en XLM-RoBERTa...")
model_quantized = torch.quantization.quantize_dynamic(
    model_transformer, {torch.nn.Linear}, dtype=torch.qint8
)

scaler = joblib.load("models/scaler.joblib")
svm_model = joblib.load("models/clasificador_phishing.joblib")

df = pd.read_csv("data/processed/corpus_unificado_20k.csv")
sample_emails = df.sample(n=N_SAMPLES, random_state=SEED)[['asunto', 'cuerpo']].values

print(f"🧪 Midiendo latencia optimizada sobre {N_SAMPLES} correos (max_length={MAX_LEN_OPTIMIZADO})...")

latencias_totales = []

# Warm-up (calentamiento de memoria)
text_warm = str(sample_emails[0][0]) + " " + str(sample_emails[0][1])
inputs_w = tokenizer(text_warm, return_tensors="pt", truncation=True, max_length=MAX_LEN_OPTIMIZADO, padding=True)
with torch.no_grad():
    out_w = model_quantized(**inputs_w)
    emb_w = out_w.last_hidden_state[:, 0, :].numpy()
    _ = svm_model.predict(scaler.transform(emb_w))

# MEDICIÓN INDIVIDUAL
for asunto, cuerpo in sample_emails:
    texto_completo = f"{asunto} {cuerpo}".strip()
    
    t0 = time.perf_counter()
    
    inputs = tokenizer(texto_completo, return_tensors="pt", truncation=True, max_length=MAX_LEN_OPTIMIZADO, padding=True)
    with torch.no_grad():
        outputs = model_quantized(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
    emb_scaled = scaler.transform(cls_embedding)
    prediccion = svm_model.predict(emb_scaled)
    
    t1 = time.perf_counter()
    latencias_totales.append((t1 - t0) * 1000)

mediana_opt = np.median(latencias_totales)
p95_opt = np.percentile(latencias_totales, 95)
promedio_opt = np.mean(latencias_totales)

print("\n" + "=" * 80)
print("📊 RESULTADOS OPTIMIZADOS (OPCIÓN A)")
print("=" * 80)
print(f" • Tiempo Promedio Total:      {promedio_opt:.2f} ms")
print(f" • MEDIANA (P50) Total:        {mediana_opt:.2f} ms")
print(f" • PERCENTIL 95 (P95) Total:   {p95_opt:.2f} ms")
print("=" * 80)
