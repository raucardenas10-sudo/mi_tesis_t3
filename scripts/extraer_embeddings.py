import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)

print("🚀 Iniciando extracción semántica vectorial con XLM-RoBERTa Base (PyTorch)...")

df = pd.read_csv("data/processed/corpus_unificado_20k.csv")

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# Desactivar gradientes para optimizar memoria y CPU
for param in model.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extraer_vectores(textos, batch_size=16):
    mean_embeddings = []
    cls_embeddings = []
    
    # Barra de progreso en tiempo real con tqdm
    for i in tqdm(range(0, len(textos), batch_size), desc="⏳ Lotes procesados"):
        lote = textos[i:i+batch_size]
        # Optimizado a max_length=128 para acelerar el cálculo en CPU
        inputs = tokenizer(lote, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            
            # 1. Estrategia Mean Pooling
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
            
            # 2. Estrategia Token [CLS]
            cls_pooled = token_embeddings[:, 0, :].cpu().numpy()
            
            mean_embeddings.append(mean_pooled)
            cls_embeddings.append(cls_pooled)
            
    return np.vstack(mean_embeddings), np.vstack(cls_embeddings)

textos_completos = (df['asunto'].fillna('') + " " + df['cuerpo'].fillna('')).str.strip().tolist()

print(f"📦 Total de correos a procesar: {len(textos_completos)}")
X_mean, X_cls = extraer_vectores(textos_completos)

os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/X_mean_20k.npy", X_mean)
np.save("data/processed/X_cls_20k.npy", X_cls)

print("\n✅ Extracción finalizada de raíz:")
print(f"   - Embeddings Mean Pooling guardados en: 'data/processed/X_mean_20k.npy' ({X_mean.shape})")
print(f"   - Embeddings Token [CLS] guardados en: 'data/processed/X_cls_20k.npy' ({X_cls.shape})")
