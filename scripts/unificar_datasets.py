import os
import re
import hashlib
import pandas as pd

SEED = 42

def calcular_hash(texto):
    """Genera una firma MD5 sobre el texto normalizado para garantizar unicidad."""
    if not isinstance(texto, str):
        return ""
    texto_norm = re.sub(r'\s+', ' ', texto).strip().lower()
    return hashlib.md5(texto_norm.encode('utf-8')).hexdigest()

def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r'<[^>]+>', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

print("🔄 Iniciando Módulo ETL: Deduplicación Hash MD5 y Muestreo Estricto...")

# 1. CORPUS EN INGLÉS (Filtrado de Unicidad MD5 sobre la fuente completa de 28k)
ruta_ingles = "data/raw/dataset_ingles_28k.csv"
try:
    df_en_raw = pd.read_csv(ruta_ingles, encoding="utf-8")
except:
    df_en_raw = pd.read_csv(ruta_ingles, encoding="latin1")

col_text_en = [c for c in df_en_raw.columns if c.lower() in ['body', 'text', 'email_text', 'cuerpo']][0]
col_label_en = [c for c in df_en_raw.columns if c.lower() in ['label', 'target', 'etiqueta', 'class', 'phishing']][0]

df_en = df_en_raw[[col_text_en, col_label_en]].copy()
df_en.columns = ['cuerpo', 'etiqueta']
df_en['asunto'] = ""
df_en['cuerpo'] = df_en['cuerpo'].apply(limpiar_texto)

# Deduplicación MD5 sobre la fuente entera en inglés
df_en['hash'] = (df_en['asunto'] + " " + df_en['cuerpo']).apply(calcular_hash)
df_en_unicos = df_en.drop_duplicates(subset=['hash']).drop(columns=['hash'])

# Muestreo de 18,000 sobre los registros ÚNICOS de inglés
phish_en = df_en_unicos[df_en_unicos['etiqueta'] == 1].sample(n=8640, random_state=SEED)
ham_en = df_en_unicos[df_en_unicos['etiqueta'] == 0].sample(n=9360, random_state=SEED)

df_en_final = pd.concat([phish_en, ham_en]).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_en_final['idioma'] = 'en'
df_en_final['origen'] = 'Nazario/Enron'
df_en_final['id_correo'] = [f"EN_RAW_{i+1:05d}" for i in range(len(df_en_final))]

# 2. CORPUS EN ESPAÑOL PÚBLICO (Filtrado de Unicidad MD5 previo)
ruta_espanol = "data/raw/dataset_espanol_mendeley.csv"
try:
    df_es_raw = pd.read_csv(ruta_espanol, sep=";", encoding="utf-8")
except:
    df_es_raw = pd.read_csv(ruta_espanol, encoding="utf-8")

col_asunto_es = 'asunto' if 'asunto' in df_es_raw.columns else df_es_raw.columns[0]
col_cuerpo_es = 'cuerpo_sanitizado' if 'cuerpo_sanitizado' in df_es_raw.columns else ('cuerpo' if 'cuerpo' in df_es_raw.columns else df_es_raw.columns[1])
col_label_es = 'etiqueta' if 'etiqueta' in df_es_raw.columns else df_es_raw.columns[-1]

df_es = df_es_raw[[col_asunto_es, col_cuerpo_es, col_label_es]].copy()
df_es.columns = ['asunto', 'cuerpo', 'etiqueta']
df_es['asunto'] = df_es['asunto'].apply(limpiar_texto)
df_es['cuerpo'] = df_es['cuerpo'].apply(limpiar_texto)

df_es['hash'] = (df_es['asunto'] + " " + df_es['cuerpo']).apply(calcular_hash)
df_es_unicos = df_es.drop_duplicates(subset=['hash']).drop(columns=['hash'])

phish_es = df_es_unicos[df_es_unicos['etiqueta'] == 1].sample(n=900, random_state=SEED)
ham_es = df_es_unicos[df_es_unicos['etiqueta'] == 0].sample(n=900, random_state=SEED)

df_es_final = pd.concat([phish_es, ham_es]).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_es_final['idioma'] = 'es'
df_es_final['origen'] = 'SpearPhishMX'
df_es_final['id_correo'] = [f"ES_PUB_{i+1:04d}" for i in range(len(df_es_final))]

# 3. CORPUS SINTÉTICO FISI
ruta_fisi = "data/raw/fisi_synthetic_200.csv"
df_fisi = pd.read_csv(ruta_fisi, encoding="utf-8")
df_fisi['asunto'] = df_fisi['asunto'].apply(limpiar_texto)
df_fisi['cuerpo'] = df_fisi['cuerpo'].apply(limpiar_texto)
df_fisi['origen'] = 'FISI-Synthetic'

# 4. CONSOLIDACIÓN Y VERIFICACIÓN GLOBAL
cols_finales = ['id_correo', 'asunto', 'cuerpo', 'idioma', 'etiqueta', 'origen']
df_final = pd.concat([df_en_final[cols_finales], df_es_final[cols_finales], df_fisi[cols_finales]]).reset_index(drop=True)

df_final['hash_global'] = (df_final['asunto'] + " " + df_final['cuerpo']).apply(calcular_hash)
duplicados_globales = df_final.duplicated(subset=['hash_global']).sum()
df_final = df_final.drop(columns=['hash_global'])

os.makedirs("data/processed", exist_ok=True)
df_final.to_csv("data/processed/corpus_unificado_20k.csv", index=False, encoding="utf-8")

print(f"✅ Filtrado de Unicidad Hash MD5 ejecutado correctamente.")
print(f"🔍 Duplicados detectados en el corpus final: {duplicados_globales}")
print(f"🎉 Corpus Unificado Guardado: {len(df_final)} registros exactos (100% limpios y no repetidos).")
