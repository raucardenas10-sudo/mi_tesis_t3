import os
import re
import pandas as pd

SEED = 42

def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r'<[^>]+>', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

print("🔄 Iniciando proceso ETL y consolidación del corpus de 20,000 correos...")

# 1. CORPUS INGLÉS (18,000 REGISTROS EXACTOS)
ruta_ingles = "data/raw/dataset_ingles_28k.csv"
try:
    df_en_raw = pd.read_csv(ruta_ingles, encoding="utf-8")
except:
    df_en_raw = pd.read_csv(ruta_ingles, encoding="latin1")

col_text_en = [c for c in df_en_raw.columns if c.lower() in ['body', 'text', 'email_text', 'cuerpo', 'text_combined']][0] if any(c.lower() in ['body', 'text', 'email_text', 'cuerpo', 'text_combined'] for c in df_en_raw.columns) else df_en_raw.columns[0]
col_label_en = [c for c in df_en_raw.columns if c.lower() in ['label', 'target', 'etiqueta', 'class', 'phishing']][0] if any(c.lower() in ['label', 'target', 'etiqueta', 'class', 'phishing'] for c in df_en_raw.columns) else df_en_raw.columns[-1]

df_en = df_en_raw[[col_text_en, col_label_en]].copy()
df_en.columns = ['cuerpo', 'etiqueta']
df_en['asunto'] = ""

df_en['cuerpo'] = df_en['cuerpo'].apply(limpiar_texto)
df_en = df_en[df_en['cuerpo'] != ""].drop_duplicates(subset=['cuerpo'])

# Muestreo estratificado de 18,000 (8,640 Phishing / 9,360 Ham)
phish_en = df_en[df_en['etiqueta'] == 1].sample(n=8640, random_state=SEED, replace=False if len(df_en[df_en['etiqueta'] == 1]) >= 8640 else True)
ham_en = df_en[df_en['etiqueta'] == 0].sample(n=9360, random_state=SEED, replace=False if len(df_en[df_en['etiqueta'] == 0]) >= 9360 else True)

df_en_18k = pd.concat([phish_en, ham_en]).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_en_18k['idioma'] = 'en'
df_en_18k['origen'] = 'Nazario/Enron'
df_en_18k['id_correo'] = [f"EN_RAW_{i+1:05d}" for i in range(len(df_en_18k))]

print(f"  ✓ Corpus Inglés procesado: {len(df_en_18k)} registros.")

# 2. CORPUS ESPAÑOL PÚBLICO (1,800 REGISTROS EXACTOS)
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
df_es = df_es.drop_duplicates(subset=['cuerpo'])

phish_es = df_es[df_es['etiqueta'] == 1].sample(n=900, random_state=SEED)
ham_es = df_es[df_es['etiqueta'] == 0].sample(n=900, random_state=SEED)

df_es_1800 = pd.concat([phish_es, ham_es]).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_es_1800['idioma'] = 'es'
df_es_1800['origen'] = 'SpearPhishMX'
df_es_1800['id_correo'] = [f"ES_PUB_{i+1:04d}" for i in range(len(df_es_1800))]

print(f"  ✓ Corpus Español Público procesado: {len(df_es_1800)} registros.")

# 3. CORPUS SINTÉTICO FISI (200 REGISTROS EXACTOS)
ruta_fisi = "data/raw/fisi_synthetic_200.csv"
df_fisi_200 = pd.read_csv(ruta_fisi, encoding="utf-8")
df_fisi_200['origen'] = 'FISI-Synthetic'

print(f"  ✓ Corpus Sintético FISI cargado: {len(df_fisi_200)} registros.")

# 4. CONSOLIDACIÓN Y EXPORTACIÓN
cols_finales = ['id_correo', 'asunto', 'cuerpo', 'idioma', 'etiqueta', 'origen']

df_final = pd.concat([
    df_en_18k[cols_finales],
    df_es_1800[cols_finales],
    df_fisi_200[cols_finales]
]).reset_index(drop=True)

os.makedirs("data/processed", exist_ok=True)
df_final.to_csv("data/processed/corpus_unificado_20k.csv", index=False, encoding="utf-8")

print("\n==================================================================")
print(f"🎉 Corpus Unificado Guardado en: 'data/processed/corpus_unificado_20k.csv'")
print(f"📊 Total de registros procesados: {len(df_final)}")
print("==================================================================")
