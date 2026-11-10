import pandas as pd
import hashlib

print("=" * 80)
print("🔍 AUDITORÍA GENERAL DE DUPLICADOS EN EL CORPUS")
print("=" * 80)

# 1. Cargar el corpus unificado
df = pd.read_csv("data/processed/corpus_unificado_20k.csv")

print(f"📊 Total de registros en el corpus actual: {len(df)}")
print("-" * 80)

# Rellenar valores nulos para evitar fallos de concatenación
df['asunto'] = df['asunto'].fillna('')
df['cuerpo'] = df['cuerpo'].fillna('')

# Crear texto completo concatenado y calcular su Hash MD5
df['texto_completo'] = (df['asunto'] + " " + df['cuerpo']).str.strip().str.lower()
df['hash_md5'] = df['texto_completo'].apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())

# 2. Análisis por Origen (Fuente)
print("\n1. DUPLICADOS EXACTOS POR ORIGEN (SUB-DATASET):")
for origen_val in df['origen'].unique():
    df_sub = df[df['origen'] == origen_val]
    total = len(df_sub)
    dup_md5 = df_sub.duplicated(subset=['hash_md5']).sum()
    unicos = df_sub['hash_md5'].nunique()
    print(f"   • Origen '{origen_val}':")
    print(f"     - Total registros: {total}")
    print(f"     - Duplicados idénticos: {dup_md5}")
    print(f"     - Correos únicos: {unicos}")

# 3. Análisis de Duplicados Exactos Globales (Cruzados entre fuentes)
print("\n2. DUPLICADOS EXACTOS GLOBALES (EN TODO EL CORPUS):")
dup_globales = df.duplicated(subset=['hash_md5']).sum()
unicos_globales = df['hash_md5'].nunique()
print(f"   • Total de correos 100% IDÉNTICOS (duplicados): {dup_globales}")
print(f"   • Total de correos ÚNICOS reales: {unicos_globales}")

# 4. Auditoría Específica sobre la Fuente Sintética FISI
fisi_mask = df['origen'].str.contains('FISI', case=False, na=False)
if fisi_mask.any():
    df_fisi = df[fisi_mask]
    print(f"\n3. AUDITORÍA DETALLADA DE LA FUENTE FISI ({len(df_fisi)} correos):")
    print(f"   • Textos 100% idénticos en FISI: {df_fisi.duplicated(subset=['hash_md5']).sum()}")
    print(f"   • Textos únicos en FISI: {df_fisi['hash_md5'].nunique()}")
    print(f"   • Asuntos únicos: {df_fisi['asunto'].nunique()}")
    print(f"   • Cuerpos únicos: {df_fisi['cuerpo'].nunique()}")

print("=" * 80)
