import pandas as pd

def arreglar_texto(texto):
    if not isinstance(texto, str):
        return texto
    try:
        # Repara la doble codificación UTF-8 -> Latin-1 común cuando Excel o Python leen mal
        return texto.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto

# 1. Cargar el dataset sintético que tiene problemas de caracteres
path_csv = "data/raw/fisi_synthetic_200.csv"
try:
    df = pd.read_csv(path_csv, encoding="utf-8")
except Exception:
    df = pd.read_csv(path_csv, encoding="latin-1")

# 2. Aplicar la limpieza de caracteres corrompidos
if 'asunto' in df.columns:
    df['asunto'] = df['asunto'].apply(arreglar_texto)
if 'cuerpo' in df.columns:
    df['cuerpo'] = df['cuerpo'].apply(arreglar_texto)

# 3. Guardar de nuevo en UTF-8 limpio para Excel (utf-8-sig)
df.to_csv(path_csv, index=False, encoding="utf-8-sig")
print("✅ Archivo 'fisi_synthetic_200.csv' corregido y guardado en UTF-8 limpio.")
