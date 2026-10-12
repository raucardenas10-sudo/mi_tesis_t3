import time
import joblib
import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModel

# Configuración de la página
st.set_page_config(
    page_title="Detector Híbrido de Phishing FISI-UNMSM",
    page_icon="🛡️",
    layout="centered"
)

# Estilos CSS personalizados (Título centrado, en mayúsculas y azul institucional)
st.markdown("""
    <style>
    .main-title {
        font-size: 24px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        text-transform: uppercase;
        border-bottom: 2px solid #1E3A8A;
        padding-bottom: 10px;
        margin-bottom: 25px;
    }
    .sub-card {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal centrado, en mayúsculas y azul
st.markdown("<div class='main-title'>🛡️ [PROTOTIPO FISI-UNMSM] | DETECTOR HÍBRIDO MULTILINGÜE DE PHISHING</div>", unsafe_allow_html=True)

# Cargar Modelo y Tokenizer (Cache)
@st.cache_resource
def cargar_componentes():
    MODEL_NAME = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    clasificador = joblib.load("models/clasificador_phishing.joblib")
    return tokenizer, model, clasificador

tokenizer, model, clasificador = cargar_componentes()

# Función para limpiar los campos del formulario
def limpiar_campos():
    st.session_state["asunto_input"] = ""
    st.session_state["cuerpo_input"] = ""

# Inicializar estados si no existen
if "asunto_input" not in st.session_state:
    st.session_state["asunto_input"] = ""
if "cuerpo_input" not in st.session_state:
    st.session_state["cuerpo_input"] = ""

# Extraer embedding mediante Mean Pooling
def extraer_embedding_correo(texto):
    inputs = tokenizer([texto], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pooled = (sum_embeddings / sum_mask).numpy()
    return mean_pooled

# 1. INGRESO DE DATOS
st.subheader("✉️ INGRESO DE DATOS DEL CORREO DE PRUEBA")

asunto = st.text_input(
    "Asunto del Correo:", 
    key="asunto_input",
    placeholder="Ejemplo: [URGENTE] Actualización requerida en el Sistema Unificado de Matrícula (SUM)"
)

cuerpo = st.text_area(
    "Cuerpo del Mensaje:", 
    key="cuerpo_input",
    height=150,
    placeholder="Estimado estudiante de la FISI, se requiere que ingrese al siguiente enlace para validar sus credenciales..."
)

# Botones de Acción (Ejecutar e Iniciar Nueva Evaluación)
col1, col2 = st.columns([2, 1])

with col1:
    btn_evaluar = st.button("🔍 EJECUTAR INFERENCIA DEL PIPELINE", use_container_width=True, type="primary")

with col2:
    btn_limpiar = st.button("🔄 NUEVA EVALUACIÓN", use_container_width=True, on_click=limpiar_campos)

# Procesar Inferencia si se presiona evaluar
if btn_evaluar:
    texto_completo = f"{asunto} {cuerpo}".strip()
    
    if not texto_completo:
        st.warning("⚠️ Por favor ingrese al menos el asunto o el cuerpo del correo para evaluar.")
    else:
        start_time = time.time()
        
        vector_embedding = extraer_embedding_correo(texto_completo)
        prediccion = clasificador.predict(vector_embedding)[0]
        
        if hasattr(clasificador, "predict_proba"):
            probabilidades = clasificador.predict_proba(vector_embedding)[0]
            prob_phishing = probabilidades[1] * 100
        else:
            prob_phishing = 95.0 if prediccion == 1 else 5.0
            
        end_time = time.time()
        latencia_ms = (end_time - start_time) * 1000

        # 2. RESULTADOS DIAGNÓSTICOS
        st.markdown("---")
        st.subheader("📊 RESULTADOS DIAGNÓSTICOS DE LA PRUEBA")

        if prediccion == 1 or prob_phishing >= 50.0:
            st.error("🚨 **DIAGNÓSTICO: PHISHING DETECTADO** (Nivel de Riesgo: ALTO)")
        else:
            st.success("✅ **DIAGNÓSTICO: CORREO LEGÍTIMO (HAM)** (Nivel de Riesgo: BAJO)")

        st.markdown(f"""
        <div class='sub-card'>
            <ul>
                <li><b>Probabilidad asignada:</b> {prob_phishing:.1f}%</li>
                <li><b>Tiempo de inferencia local:</b> {latencia_ms:.2f} ms</li>
                <li><b>Extractor semántico:</b> XLM-RoBERTa Base (Vector denso d=768 / Mean Pooling)</li>
                <li><b>Clasificador supervisado:</b> Regresión Logística (joblib)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
