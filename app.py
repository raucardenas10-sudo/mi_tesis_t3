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

# Estilos CSS personalizados
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

st.markdown("<div class='main-title'>🛡️ [PROTOTIPO FISI-UNMSM] | DETECTOR HÍBRIDO MULTILINGÜE DE PHISHING</div>", unsafe_allow_html=True)

# Cargar Modelo, Tokenizer, Scaler y Clasificador
@st.cache_resource
def cargar_componentes():
    MODEL_NAME = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_base = AutoModel.from_pretrained(MODEL_NAME)
    model_base.eval()
    
    # Optimización mediante Cuantización Dinámica INT8 en CPU
    model_quantized = torch.quantization.quantize_dynamic(
        model_base, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    for param in model_quantized.parameters():
        param.requires_grad = False
        
    scaler = joblib.load("models/scaler.joblib")
    clasificador = joblib.load("models/clasificador_phishing.joblib")
    
    return tokenizer, model_quantized, scaler, clasificador

tokenizer, model, scaler, clasificador = cargar_componentes()

def limpiar_campos():
    st.session_state["asunto_input"] = ""
    st.session_state["cuerpo_input"] = ""

if "asunto_input" not in st.session_state:
    st.session_state["asunto_input"] = ""
if "cuerpo_input" not in st.session_state:
    st.session_state["cuerpo_input"] = ""

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

col1, col2 = st.columns([2, 1])

with col1:
    btn_evaluar = st.button("🔍 EJECUTAR INFERENCIA DEL PIPELINE", use_container_width=True, type="primary")

with col2:
    btn_limpiar = st.button("🔄 NUEVA EVALUACIÓN", use_container_width=True, on_click=limpiar_campos)

# Procesar Inferencia Extremo a Extremo
if btn_evaluar:
    texto_completo = f"{asunto} {cuerpo}".strip()
    
    if not texto_completo:
        st.warning("⚠️ Por favor ingrese al menos el asunto o el cuerpo del correo para evaluar.")
    else:
        # MEDICIÓN COMPLETA EXTREMO A EXTREMO (Tokenización + XLM-R + Scaler + SVM)
        t_inicio = time.perf_counter()
        
        # 1. Tokenización
        inputs = tokenizer(
            [texto_completo], 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # 2. Extracción de Embedding con Token [CLS]
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        # 3. Estandarización con StandardScaler
        scaled_embedding = scaler.transform(cls_embedding)
        
        # 4. Clasificación SVM
        prediccion = int(clasificador.predict(scaled_embedding)[0])
        
        t_fin = time.perf_counter()
        latencia_extremo_a_extremo = (t_fin - t_inicio) * 1000

        if hasattr(clasificador, "predict_proba"):
            probabilidades = clasificador.predict_proba(scaled_embedding)[0]
            prob_phishing = probabilidades[1] * 100
        else:
            prob_phishing = 95.0 if prediccion == 1 else 5.0

        # RESULTADOS DIAGNÓSTICOS
        st.markdown("---")
        st.subheader("📊 RESULTADOS DIAGNÓSTICOS DE LA PRUEBA")

        if prediccion == 1:
            st.error("🚨 **DIAGNÓSTICO: PHISHING DETECTADO** (Nivel de Riesgo: ALTO)")
        else:
            st.success("✅ **DIAGNÓSTICO: CORREO LEGÍTIMO (HAM)** (Nivel de Riesgo: BAJO)")

        st.markdown(f"""
        <div class='sub-card'>
            <ul>
                <li><b>Probabilidad asignada:</b> {prob_phishing:.1f}%</li>
                <li><b>Latencia extremo a extremo (Tokenización + XLM-R + Scaler + SVM):</b> {latencia_extremo_a_extremo:.2f} ms</li>
                <li><b>Extractor semántico:</b> XLM-RoBERTa Base (Embedding Token [CLS] / max_len=128 / INT8 Quantized)</li>
                <li><b>Preprocesador de vectores:</b> StandardScaler (scaler.joblib)</li>
                <li><b>Clasificador supervisado:</b> SVM con Kernel RBF (clasificador_phishing.joblib)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        