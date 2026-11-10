import os
import random
import hashlib
import pandas as pd

SEED = 42
random.seed(SEED)

print("=" * 80)
print("⚙️ GENERADOR SINTÉTICO DIVERSO DE CORREOS FISI-UNMSM (GARANTÍA DE UNICIDAD)")
print("=" * 80)

# ------------------------------------------------------------------------------
# DICCIONARIOS DE DIVERSIDAD (AMPLIADOS)
# ------------------------------------------------------------------------------

SALUDOS = [
    "Estimado estudiante de la FISI,", "Estimada comunidad académica,", "Atención miembro de la UNMSM,",
    "Estimado docente de la Escuela Profesional de Ingeniería de Software,", "Estimado usuario del correo @unmsm.edu.pe,",
    "Estimado(a) participante del evento,", "A la comunidad universitaria FISI,", "Estimado egresado de la facultad,"
]

ENTIDADES = [
    "la Decanatura de la FISI", "la Dirección Académica de la Escuela de Ingeniería de Software",
    "el DACC (Departamento Académico de Computación)", "el Comité de Grados y Títulos",
    "el Laboratorio de Inteligencia Artificial y Robótica", "la Unidad de Posgrado de la FISI",
    "el Centro de Cómputo e Informática UNMSM", "la Coordinación del Registro de Matrícula y Actas"
]

ACCIONES_PHISH = [
    "ha detectado accesos no autorizados a su cuenta institucional. Ingrese inmediatamente al portal de verificación para evitar el bloqueo permanente:",
    "informa que su cupo de almacenamiento en la plataforma ha llegado al límite. Ingrese a la siguiente URL para liberar espacio:",
    "ha publicado el padrón extraordinario de actas y corrección de notas 2026-II. Valide sus datos en la siguiente plataforma externa:",
    "comunica el vencimiento inminente de sus credenciales de acceso al SUM (Sistema Único de Matrícula). Actualice su contraseña aquí:",
    "ofrece la beca extraordinaria de certificación internacional en Cloud Computing. Regístrese completando el formulario obligatorio:"
]

URLS_PHISH = [
    "http://unmsm-verificacion-portal.net/login-fisi",
    "http://sum-unmsm-actualizacion.com/credenciales",
    "http://seguridad-fisi-unmsm.org/validar-cuenta",
    "http://becas-certificacion-unmsm.cloud/registro",
    "http://portal-actas-fisi2026.info/login"
]

FIRMAS = [
    "Atentamente,\nOficina de Infraestructura Tecnológica FISI - UNMSM",
    "Saludos cordiales,\nComité Director del DACC - UNMSM",
    "Atte.\nUnidad de Soporte Técnico y Redes FISI",
    "Atentamente,\nDirección de la EP de Ingeniería de Software - UNMSM"
]

ASUNTOS_PHISH_BASE = [
    "[URGENTE] Verificación de credenciales en el SUM - Cierre de cuenta",
    "[ALERTA SEGURIDAD] Accesos sospechosos en su correo @unmsm.edu.pe",
    "[FISI UNMSM] Rectificación extraordinaria de actas y notas 2026-II",
    "[ACCESO RESTRINGIDO] Su cuota de almacenamiento institucional expiró",
    "[OPORTUNIDAD] Beca de certificación internacional para la comunidad FISI"
]

ASUNTOS_HAM_BASE = [
    "Convocatoria abierta: Proyectos de Innovación Tecnológica 2026-II UNMSM",
    "Boletín Informativo Quincenal de la Escuela Profesional de Ingeniería de Software",
    "Invitación al evento: Feria de Proyectos de Software y UX - FISI UNMSM",
    "Aviso del DACC: Horarios de atención y laboratorios académicos 2026",
    "[Software Innovation Day] Confirmación de registro para presentación de proyectos"
]

CUERPOS_HAM_BASE = [
    "Nos complace invitar a todos los estudiantes e investigadores a presentar sus propuestas en la nueva edición de la feria académica de la facultad. La recepción de borradores estará abierta hasta este viernes.",
    "Se informa a la comunidad de la FISI que el horario de atención en los laboratorios de cómputo para el desarrollo de proyectos integradores ha sido extendido. Pueden consultar la programación en el portal institucional oficial.",
    "El Comité de Grados y Títulos comunica que el cronograma oficial para la entrega de borradores de tesis y sustentaciones del presente semestre ya se encuentra disponible en la cartelera digital de la facultad.",
    "Se solicita a los delegados de curso de la Escuela de Ingeniería de Software revisar la publicación de horarios de evaluaciones finales para coordinar los espacios de laboratorio requeridos."
]

# ------------------------------------------------------------------------------
# BUCLE DE GENERACIÓN CON VERIFICADOR DE UNICIDAD (HASH MD5)
# ------------------------------------------------------------------------------

correos_generados = []
hashes_registrados = set()
asuntos_registrados = set()
cuerpos_registrados = set()

total_objetivo = 200
phish_count = 0
ham_count = 0

print("⏳ Generando 200 correos sintéticos únicos...")

intentos = 0
while len(correos_generados) < total_objetivo:
    intentos += 1
    es_phishing = 1 if len(correos_generados) % 2 == 0 else 0
    
    saludo = random.choice(SALUDOS)
    entidad = random.choice(ENTIDADES)
    firma = random.choice(FIRMAS)
    
    if es_phishing == 1:
        phish_count += 1
        asunto_base = random.choice(ASUNTOS_PHISH_BASE)
        asunto = f"{asunto_base} #{phish_count:03d}"  # Garantiza variabilidad única en el asunto
        accion = random.choice(ACCIONES_PHISH)
        url = random.choice(URLS_PHISH)
        cuerpo = f"{saludo}\n\nSe le informa que {entidad} {accion}\n{url}\n\n{firma}"
        id_correo = f"ES_FISI_PHISH_{phish_count:03d}"
    else:
        ham_count += 1
        asunto_base = random.choice(ASUNTOS_HAM_BASE)
        asunto = f"{asunto_base} - Edición #{ham_count:03d}"  # Garantiza variabilidad única en el asunto
        cuerpo_base = random.choice(CUERPOS_HAM_BASE)
        cuerpo = f"{saludo}\n\n{entidad} comunica lo siguiente:\n{cuerpo_base}\n\n{firma}"
        id_correo = f"ES_FISI_HAM_{ham_count:03d}"
        
    texto_concatenado = f"{asunto} {cuerpo}".strip().lower()
    hash_md5 = hashlib.md5(texto_concatenado.encode('utf-8')).hexdigest()
    
    # CONTROL ESTRICTO DE UNICIDAD (Evita duplicados exactos y traslapes)
    if hash_md5 not in hashes_registrados and asunto not in asuntos_registrados and cuerpo not in cuerpos_registrados:
        hashes_registrados.add(hash_md5)
        asuntos_registrados.add(asunto)
        cuerpos_registrados.add(cuerpo)
        
        correos_generados.append({
            "id_correo": id_correo,
            "asunto": asunto,
            "cuerpo": cuerpo,
            "idioma": "es",
            "etiqueta": es_phishing,
            "origen": "FISI-Synthetic"
        })

df_fisi_nuevo = pd.DataFrame(correos_generados)

os.makedirs("data/raw", exist_ok=True)
df_fisi_nuevo.to_csv("data/raw/fisi_synthetic_200.csv", index=False)

print("-" * 80)
print(f"✅ Proceso completado exitosamente tras {intentos} iteraciones.")
print(f"📊 Registros sintéticos FISI generados: {len(df_fisi_nuevo)}")
print(f"   • Correos Phishing (etiqueta 1): {(df_fisi_nuevo['etiqueta'] == 1).sum()}")
print(f"   • Correos Ham (etiqueta 0): {(df_fisi_nuevo['etiqueta'] == 0).sum()}")
print(f"   • Asuntos 100% únicos: {df_fisi_nuevo['asunto'].nunique()}")
print(f"   • Cuerpos 100% únicos: {df_fisi_nuevo['cuerpo'].nunique()}")
print("💾 Archivo actualizado guardado en 'data/raw/fisi_synthetic_200.csv'")
print("=" * 80)
