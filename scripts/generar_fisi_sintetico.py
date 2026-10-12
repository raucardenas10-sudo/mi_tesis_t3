import os
import pandas as pd
import random

# Fijar semilla estocástica para garantizar reproducibilidad (RNF04)
random.seed(42)

# ==============================================================================
# VOCABULARIO Y CONTEXTO EXTRAÍDO DE LA FISI-UNMSM
# ==============================================================================

entidades_fisi = [
    "Facultad de Ingeniería de Sistemas e Informática (FISI-UNMSM)",
    "Departamento Académico de Ciencias de la Computación (DACC)",
    "Escuela Profesional de Ingeniería de Software",
    "Escuela Profesional de Ingeniería de Sistemas",
    "Vicedecanatura Académica FISI",
    "Comité de Grados y Títulos FISI",
    "Unidad de Posgrado e Investigación FISI"
]

docentes_autoridades = [
    "Dra. Marlene Reyes",
    "Dirección de la Escuela Profesional de Ingeniería de Software",
    "Jefatura del DACC",
    "Comisión Organizadora FISI",
    "Coordinación de Prácticas Preprofesionales"
]

eventos_actividades = [
    "Software Innovation Day",
    "I Congreso Shark Tank FISI",
    "Feria de Proyectos de Ingeniería de Software y UX",
    "Seminario de Inteligencia Artificial y Transformación Digital",
    "Convocatoria de Proyectos de Innovación Tecnológica 2026-II"
]

proyectos_ejemplo = [
    "Púntualo (Calificación Docente)",
    "Arcast (Streaming Peruano)",
    "Filmate (Plataforma de Cines)",
    "EmprendeX (Gestión para Emprendedores)",
    "Aula-Tech (Gestión Escolar LMS)"
]

lugares_canales = [
    "Auditorio Principal de la FISI – Ciudad Universitaria",
    "Laboratorio de Redes y Software FISI",
    "Transmisión en vivo por Facebook Live",
    "Portal del Sistema Unificado de Matrícula (SUM)",
    "Mesa de Partes Virtual UNMSM"
]

hashtags_fisi = "#SoftwareInnovationDay #FISIUNMSM #UNMSM #IngenieríaDeSoftware #Innovación #UX #InteligenciaArtificial #TransformaciónDigital #OrgulloSanMarquino"

# ==============================================================================
# 1. PLANTILLAS DE SPEAR-PHISHING / ATAQUES DE ENGAÑO (100 REGISTROS - CLASE 1)
# ==============================================================================

asuntos_phishing = [
    "[URGENTE] Validación obligatoria de credenciales en el SUM - Cierre de matrícula FISI",
    "ALERTA DACC: Regularización de actas de notas y acceso a cuenta @unmsm.edu.pe",
    "[Software Innovation Day] Confirmación inmediata de registro para presentación de proyectos",
    "Vicedecanatura FISI: Verificación de pagos de tasa de matrícula y acceso al portal",
    "Aviso de suspensión de cuenta institucional @unmsm.edu.pe por inactividad",
    "[Shark Tank FISI] Descarga de certificados y aprobación de proyectos seleccionados",
    "Comité de Grados y Títulos FISI: Inconsistencia detectada en expediente de graduación",
    "Convocatoria de Becas y Laptops Institucionales UNMSM - Registro de datos urgente"
]

cuerpos_phishing_templates = [
    "Estimado estudiante de la FISI, se ha detectado una inconsistencia crítica en sus credenciales del Sistema Unificado de Matrícula (SUM). Para evitar la rectificación o anulación de sus asignaturas matriculadas con el docente responsable, ingrese de inmediato al siguiente enlace seguro para validar su identidad: http://unmsm-fisi-sum-verificación.portal-login.net/ingreso",
    "Estimado miembro de la comunidad universitaria, el {entidad} le informa que su correo institucional @unmsm.edu.pe presenta intentos de acceso no autorizados. Para evitar la suspensión definitiva de sus servicios en la plataforma virtual, confirme sus datos en nuestro portal de contingencia: http://fisi-unmsm-soporte-dacc.security-verif.org",
    "Estimado participante del evento {evento}, se requiere la validación del proyecto {proyecto} para mantener su vacante en el {lugar}. De no verificar la documentación de su equipo antes de las 23:59 hrs de hoy, su postulación será dada de baja. Acceda aquí: http://fisi-unmsm-eventos-registro.phish.com/login",
    "Estimado estudiante de la {entidad}, la Vicedecanatura exige el envío y validación de su comprobante de matrícula del presente semestre lectivo. Evite multas y la suspensión del servicio de biblioteca ingresando al formulario oficial habilitado: http://fisi-unmsm-tesoreria-pagos.com/verificar"
]

# ==============================================================================
# 2. PLANTILLAS DE COMUNICADOS LEGÍTIMOS / HAM (100 REGISTROS - CLASE 0)
# ==============================================================================

asuntos_ham = [
    "Invitación al evento: {evento} - FISI UNMSM",
    "Boletín Informativo Quincenal del {entidad}",
    "Convocatoria de participación: {evento} en el {lugar}",
    "Aviso del DACC: Horarios de atención y laboratorios académicos FISI",
    "Publicación de proyectos destacados de Ingeniería de Software y UX",
    "Recordatorio sobre el proceso de rectificación de matrícula en la FISI"
]

cuerpos_ham_templates = [
    "Estimada comunidad de la FISI-UNMSM, los invitamos a participar del {evento}, espacio donde los equipos finalistas de la carrera de Ingeniería de Software presentarán sus proyectos tecnológicos innovadores como {proyecto}. El evento contará con la presencia de la {docente} y se llevará a cabo en el {lugar}. Entrada libre para alumnos y docentes. {hashtags}",
    "Estimados estudiantes, el {entidad} les informa que las actividades académicas y talleres de laboratorio se vienen desarrollando en los horarios establecidos. Se recuerda a los alumnos verificar el rol de exámenes parciales publicado en la página web oficial institucional de la facultad. Agradecemos la gentil atención al presente.",
    "Llega el {evento}, el escenario donde las mejores ideas de innovación tecnológica se ponen a prueba. En esta edición contaremos con la participación de estudiantes de los cursos de Gestión de Configuración de Software y UX. Fecha y hora coordinadas por la {docente} en el {lugar}. {hashtags}",
    "El {entidad} reafirma su compromiso con la innovación y el emprendimiento en nuestra facultad. Se comunica a los docentes y alumnos que la atención en trámite documentario y mesa de partes virtual atiende regularmente de lunes a viernes de 08:00 a 18:00 hrs."
]

# ==============================================================================
# GENERACIÓN DE LOS 200 REGISTROS
# ==============================================================================

data = []

# Generar 100 correos de Phishing (Clase 1)
for i in range(1, 101):
    asunto = random.choice(asuntos_phishing)
    plantilla_cuerpo = random.choice(cuerpos_phishing_templates)
    cuerpo = plantilla_cuerpo.format(
        entidad=random.choice(entidades_fisi),
        evento=random.choice(eventos_actividades),
        proyecto=random.choice(proyectos_ejemplo),
        lugar=random.choice(lugares_canales)
    )
    data.append({
        "id_correo": f"ES_FISI_PHISH_{i:03d}",
        "asunto": asunto,
        "cuerpo": cuerpo,
        "idioma": "es",
        "etiqueta": 1
    })

# Generar 100 correos Legítimos / Ham (Clase 0)
for i in range(1, 101):
    plantilla_asunto = random.choice(asuntos_ham)
    asunto = plantilla_asunto.format(
        evento=random.choice(eventos_actividades),
        entidad=random.choice(entidades_fisi),
        lugar=random.choice(lugares_canales)
    )
    plantilla_cuerpo = random.choice(cuerpos_ham_templates)
    cuerpo = plantilla_cuerpo.format(
        entidad=random.choice(entidades_fisi),
        evento=random.choice(eventos_actividades),
        proyecto=random.choice(proyectos_ejemplo),
        docente=random.choice(docentes_autoridades),
        lugar=random.choice(lugares_canales),
        hashtags=hashtags_fisi
    )
    data.append({
        "id_correo": f"ES_FISI_HAM_{i:03d}",
        "asunto": asunto,
        "cuerpo": cuerpo,
        "idioma": "es",
        "etiqueta": 0
    })

# Convertir a DataFrame y mezclar aleatoriamente
df_fisi = pd.DataFrame(data)
df_fisi = df_fisi.sample(frac=1, random_state=42).reset_index(drop=True)

# Crear automáticamente la ruta de carpetas si no existe
os.makedirs("data/raw", exist_ok=True)

# Guardar en la carpeta data/raw/
df_fisi.to_csv("data/raw/fisi_synthetic_200.csv", index=False, encoding="utf-8")
print("✅ Dataset sintético FISI-UNMSM creado con éxito (200 registros en ESPAÑOL) en 'data/raw/fisi_synthetic_200.csv'.")
