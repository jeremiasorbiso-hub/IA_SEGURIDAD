"""
PREDICT API V2.5 - API de Predicción y Captura de Feedback
==========================================================
API REST para predicciones en tiempo real y captura de correcciones.
INCLUYE: Normalización avanzada de jerga policial y corrección de logs.

Autor: Sistema IA Seguridad
Fecha: Enero 2026
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import json
import pandas as pd
from datetime import datetime
import logging
import os
import re
import unicodedata
from transformers import BertTokenizer, BertModel

# ============================================================================
# 1. CONFIGURACIÓN DE LOGGING ROBUSTA (SIN EMOJIS)
# ============================================================================
# Se fuerza UTF-8 para evitar error 'charmap' en Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 2. DICCIONARIO MAESTRO DE TRADUCCIÓN POLICIAL
# ============================================================================
# Este diccionario convierte la jerga telegráfica en lenguaje que BERT entiende.
MAPEO_JERGA_POLICIAL = {
    r"\bMASC\b": "MASCULINO",
    r"\bFEM\b": "FEMENINA",
    r"\bMASCULINOS\b": "MASCULINOS",
    r"\bFEMENINAS\b": "FEMENINAS",
    r"\bREF\b": "REFIERE",
    r"\bMANIF\b": "MANIFIESTA",
    r"\bSUST\b": "SUSTRAJO",
    r"\bSUSTRAJERON\b": "SUSTRAJERON",
    r"\bSOL\b": "SOLICITA",
    r"\bMOV\b": "MOVIL",
    r"\bUNID\b": "UNIDAD",
    r"\bAUOT\b": "AUTO",       # Corrección de typo común
    r"\bAUTOP\b": "AUTOPISTA",
    r"\bDOM\b": "DOMICILIO",
    r"\bBTO\b": "BARRIO",
    r"\bCUBIERTA\b": "NEUMATICO",
    r"\bH DE F\b": "ARMA DE FUEGO",
    r"\bH\. DE F\.\b": "ARMA DE FUEGO",
    r"\bCP\b": "CUIDADORES DE VEHICULOS",
    r"\bINT\b": "INTENCION",
    r"\bVIOL\b": "VIOLENCIA",
    r"\bARM\b": "ARMADO",
    r"\bP\.A\.\b": "PROTOCOLO DE ACCION",
    r"\b911\b": "EMERGENCIAS",
    r"\bNN\b": "DESCONOCIDO",
    r"\bS\/N\b": "SIN NUMERO"
}

# ============================================================================
# 3. MODELOS DE DATOS (PYDANTIC)
# ============================================================================
class PredictionRequest(BaseModel):
    """Request para predicción de categorías"""
    text: str
    incidente_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Robo con violencia en establecimiento comercial. Sustrajeron mercancía valorada en $5000.",
                "incidente_id": "INC-2026-001234"
            }
        }

class RefinedPredictionRequest(BaseModel):
    """Request para predicción refinada de una categoría específica"""
    text: str
    category: str
    incidente_id: Optional[str] = None

class PredictionResponse(BaseModel):
    """Response con las predicciones de todas las categorías"""
    incidente_id: Optional[str]
    predictions: Dict[str, Dict[str, float]]
    best_predictions: Dict[str, str]
    confidence: Dict[str, float]
    uncertainties: Dict[str, bool]
    timestamp: str

class FeedbackRequest(BaseModel):
    """Request cuando el usuario corrige una predicción"""
    incidente_id: Optional[str]
    text: str
    predicted_categories: Dict[str, str]
    corrected_categories: Dict[str, str]

class FeedbackResponse(BaseModel):
    """Response al guardar feedback"""
    status: str
    message: str
    corrections_count: int

# ============================================================================
# 4. MODELO MULTI-HEAD (BERT)
# ============================================================================
class MultiHeadBERT(nn.Module):
    """
    Arquitectura Multi-Head: Un clasificador independiente por cada categoría.
    """
    def __init__(self, n_classes_per_head):
        super(MultiHeadBERT, self).__init__()
        self.bert = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
        self.drop = nn.Dropout(p=0.3)
        
        self.heads = nn.ModuleList()
        self.category_names = list(n_classes_per_head.keys())
        
        for cat_name, num_classes in n_classes_per_head.items():
            self.heads.append(nn.Linear(self.bert.config.hidden_size, num_classes))
            
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.drop(output.pooler_output)
        return [head(pooled_output) for head in self.heads]

# ============================================================================
# 5. PREDICTOR (EL CEREBRO MEJORADO)
# ============================================================================
class IncidentPredictor:
    """
    Clase para cargar el modelo entrenado y hacer predicciones con mejoras de calibración
    y normalización policial avanzada.
    """
    
    def __init__(self, model_path='model/checkpoints/model_best.pt', config_path='model/checkpoints/model_config.json', maps_path='data/processed/label_maps.json'):
        logger.info("=" * 70)
        logger.info("INICIANDO PREDICTOR DE INCIDENTES V2.5 (PRODUCCION)")
        logger.info("=" * 70)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Dispositivo: {self.device}")
        
        # Cargar configuración
        logger.info(f"Cargando configuracion desde: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.category_names = self.config['category_names']
        self.n_classes_per_head = self.config['n_classes_per_head']
        
        # Cargar mapeos de etiquetas
        logger.info(f"Cargando mapeos desde: {maps_path}")
        with open(maps_path, 'r', encoding='utf-8') as f:
            self.label_maps = json.load(f)
        
        # Crear mapeos inversos (índice -> texto)
        self.inverse_label_maps = {}
        for category, labels in self.label_maps.items():
            self.inverse_label_maps[category] = {str(idx): label for label, idx in labels.items()}
        
        # Cargar tokenizer
        logger.info("Cargando tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
        
        # Cargar modelo
        logger.info("Construyendo modelo Multi-Head BERT...")
        self.model = MultiHeadBERT(self.n_classes_per_head)
        
        logger.info(f"Cargando pesos desde: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # Parámetros de calibración (AJUSTADOS PARA SER MÁS AGRESIVOS)
        self.temperature = 1.0 
        self.confidence_threshold = 0.35  # Bajado de 0.45 a 0.35 para evitar "No Registra" excesivo
        self.entropy_threshold = 2.2      # Subido para tolerar más ambigüedad
        
        logger.info("Modelo cargado correctamente")
        logger.info(f"Categorias: {self.category_names}")
        logger.info("=" * 70)
    
    def normalize_text(self, text: str) -> str:
        """
        NORMALIZADOR POLICIAL AVANZADO:
        Traduce jerga, corrige typos y prepara el texto para BERT.
        """
        # 1. Pasar a Mayúsculas para procesamiento uniforme
        text = text.upper()

        # 2. Separar números pegados a letras (ej: 'con2' -> 'con 2')
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

        # 3. Aplicar traducción de jerga usando el diccionario maestro
        for patron, reemplazo in MAPEO_JERGA_POLICIAL.items():
            text = re.sub(patron, reemplazo, text)

        # 4. Limpieza estándar de acentos (NFD)
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # 5. Limpieza de caracteres especiales (dejamos letras y números)
        text = re.sub(r'[^A-Z0-9 ]', ' ', text)
        
        # 6. Eliminar exceso de espacios
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 7. Retornar en minúsculas (formato que espera BERT)
        return text.lower()
    
    def calculate_entropy(self, probs: torch.Tensor) -> float:
        """Calcula la entropía de las probabilidades (mide incertidumbre)"""
        probs = probs + 1e-10 # Evitar log(0)
        entropy = -(probs * torch.log(probs)).sum().item()
        return entropy
    
    def predict(self, text: str, top_k: int = 3) -> Dict:
        """Realiza predicción de todas las categorías."""
        
        # Normalizar texto de entrada (Paso crucial)
        text = self.normalize_text(text)
        
        # Tokenizar
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config['max_len'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predicción
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # Convertir logits a probabilidades
        predictions = {}
        best_predictions = {}
        confidence_scores = {}
        uncertainties = {}
        
        for i, category in enumerate(self.category_names):
            logits = outputs[i] / self.temperature
            probs = torch.softmax(logits, dim=1)[0]
            
            entropy = self.calculate_entropy(probs)
            
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            
            category_preds = {}
            for prob, idx in zip(top_probs, top_indices):
                label_text = self.inverse_label_maps[category][str(idx.item())]
                category_preds[label_text] = round(prob.item(), 4)
            
            predictions[category] = category_preds
            
            # Mejor predicción
            best_prob = top_probs[0].item()
            best_idx = top_indices[0].item()
            best_predictions[category] = self.inverse_label_maps[category][str(best_idx)]
            confidence_scores[category] = round(best_prob, 4)
            
            is_uncertain = best_prob < self.confidence_threshold or entropy > self.entropy_threshold
            uncertainties[category] = is_uncertain
        
        return {
            'predictions': predictions,
            'best_predictions': best_predictions,
            'confidence': confidence_scores,
            'uncertainties': uncertainties
        }
    
    def set_calibration_params(self, temperature: float = 1.0, confidence_threshold: float = 0.65):
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        logger.info(f"Parametros de calibracion actualizados: T={temperature}, Threshold={confidence_threshold}")

# ============================================================================
# 6. GESTOR DE FEEDBACK
# ============================================================================
class FeedbackManager:
    """Gestiona el almacenamiento de feedback para reentrenamiento."""
    
    def __init__(self, feedback_file='data/processed/feedback_buffer.csv'):
        self.feedback_file = feedback_file
        if not os.path.exists(self.feedback_file):
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
            df_empty = pd.DataFrame(columns=['timestamp', 'incidente_id', 'Texto'])
            df_empty.to_csv(self.feedback_file, index=False, encoding='utf-8')
            logger.info(f"Archivo de feedback creado: {self.feedback_file}")
    
    def save_correction(self, feedback_data: FeedbackRequest):
        logger.info(f"Guardando correccion: {feedback_data.incidente_id}")
        
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'incidente_id': feedback_data.incidente_id or 'N/A',
            'Texto': feedback_data.text
        }
        
        for category, value in feedback_data.corrected_categories.items():
            record[category] = value
        
        corrections = []
        for category in feedback_data.corrected_categories.keys():
            if category in feedback_data.predicted_categories:
                if feedback_data.predicted_categories[category] != feedback_data.corrected_categories[category]:
                    corrections.append(category)
        
        record['corrections'] = ','.join(corrections) if corrections else 'none'
        
        df = pd.DataFrame([record])
        
        if os.path.getsize(self.feedback_file) > 0:
            df.to_csv(self.feedback_file, mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv(self.feedback_file, index=False, encoding='utf-8')
        
        logger.info(f"Correccion guardada. Categorias modificadas: {corrections}")
        return len(corrections)

# ============================================================================
# 7. API REST - ENDPOINTS
# ============================================================================
app = FastAPI(
    title="IA Seguridad - API de Predicción",
    description="API para clasificación automática de incidentes de seguridad con aprendizaje continuo",
    version="2.5.0"
)

predictor = None
feedback_manager = None

@app.on_event("startup")
async def startup_event():
    global predictor, feedback_manager
    try:
        predictor = IncidentPredictor()
        feedback_manager = FeedbackManager()
        logger.info("API iniciada correctamente")
    except Exception as e:
        logger.error(f"Error al iniciar API: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "IA Seguridad V2.5 - Operacional", "status": "active"}

@app.get("/health")
async def health_check():
    model_loaded = predictor is not None
    feedback_ready = feedback_manager is not None
    return {
        "status": "healthy" if (model_loaded and feedback_ready) else "degraded",
        "model_loaded": model_loaded,
        "feedback_system": feedback_ready
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not predictor: raise HTTPException(status_code=503, detail="Modelo no cargado")
    if len(request.text.strip()) < 5: raise HTTPException(status_code=400, detail="Texto muy corto")
    
    try:
        # LOG SIN EMOJIS
        logger.info(f"PREDICCION SOLICITADA: {request.incidente_id or 'N/A'}")
        
        result = predictor.predict(request.text, top_k=3)
        
        response = PredictionResponse(
            incidente_id=request.incidente_id,
            predictions=result['predictions'],
            best_predictions=result['best_predictions'],
            confidence=result['confidence'],
            uncertainties=result['uncertainties'],
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        logger.info(f"PREDICCION COMPLETADA: {request.incidente_id or 'N/A'}")
        return response
        
    except Exception as e:
        logger.error(f"Error en prediccion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-refined")
async def predict_refined(request: RefinedPredictionRequest):
    if not predictor: raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        logger.info(f"Prediccion refinada para {request.category}")
        result = predictor.predict(request.text, top_k=5)
        
        if request.category not in result['predictions']:
            raise HTTPException(status_code=400, detail="Categoria no valida")
        
        return {
            "incidente_id": request.incidente_id,
            "category": request.category,
            "predictions": result['predictions'][request.category],
            "best_prediction": result['best_predictions'][request.category],
            "confidence": result['confidence'][request.category],
            "is_uncertain": result['uncertainties'][request.category]
        }
    except Exception as e:
        logger.error(f"Error refinado: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    if not feedback_manager: raise HTTPException(status_code=503, detail="Feedback no disponible")
    try:
        count = feedback_manager.save_correction(request)
        return {"status": "success", "message": "Feedback guardado", "corrections_count": count}
    except Exception as e:
        logger.error(f"Error feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    try:
        file_path = 'data/processed/feedback_buffer.csv'
        if not os.path.exists(file_path): return {"feedback_count": 0}
        df = pd.read_csv(file_path)
        return {"feedback_count": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/calibration")
async def update_calibration(temperature: float = 1.0, confidence_threshold: float = 0.65):
    if not predictor: raise HTTPException(status_code=503, detail="Modelo no cargado")
    predictor.set_calibration_params(temperature, confidence_threshold)
    return {"status": "updated", "message": "Parametros actualizados"}

# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 70)
    logger.info("INICIANDO SERVIDOR API (V2.5)")
    logger.info("=" * 70)
    
    # Validaciones de archivos
    required_files = [
        'model/checkpoints/model_best.pt',
        'data/processed/label_maps.json',
        'model/checkpoints/model_config.json'
    ]
    
    for f_path in required_files:
        if not os.path.exists(f_path):
            logger.error(f"ERROR: Falta archivo critico: {f_path}")
            exit(1)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")