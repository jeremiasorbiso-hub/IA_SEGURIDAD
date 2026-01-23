"""
PREDICT API - API de Predicción y Captura de Feedback (FASE 2)
===============================================================
API REST para predicciones en tiempo real y captura de correcciones.
El usuario consulta la IA y el sistema aprende de sus correcciones.

Autor: Sistema IA Seguridad
Fecha: Enero 2026
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import json
import pandas as pd
from datetime import datetime
import logging
import os
from transformers import BertTokenizer, BertModel

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODELOS DE DATOS (PYDANTIC)
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


class PredictionResponse(BaseModel):
    """Response con las predicciones de todas las categorías"""
    incidente_id: Optional[str]
    predictions: Dict[str, Dict[str, float]]  # {categoria: {valor: probabilidad}}
    best_predictions: Dict[str, str]  # {categoria: mejor_valor}
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "incidente_id": "INC-2026-001234",
                "predictions": {
                    "Delito": {"Robo": 0.85, "Hurto": 0.10, "Asalto": 0.05},
                    "Lugar": {"Comercio": 0.92, "Via_Publica": 0.08}
                },
                "best_predictions": {
                    "Delito": "Robo",
                    "Lugar": "Comercio"
                },
                "timestamp": "2026-01-21 10:30:45"
            }
        }


class FeedbackRequest(BaseModel):
    """Request cuando el usuario corrige una predicción"""
    incidente_id: Optional[str]
    text: str
    predicted_categories: Dict[str, str]  # Lo que predijo la IA
    corrected_categories: Dict[str, str]  # Lo que corrigió el usuario
    
    class Config:
        json_schema_extra = {
            "example": {
                "incidente_id": "INC-2026-001234",
                "text": "Robo con violencia...",
                "predicted_categories": {"Delito": "Hurto", "Lugar": "Comercio"},
                "corrected_categories": {"Delito": "Robo", "Lugar": "Comercio"}
            }
        }


class FeedbackResponse(BaseModel):
    """Response al guardar feedback"""
    status: str
    message: str
    corrections_count: int


# ============================================================================
# MODELO MULTI-HEAD (Copia del entrenador)
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
# PREDICTOR (CARGA Y USA EL MODELO)
# ============================================================================
class IncidentPredictor:
    """
    Clase para cargar el modelo entrenado y hacer predicciones.
    """
    
    def __init__(self, model_path='model/checkpoints/model_best.pt', config_path='model/checkpoints/model_config.json', maps_path='data/processed/label_maps.json'):
        logger.info("=" * 70)
        logger.info("INICIANDO PREDICTOR DE INCIDENTES")
        logger.info("=" * 70)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Dispositivo: {self.device}")
        
        # Cargar configuración
        logger.info(f"Cargando configuración desde: {config_path}")
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
        
        logger.info("Modelo cargado correctamente")
        logger.info(f"Categorías: {self.category_names}")
        logger.info("=" * 70)
    
    def predict(self, text: str, top_k: int = 3) -> Dict:
        """
        Realiza predicción de todas las categorías para un texto.
        
        Args:
            text: Texto del incidente
            top_k: Número de predicciones a retornar por categoría
        
        Returns:
            Diccionario con predicciones y probabilidades
        """
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
        
        for i, category in enumerate(self.category_names):
            # Softmax para obtener probabilidades
            logits = outputs[i]
            probs = torch.softmax(logits, dim=1)[0]
            
            # Top-k predicciones
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            
            category_preds = {}
            for prob, idx in zip(top_probs, top_indices):
                label_text = self.inverse_label_maps[category][str(idx.item())]
                category_preds[label_text] = round(prob.item(), 4)
            
            predictions[category] = category_preds
            
            # Mejor predicción
            best_idx = torch.argmax(probs).item()
            best_predictions[category] = self.inverse_label_maps[category][str(best_idx)]
        
        return {
            'predictions': predictions,
            'best_predictions': best_predictions
        }


# ============================================================================
# GESTOR DE FEEDBACK
# ============================================================================
class FeedbackManager:
    """
    Gestiona el almacenamiento de feedback para reentrenamiento.
    """
    
    def __init__(self, feedback_file='data/processed/feedback_buffer.csv'):
        self.feedback_file = feedback_file
        
        # Crear archivo si no existe
        if not os.path.exists(self.feedback_file):
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
            # Crear CSV vacío con headers
            df_empty = pd.DataFrame(columns=['timestamp', 'incidente_id', 'Texto'])
            df_empty.to_csv(self.feedback_file, index=False, encoding='utf-8')
            logger.info(f"Archivo de feedback creado: {self.feedback_file}")
    
    def save_correction(self, feedback_data: FeedbackRequest):
        """
        Guarda una corrección en el buffer de feedback.
        """
        logger.info(f"Guardando corrección: {feedback_data.incidente_id}")
        
        # Preparar registro
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'incidente_id': feedback_data.incidente_id or 'N/A',
            'Texto': feedback_data.text
        }
        
        # Agregar categorías corregidas
        for category, value in feedback_data.corrected_categories.items():
            record[category] = value
        
        # Marcar si hubo corrección
        corrections = []
        for category in feedback_data.corrected_categories.keys():
            if category in feedback_data.predicted_categories:
                if feedback_data.predicted_categories[category] != feedback_data.corrected_categories[category]:
                    corrections.append(category)
        
        record['corrections'] = ','.join(corrections) if corrections else 'none'
        
        # Guardar en CSV
        df = pd.DataFrame([record])
        
        # Append al archivo existente
        if os.path.getsize(self.feedback_file) > 0:
            df.to_csv(self.feedback_file, mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv(self.feedback_file, index=False, encoding='utf-8')
        
        logger.info(f"Corrección guardada. Categorías modificadas: {corrections}")
        
        return len(corrections)


# ============================================================================
# API REST
# ============================================================================
app = FastAPI(
    title="IA Seguridad - API de Predicción",
    description="API para clasificación automática de incidentes de seguridad con aprendizaje continuo",
    version="1.0.0"
)

# Inicializar componentes
predictor = None
feedback_manager = None


@app.on_event("startup")
async def startup_event():
    """Inicializa el predictor al arrancar la API"""
    global predictor, feedback_manager
    
    try:
        predictor = IncidentPredictor(
            model_path='model/checkpoints/model_best.pt',
            config_path='model/checkpoints/model_config.json',
            maps_path='data/processed/label_maps.json'
        )
        feedback_manager = FeedbackManager()
        logger.info("API iniciada correctamente")
    except Exception as e:
        logger.error(f"Error al iniciar API: {e}")
        raise


@app.get("/")
async def root():
    """Endpoint de bienvenida"""
    return {
        "message": "IA Seguridad - API de Clasificación de Incidentes",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "/predict",
            "feedback": "/feedback",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica el estado del sistema"""
    model_loaded = predictor is not None
    feedback_ready = feedback_manager is not None
    
    return {
        "status": "healthy" if (model_loaded and feedback_ready) else "degraded",
        "model_loaded": model_loaded,
        "feedback_system": feedback_ready,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predice las categorías de un incidente.
    
    - **text**: Descripción del incidente (mínimo 10 caracteres)
    - **incidente_id**: ID opcional del incidente
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    if len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="El texto debe tener al menos 10 caracteres")
    
    try:
        logger.info(f"🔍 Predicción solicitada: {request.incidente_id or 'N/A'}")
        
        # Realizar predicción
        result = predictor.predict(request.text, top_k=3)
        
        response = PredictionResponse(
            incidente_id=request.incidente_id,
            predictions=result['predictions'],
            best_predictions=result['best_predictions'],
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        logger.info(f"✅ Predicción completada: {request.incidente_id or 'N/A'}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Guarda el feedback cuando el usuario corrige una predicción.
    Esta información se usa para reentrenar el modelo.
    
    - **text**: Texto del incidente
    - **predicted_categories**: Lo que predijo la IA
    - **corrected_categories**: Lo que corrigió el usuario
    """
    if not feedback_manager:
        raise HTTPException(status_code=503, detail="Sistema de feedback no disponible")
    
    try:
        corrections_count = feedback_manager.save_correction(request)
        
        return FeedbackResponse(
            status="success",
            message="Feedback guardado correctamente",
            corrections_count=corrections_count
        )
        
    except Exception as e:
        logger.error(f"❌ Error al guardar feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar feedback: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """
    Retorna estadísticas del sistema de feedback.
    """
    try:
        feedback_file = 'data/processed/feedback_buffer.csv'
        
        if not os.path.exists(feedback_file) or os.path.getsize(feedback_file) == 0:
            return {
                "feedback_count": 0,
                "message": "No hay feedback registrado aún"
            }
        
        df = pd.read_csv(feedback_file)
        
        return {
            "feedback_count": len(df),
            "corrections": df['corrections'].value_counts().to_dict() if 'corrections' in df.columns else {},
            "last_update": df['timestamp'].max() if 'timestamp' in df.columns else None
        }
        
    except Exception as e:
        logger.error(f"❌ Error al obtener estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 70)
    logger.info("🚀 INICIANDO SERVIDOR API")
    logger.info("=" * 70)
    
    # Verificar que los archivos necesarios existen
    if not os.path.exists('model/checkpoints/model_best.pt'):
        logger.error("❌ ERROR: No se encontró el modelo entrenado")
        logger.error("   Ruta esperada: model/checkpoints/model_best.pt")
        exit(1)
    
    if not os.path.exists('data/processed/label_maps.json'):
        logger.error("❌ ERROR: No se encontró el mapeo de etiquetas")
        logger.error("   Ruta esperada: data/processed/label_maps.json")
        exit(1)
    
    if not os.path.exists('model/checkpoints/model_config.json'):
        logger.error("❌ ERROR: No se encontró la configuración del modelo")
        logger.error("   Ruta esperada: model/checkpoints/model_config.json")
        exit(1)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
