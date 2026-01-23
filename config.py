"""
CONFIG - Configuración Centralizada del Sistema
===============================================
Archivo de configuración para todos los componentes del sistema.
"""

import os
from pathlib import Path

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Archivos de datos
EXCEL_BASE = RAW_DATA_DIR / "base_original.xlsx"
FEEDBACK_BUFFER = PROCESSED_DATA_DIR / "feedback_buffer.csv"
LABEL_MAPS = PROCESSED_DATA_DIR / "label_maps.json"
DATASET_V1 = PROCESSED_DATA_DIR / "dataset_v1.jsonl"
DATASET_FULL = PROCESSED_DATA_DIR / "dataset_full.jsonl"

# Modelo
MODEL_PATH = MODELS_DIR / "model_best.pt"
BACKUP_DIR = MODELS_DIR / "backups"

# ============================================================================
# CONFIGURACIÓN DE MODELO
# ============================================================================
MODEL_CONFIG = {
    # Modelo BERT en español
    # Opciones:
    # - 'dccuchile/bert-base-spanish-wwm-uncased'
    # - 'PlanTL-GOB-ES/roberta-base-bne'
    # - 'bertin-project/bertin-roberta-base-spanish'
    'bert_model': 'dccuchile/bert-base-spanish-wwm-uncased',
    
    # Hiperparámetros
    'max_length': 512,          # Longitud máxima de texto (tokens)
    'batch_size': 8,            # Tamaño de batch (ajustar según GPU/RAM)
    'learning_rate': 2e-5,      # Tasa de aprendizaje
    'epochs': 3,                # Número de épocas de entrenamiento
    'warmup_ratio': 0.1,        # Proporción de warmup steps
    'weight_decay': 0.01,       # Regularización L2
    'dropout': 0.3,             # Dropout rate
    'gradient_clip': 1.0,       # Gradient clipping max norm
}

# ============================================================================
# CONFIGURACIÓN DE API
# ============================================================================
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': False,            # Auto-reload en desarrollo
    'log_level': 'info',
    'top_k_predictions': 3,     # Número de predicciones a retornar
}

# ============================================================================
# CONFIGURACIÓN DE LOGS
# ============================================================================
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
}

# ============================================================================
# CONFIGURACIÓN DE REENTRENAMIENTO
# ============================================================================
RETRAIN_CONFIG = {
    'backup_keep': 5,           # Número de backups a mantener
    'min_feedback_records': 0,  # Mínimo de feedback para reentrenar (0 = siempre)
}

# ============================================================================
# DETECCIÓN AUTOMÁTICA DE COLUMNAS
# ============================================================================
COLUMN_DETECTION = {
    # Posibles nombres de la columna de texto
    'text_candidates': [
        'Texto', 'Descripcion', 'Descripción', 
        'Incidente', 'Detalle', 'Narrativa',
        'Reporte', 'Observaciones'
    ],
    
    # Prefijos para columnas de categorías
    'category_prefixes': [
        'Categoria', 'Categoría', 'Cat_', 
        'Tipo_', 'Clasificacion', 'Clasificación'
    ],
    
    # Criterios para detectar columnas categóricas
    'category_detection': {
        'min_unique': 2,        # Mínimo de valores únicos
        'max_unique': 500,      # Máximo de valores únicos
        'max_ratio': 0.5,       # Máx. ratio unique/total
        'min_text_length': 10,  # Longitud mínima de texto principal
    }
}

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================
def ensure_directories():
    """Crea todos los directorios necesarios si no existen."""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                      MODELS_DIR, LOGS_DIR, BACKUP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def get_config_summary():
    """Retorna un resumen de la configuración actual."""
    return {
        'Model': MODEL_CONFIG['bert_model'],
        'Max Length': MODEL_CONFIG['max_length'],
        'Batch Size': MODEL_CONFIG['batch_size'],
        'Epochs': MODEL_CONFIG['epochs'],
        'Learning Rate': MODEL_CONFIG['learning_rate'],
        'API Port': API_CONFIG['port'],
    }


if __name__ == "__main__":
    print("=" * 70)
    print("CONFIGURACIÓN DEL SISTEMA IA SEGURIDAD")
    print("=" * 70)
    print("\nConfiguraciones cargadas:")
    for key, value in get_config_summary().items():
        print(f"  {key}: {value}")
    print("\nDirectorios:")
    print(f"  Base: {BASE_DIR}")
    print(f"  Datos: {DATA_DIR}")
    print(f"  Modelos: {MODELS_DIR}")
    print("=" * 70)
