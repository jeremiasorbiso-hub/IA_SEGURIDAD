# Archivo de ejemplo del Excel esperado
# Coloca tu archivo base_original.xlsx en este directorio

# El Excel debe contener:
# 1. Una columna con el texto del incidente (ej: "Texto", "Descripcion", "Incidente")
# 2. Columnas de categorías que deseas predecir (ej: "Delito", "Lugar", "Hora", etc.)

# Ejemplo de estructura:
# -------------------------------------------------
# | Texto                        | Delito  | Lugar    | Hora       |
# |------------------------------|---------|----------|------------|
# | Robo con violencia en...     | Robo    | Comercio | Nocturna   |
# | Hurto de vehículo en...      | Hurto   | Parking  | Vespertina |
# | Asalto en la vía pública...  | Asalto  | Calle    | Madrugada  |
# -------------------------------------------------

# El sistema detectará automáticamente:
# ✓ Qué columna contiene el texto
# ✓ Qué columnas son categorías
# ✓ Valores únicos de cada categoría
