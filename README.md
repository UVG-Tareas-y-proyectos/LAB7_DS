# Laboratorio 7 - Spark MLlib (ENEIC, salarios)

## Correr

Los archivos originales de la ENEIC (`.xlsx`, uno por trimestre) van en `data/raw/` — no se
versionan por tamaño. Se necesitan los cuatro trimestres de 2025 y el primer trimestre de 2026.

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python src/preparar_datos.py
./venv/bin/jupyter notebook notebooks/laboratorio7.ipynb
```

Los nombres de columna del `.xlsx` (edad, nivel educativo, antigüedad, horas habituales, etc.)
ya están confirmados contra el diccionario de datos real de la ENEIC (Personas I-2025, II-2025
y I-2026 comparten la misma posición y nombre de columna) -- ver los comentarios al inicio de
`src/preparar_datos.py`.

## Estructura

- `notebooks/laboratorio7.ipynb` — carga y armonización, calidad de datos, estadística
  descriptiva, correlaciones y segmentación con KMeans (secciones 1 a 4 del enunciado)
- `src/preparar_datos.py` — lee los `.xlsx` de cada trimestre, homologa tipos y columnas, agrega
  `periodo_archivo`, `anio_archivo`, `trimestre_calendario` y `archivo_origen`, y guarda el
  resultado en Parquet (`data/processed/`)
- `data/raw/` — archivos originales de la ENEIC y sus diccionarios de datos, no se modifican ni
  se versionan
- `data/processed/` — Parquet de 2025 (unión de los cuatro trimestres) y de 2026 (evaluación
  final), separados
