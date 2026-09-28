# Laboratorio 7 Spark MLlib

**Curso:** CC3066 Data Science · **Semestre 02, 2026**

Análisis reproducible de los salarios de personas asalariadas observadas por
la Encuesta Nacional de Empleo e Ingresos Continua (ENEIC). El proyecto usa
PySpark 3.5 para preparar los cinco cortes, explorar la población elegible,
segmentar perfiles con KMeans y comparar regresión lineal y Random Forest. Los
cuatro trimestres de 2025 se usan para desarrollo; el primer trimestre de 2026
se reserva hasta la prueba final.

Todo el procedimiento, las salidas, las figuras y la discusión se encuentran
en un cuaderno ejecutado de principio a fin:

```text
notebooks/laboratorio7.ipynb
```

## Contenido del cuaderno

1. Carga por archivo, armonización por nombre y auditoría de calidad.
2. Estadística descriptiva y exploración de salarios y perfiles.
3. Correlaciones con `VectorAssembler` y `Correlation.corr()`.
4. Segmentación KMeans con comparación de K = 2, 3, 4 y 5.
5. Pipeline de regresión lineal con ajuste de regularización.
6. Pipeline de Random Forest con dos configuraciones.
7. Evaluación temporal sobre 2026 con MAE, RMSE y R².
8. Diagnóstico de residuos y errores por educación, dominio y tramo salarial.

El modelado supervisado utiliza exactamente seis predictores: edad,
antigüedad, horas semanales, nivel educativo, categoría ocupacional y dominio.
La regresión lineal estandariza sus predictores; Random Forest no lo hace. Los
transformadores y modelos se ajustan únicamente con el conjunto de
entrenamiento de 2025.

## Datos

Descargue las bases de Personas y sus diccionarios desde la [página oficial de
la ENEIC del INE](https://www.ine.gob.gt/encuesta-nacional-de-empleo-e-ingresos/)
y coloque estos cinco archivos en `data/raw/`:

```text
Personas_ENEIC_T1_2025.xlsx
Personas-ENEIC-T2-2025.xlsx
Base-de-datos-Personas-ENEIC-III-2025.xlsx
Base-de-datos-Personas-ENEIC-IV-2025.xlsx
Base-de-datos-Personas-ENEIC-I-2026.xlsx
```

Los Excel ocupan cerca de 230 MB en total y no se versionan. El script valida
los conteos oficiales: 51,588; 51,167; 51,583; 49,338 y 49,843 registros. El
detalle de variables, códigos, filtros y limitaciones está en
[`codebook.md`](codebook.md).

## Cómo ejecutarlo

Se requiere Python 3.10–3.12 y Java 17. En Windows, `JAVA_HOME` debe apuntar a
la instalación del JDK o JRE. Spark también requiere `HADOOP_HOME` con
`winutils.exe` y `hadoop.dll` compatibles con Hadoop 3.3 para escribir archivos
locales; en Linux y macOS no se necesita ese componente.

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe src\preparar_datos.py
venv\Scripts\jupyter.exe notebook notebooks\laboratorio7.ipynb
```

En macOS o Linux, active el entorno con `source venv/bin/activate` y use los
mismos scripts. La primera ejecución convierte cada Excel por separado y
genera dos Parquet bajo `data/processed/`. El cuaderno también ejecuta esa
preparación automáticamente si no encuentra los Parquet.

Para reconstruir el cuaderno desde su fuente y volver a incrustar salidas:

```powershell
venv\Scripts\python.exe build_notebook.py
venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=3600 notebooks\laboratorio7.ipynb
```

## Estructura

```text
notebooks/
└── laboratorio7.ipynb          # entrega completa con salidas

src/
└── preparar_datos.py           # lectura eficiente y Parquet por año

data/raw/                       # Excel y diccionarios oficiales; no se versionan
data/processed/                 # Parquet y modelos guardados; no se versionan

build_notebook.py               # genera el cuaderno reproducible
codebook.md                     # variables, códigos, filtros y limitaciones
requirements.txt                # dependencias del proyecto
```

## Lectura responsable

- El análisis y las métricas son no ponderados, como establece el enunciado.
  Describen registros elegibles y no estimaciones oficiales de Guatemala.
- La ENEIC es longitudinal: una persona puede aparecer en más de un período.
- El salario no se imputa ni se recorta. Los resultados se refieren a personas
  asalariadas con salario positivo registrado y variables válidas.
- Las asociaciones y predicciones no demuestran causalidad ni indican cuánto
  debería ganar una persona.
