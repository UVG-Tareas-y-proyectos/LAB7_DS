"""Construye el cuaderno reproducible del Laboratorio 7.

El archivo generado se ejecuta después con Jupyter para incrustar todas las
salidas, tablas y figuras que forman parte de la entrega.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "laboratorio7.ipynb"
v4 = nbf.v4


def md(source: str):
    return v4.new_markdown_cell(source.strip())


def code(source: str):
    return v4.new_code_cell(source.strip())


cells = [
    md(
        """
# Laboratorio 7 Spark MLlib

## Perfiles de trabajadores asalariados y estimación del salario mensual con ENEIC

**Curso:** CC3066 Data Science  
**Integrantes:** Ihan Marroquín 23108 y Diego Patzán 23525  
**Fecha:** 27 de septiembre de 2026

Este cuaderno responde las ocho actividades del enunciado con PySpark 3.5. La
población analítica se construye con los cuatro trimestres de 2025. El primer
trimestre de 2026 se mantiene separado hasta la evaluación final. El análisis
es no ponderado; `FACTOR` se conserva para documentar el diseño muestral, pero
los resultados describen los registros elegibles y no estimaciones oficiales
de la población guatemalteca.
"""
    ),
    md(
        """
## Configuración y reproducibilidad

El cuaderno usa únicamente Spark para la preparación analítica, el clustering,
los modelos y las métricas. Pandas se limita a tablas agregadas y muestras de
hasta 10,000 registros para graficar. No se utiliza scikit-learn para entrenar
modelos.
"""
    ),
    code(
        """
from pathlib import Path
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.stat import Correlation

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda value: f"{value:,.3f}")
sns.set_theme(style="whitegrid")

BASE = Path.cwd().resolve()
if BASE.name == "notebooks":
    BASE = BASE.parent
sys.path.insert(0, str(BASE / "src"))
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (
    SparkSession.builder.master("local[4]")
    .appName("laboratorio7-eneic")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("Spark", spark.version)
"""
    ),
    md(
        """
## 1 Carga armonización y calidad de datos

`preparar_datos.py` lee cada Excel oficial de forma independiente, selecciona
las columnas por nombre y las convierte a un esquema común. Esto evita dos
problemas: cargar simultáneamente cinco archivos de 270–302 columnas y mezclar
variables cuando el IV trimestre cambia el orden. Los cuatro cortes de 2025 se
combinan con `unionByName`; 2026 se escribe en otro Parquet.
"""
    ),
    code(
        """
from preparar_datos import ARCHIVOS, preparar_bases

PROCESSED = BASE / "data" / "processed"
ruta_2025 = PROCESSED / "eneic_2025.parquet"
ruta_2026 = PROCESSED / "eneic_2026.parquet"

if not ruta_2025.exists() or not ruta_2026.exists():
    auditoria_lectura = preparar_bases(spark, BASE / "data" / "raw", PROCESSED)
else:
    auditoria_lectura = pd.DataFrame(
        {
            "periodo_archivo": [a.periodo_archivo for a in ARCHIVOS],
            "archivo_origen": [a.nombre for a in ARCHIVOS],
            "registros": [a.registros_esperados for a in ARCHIVOS],
            "columnas_originales": [302 if a.periodo_archivo == "2025T4" else 270 for a in ARCHIVOS],
        }
    )

display(auditoria_lectura)

df_2025 = spark.read.parquet(str(ruta_2025)).cache()
df_2026 = spark.read.parquet(str(ruta_2026)).cache()
print("Registros 2025 sin filtrar:", f"{df_2025.count():,}")
print("Registros 2026 sin filtrar:", f"{df_2026.count():,}")
df_2025.printSchema()
df_2025.select(
    "periodo_archivo", "num_hogar", "num_persona", "edad", "antiguedad",
    "horas_semanales", "nivel_educativo", "categoria_ocupacional", "dominio",
    "salario_mensual"
).show(5, truncate=False)
"""
    ),
    md(
        """
La procedencia define el período. `TRIMESTRE` se conserva como
`trimestre_original`, pero no se interpreta como trimestre calendario: T1 de
2025 usa el valor 2; T2 usa 3 salvo 175 registros con 2; T3, T4 y T1 de 2026
usan 4, 5 y 6. Alterar esos valores supondría corregir una fuente sin evidencia.
"""
    ),
    code(
        """
valores_trimestre = (
    df_2025.groupBy("periodo_archivo", "trimestre_original")
    .count()
    .orderBy("periodo_archivo", "trimestre_original")
    .toPandas()
)
display(valores_trimestre)
"""
    ),
    md(
        """
### Faltantes antes de filtrar

Se cuenta como faltante un nulo o una cadena vacía. Los porcentajes utilizan
todos los registros originales de 2025, antes de restringir la población.
"""
    ),
    code(
        """
columnas_auditadas = [
    "edad", "antiguedad_anios", "antiguedad_meses", "horas_semanales",
    "nivel_educativo", "categoria_ocupacional", "dominio", "salario_mensual",
    "ocupado", "factor"
]
total_original = df_2025.count()
faltantes_fila = df_2025.select(
    *[
        F.sum(
            F.when(
                F.col(columna).isNull() | (F.trim(F.col(columna).cast("string")) == ""),
                1,
            ).otherwise(0)
        ).alias(columna)
        for columna in columnas_auditadas
    ]
).first().asDict()
faltantes = pd.DataFrame(
    {
        "variable": columnas_auditadas,
        "faltantes": [faltantes_fila[columna] for columna in columnas_auditadas],
    }
)
faltantes["porcentaje"] = faltantes["faltantes"] / total_original * 100
display(faltantes)
"""
    ),
    md(
        """
### Filtros y auditoría de exclusiones

El orden se mantiene idéntico para 2025 y 2026. No se imputa el salario y no se
recortan valores altos o bajos. La antigüedad requiere años no negativos,
meses enteros entre 0 y 11 y un total que no supere la edad.
"""
    ),
    code(
        """
NIVELES = {
    "0": "Ninguno", "1": "Preprimaria", "2": "Primaria", "3": "Básico",
    "4": "Diversificado", "5": "Superior", "6": "Maestría", "7": "Doctorado",
}
CATEGORIAS = {
    "1": "Empleado de gobierno", "2": "Empresa privada",
    "3": "Jornalero o peón", "4": "Servicio doméstico",
}
DOMINIOS = {"1": "Urbano metropolitano", "2": "Resto urbano", "3": "Rural nacional"}


def es_finito(columna):
    valor = F.col(columna)
    return valor.isNotNull() & ~F.isnan(valor) & (F.abs(valor) != F.lit(float("inf")))


def etiquetar_categorias(frame):
    def aplicar(columna, etiquetas):
        pares = []
        for clave, etiqueta in etiquetas.items():
            pares.extend([F.lit(clave), F.lit(etiqueta)])
        mapa = F.create_map(*pares)
        codigo = F.when(F.col(columna).isin(list(etiquetas)), F.col(columna)).otherwise("DESCONOCIDO")
        return codigo, F.coalesce(mapa[codigo], F.lit("Desconocido"))

    nivel, nivel_etiqueta = aplicar("nivel_educativo", NIVELES)
    categoria, categoria_etiqueta = aplicar("categoria_ocupacional", CATEGORIAS)
    dominio, dominio_etiqueta = aplicar("dominio", DOMINIOS)
    return (
        frame.withColumn("nivel_educativo", nivel)
        .withColumn("nivel_educativo_etiqueta", nivel_etiqueta)
        .withColumn("categoria_ocupacional", categoria)
        .withColumn("categoria_ocupacional_etiqueta", categoria_etiqueta)
        .withColumn("dominio", dominio)
        .withColumn("dominio_etiqueta", dominio_etiqueta)
    )


def aplicar_filtros(frame):
    pasos = [("Base original", frame.count(), 0)]
    condiciones = [
        ("Edad finita y >= 15", es_finito("edad") & (F.col("edad") >= 15)),
        ("Persona ocupada", F.col("ocupado") == "1"),
        ("Categoría asalariada 1–4", F.col("categoria_ocupacional").isin("1", "2", "3", "4")),
        ("Salario finito y > 0", es_finito("salario_mensual") & (F.col("salario_mensual") > 0)),
        ("Años de antigüedad >= 0", es_finito("antiguedad_anios") & (F.col("antiguedad_anios") >= 0)),
        (
            "Meses de antigüedad enteros 0–11",
            es_finito("antiguedad_meses")
            & (F.col("antiguedad_meses") >= 0)
            & (F.col("antiguedad_meses") <= 11)
            & (F.col("antiguedad_meses") == F.floor("antiguedad_meses")),
        ),
        ("Antigüedad total <= edad", es_finito("antiguedad") & (F.col("antiguedad") <= F.col("edad"))),
        (
            "Horas finitas entre 1 y 168",
            es_finito("horas_semanales")
            & (F.col("horas_semanales") > 0)
            & (F.col("horas_semanales") <= 168),
        ),
    ]
    actual = frame
    anterior = pasos[0][1]
    for nombre, condicion in condiciones:
        actual = actual.filter(condicion)
        restante = actual.count()
        pasos.append((nombre, restante, anterior - restante))
        anterior = restante
    return etiquetar_categorias(actual).cache(), pd.DataFrame(
        pasos, columns=["paso", "registros_restantes", "excluidos_en_el_paso"]
    )


df_2025_filtrado, auditoria_2025 = aplicar_filtros(df_2025)
df_2026_filtrado, auditoria_2026 = aplicar_filtros(df_2026)
print("Auditoría 2025")
display(auditoria_2025)
print("Auditoría 2026")
display(auditoria_2026)
"""
    ),
    md(
        """
### Unicidad de la clave longitudinal

La combinación `periodo_archivo + num_hogar + num_persona` identifica una
observación dentro de un corte. Si una persona aparece en dos períodos, se
conservan ambas observaciones porque la ENEIC es longitudinal. No se usa
`dropDuplicates()` como sustituto de una investigación.
"""
    ),
    code(
        """
clave = ["periodo_archivo", "num_hogar", "num_persona"]
duplicados = df_2025_filtrado.groupBy(*clave).count().filter(F.col("count") > 1).cache()
n_claves_duplicadas = duplicados.count()
print("Filas elegibles:", f"{df_2025_filtrado.count():,}")
print("Claves duplicadas:", f"{n_claves_duplicadas:,}")

if n_claves_duplicadas:
    columnas_hash = [
        F.coalesce(F.col(columna).cast("string"), F.lit("<NULL>"))
        for columna in df_2025_filtrado.columns
    ]
    diagnostico = (
        df_2025_filtrado.join(duplicados.select(*clave), clave)
        .withColumn("hash_fila", F.sha2(F.concat_ws("§", *columnas_hash), 256))
        .groupBy(*clave)
        .agg(F.countDistinct("hash_fila").alias("versiones_distintas"))
    )
    diagnostico.groupBy("versiones_distintas").count().show()
    diagnostico.show(10, truncate=False)
else:
    print("La clave es única; no hay repeticiones exactas ni conflictos que investigar.")
"""
    ),
    md(
        """
**Respuestas de calidad de datos**

- 2025T4 no puede apilarse por posición porque tiene 302 columnas y las
  variables requeridas cambiaron de índice respecto de las bases de 270
  columnas. Apilar por posición mezclaría conceptos sin producir un error
  visible; `unionByName` empareja el significado.
- Un dato puede faltar porque el flujo del cuestionario no aplica a esa
  persona o porque no se registró una respuesta. El primero es ausencia
  estructural; el segundo es no respuesta. Convertir ambos a cero inventaría
  una medida o una categoría.
- La misma persona puede observarse en varios cortes debido al diseño
  longitudinal. Solo sería duplicado si se repite la clave dentro del mismo
  `periodo_archivo`.
- La base filtrada excluye menores de 15 años, personas no ocupadas,
  trabajadores no asalariados y registros que no permiten evaluar las reglas
  de calidad. Además, este ejercicio no aplica `FACTOR`; por eso el número de
  filas no representa a todos los trabajadores del país.
"""
    ),
    code(
        """
df_2025_filtrado.write.mode("overwrite").parquet(str(PROCESSED / "eneic_2025_filtrado.parquet"))
df_2026_filtrado.write.mode("overwrite").parquet(str(PROCESSED / "eneic_2026_filtrado.parquet"))
print("Bases filtradas guardadas por separado en data/processed/.")
"""
    ),
    md(
        """
## 2 Estadística descriptiva y exploración

Las estadísticas se calculan con todos los registros elegibles de 2025. Las
gráficas categóricas usan agregados; el histograma utiliza una muestra
reproducible de hasta 10,000 filas, pero las cifras reportadas no se calculan
sobre esa muestra.
"""
    ),
    code(
        """
def resumen_numerico(frame, columna):
    p25, mediana, p75, p95 = frame.approxQuantile(columna, [0.25, 0.50, 0.75, 0.95], 0.0001)
    fila = frame.agg(
        F.count(columna).alias("n"),
        F.mean(columna).alias("media"),
        F.stddev(columna).alias("desviacion_estandar"),
        F.min(columna).alias("minimo"),
        F.max(columna).alias("maximo"),
    ).first().asDict()
    fila.update({"variable": columna, "p25": p25, "mediana": mediana, "p75": p75, "p95": p95})
    return fila


variables_numericas = ["salario_mensual", "edad", "antiguedad", "horas_semanales"]
resumen = pd.DataFrame([resumen_numerico(df_2025_filtrado, columna) for columna in variables_numericas])
resumen = resumen[[
    "variable", "n", "media", "mediana", "desviacion_estandar", "minimo", "p25", "p75", "p95", "maximo"
]]
display(resumen)
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, columna, titulo in zip(
    axes,
    ["categoria_ocupacional_etiqueta", "nivel_educativo_etiqueta", "dominio_etiqueta"],
    ["Categoría ocupacional", "Nivel educativo", "Dominio"],
):
    agregado = (
        df_2025_filtrado.groupBy(columna).count().orderBy(F.desc("count")).toPandas()
    )
    sns.barplot(data=agregado, x="count", y=columna, ax=ax, color="#2f75b5")
    ax.set_title(titulo)
    ax.set_xlabel("Registros")
    ax.set_ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
n_muestra = min(10_000, df_2025_filtrado.count())
muestra_salario = (
    df_2025_filtrado.select("salario_mensual")
    .orderBy(F.rand(seed=42))
    .limit(n_muestra)
    .toPandas()
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(muestra_salario, x="salario_mensual", bins=60, ax=axes[0], color="#2f75b5")
axes[0].set_title("Salario mensual en escala original")
axes[0].set_xlabel("Quetzales")
sns.histplot(np.log10(muestra_salario["salario_mensual"]), bins=60, ax=axes[1], color="#70ad47")
axes[1].set_title("Salario mensual en escala log10")
axes[1].set_xlabel("log10 de quetzales")
plt.tight_layout()
plt.show()

fila_salario = resumen.loc[resumen["variable"] == "salario_mensual"].iloc[0]
print(
    f"La media es Q{fila_salario['media']:,.2f} y la mediana Q{fila_salario['mediana']:,.2f}; "
    f"el percentil 95 llega a Q{fila_salario['p95']:,.2f}."
)
"""
    ),
    code(
        """
def mediana_por(frame, codigo, etiqueta):
    return (
        frame.groupBy(codigo, etiqueta)
        .agg(
            F.count("*").alias("n"),
            F.expr("percentile_approx(salario_mensual, 0.5, 10000)").alias("salario_mediano"),
        )
        .orderBy(F.desc("salario_mediano"))
        .toPandas()
    )


print("Salario mediano por nivel educativo")
display(mediana_por(df_2025_filtrado, "nivel_educativo", "nivel_educativo_etiqueta"))
print("Salario mediano por categoría ocupacional")
display(mediana_por(df_2025_filtrado, "categoria_ocupacional", "categoria_ocupacional_etiqueta"))

por_trimestre = (
    df_2025_filtrado.groupBy("periodo_archivo")
    .agg(
        F.count("*").alias("n"),
        F.expr("percentile_approx(salario_mensual, 0.5, 10000)").alias("salario_mediano"),
    )
    .orderBy("periodo_archivo")
    .toPandas()
)
print("Tamaño analítico y salario mediano por trimestre")
display(por_trimestre)
"""
    ),
    md(
        """
La media salarial de Q3,421.68 supera la mediana de Q3,000 y el percentil 95
llega a Q8,000, mientras el máximo observado es Q99,000. Esa separación y el
histograma confirman una cola derecha: unos pocos salarios altos desplazan la
media. No se eliminan porque son valores positivos registrados y el enunciado
prohíbe recortarlos automáticamente.

El salario mediano aumenta de Q1,500 sin educación aprobada a Q12,000 en
doctorado, aunque este último grupo solo contiene 60 registros. Gobierno tiene
la mediana ocupacional más alta (Q5,000) y servicio doméstico la más baja
(Q1,000). Por trimestre, la muestra elegible cae de 13,492 registros en 2025T2
a 12,664 en 2025T4 y la mediana sube de Q3,000 a Q3,200 en la segunda mitad del
año. Estas diferencias son descriptivas y no se interpretan como efectos
causales. La escala logarítmica se usa solo para visualizar; el objetivo
permanece en quetzales para modelar.
"""
    ),
    md(
        """
## 3 Relaciones entre variables numéricas

La correlación de Pearson se calcula con `VectorAssembler` y
`Correlation.corr()` sobre todos los registros elegibles de 2025.
"""
    ),
    code(
        """
columnas_correlacion = ["salario_mensual", "edad", "antiguedad", "horas_semanales"]
ensamblador_correlacion = VectorAssembler(inputCols=columnas_correlacion, outputCol="features_corr")
vectores_correlacion = ensamblador_correlacion.transform(df_2025_filtrado).select("features_corr")
matriz_correlacion = Correlation.corr(vectores_correlacion, "features_corr").first()[0].toArray()
correlaciones = pd.DataFrame(
    matriz_correlacion, index=columnas_correlacion, columns=columnas_correlacion
)
display(correlaciones)

plt.figure(figsize=(6, 5))
sns.heatmap(correlaciones, annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1, square=True)
plt.title("Correlación de Pearson")
plt.tight_layout()
plt.show()

asociaciones_salario = correlaciones.loc["salario_mensual"].drop("salario_mensual").abs().sort_values(ascending=False)
print("Mayor asociación lineal absoluta con salario:", asociaciones_salario.index[0], f"({asociaciones_salario.iloc[0]:.3f})")
print("Correlación edad–antigüedad:", f"{correlaciones.loc['edad', 'antiguedad']:.3f}")
"""
    ),
    md(
        """
La antigüedad presenta la mayor asociación lineal con el salario (r = 0.180),
seguida de la edad (r = 0.147) y las horas semanales (r = 0.076). Las tres son
débiles, por lo que ninguna variable numérica explica por sí sola la variación
salarial. Edad y antigüedad muestran una relación positiva moderada (r =
0.487), coherente con la acumulación de experiencia, pero también limitada
mecánicamente porque la antigüedad no puede superar la edad. Correlación no
implica causalidad.
"""
    ),
    md(
        """
## 4 Segmentación de perfiles con KMeans

Los clústeres usan edad, antigüedad y horas semanales estandarizadas. El salario
se excluye para que los grupos representen perfiles personales y laborales sin
incorporar la variable que se predecirá después. Educación, categoría y
dominio se reservan para describir los grupos. Se prueban K = 2, 3, 4 y 5 y se
elige el mayor coeficiente de silueta.
"""
    ),
    code(
        """
variables_cluster = ["edad", "antiguedad", "horas_semanales"]
ensamblador_cluster = VectorAssembler(inputCols=variables_cluster, outputCol="features_cluster_raw")
escalador_cluster = StandardScaler(
    inputCol="features_cluster_raw", outputCol="features_cluster", withMean=True, withStd=True
)
base_cluster = ensamblador_cluster.transform(df_2025_filtrado)
modelo_escalador_cluster = escalador_cluster.fit(base_cluster)
base_cluster = modelo_escalador_cluster.transform(base_cluster).cache()

evaluador_cluster = ClusteringEvaluator(
    featuresCol="features_cluster", predictionCol="cluster", metricName="silhouette"
)
resultados_k = []
modelos_k = {}
for k in (2, 3, 4, 5):
    modelo = KMeans(featuresCol="features_cluster", predictionCol="cluster", k=k, seed=42).fit(base_cluster)
    predicciones = modelo.transform(base_cluster)
    resultados_k.append(
        {"k": k, "silueta": evaluador_cluster.evaluate(predicciones), "wssse": modelo.summary.trainingCost}
    )
    modelos_k[k] = modelo

resultados_k = pd.DataFrame(resultados_k)
display(resultados_k)
mejor_k = int(resultados_k.sort_values(["silueta", "k"], ascending=[False, True]).iloc[0]["k"])
print("K seleccionado por máxima silueta:", mejor_k)
"""
    ),
    code(
        """
df_clusters = modelos_k[mejor_k].transform(base_cluster).cache()

perfil_numerico = (
    df_clusters.groupBy("cluster")
    .agg(
        F.count("*").alias("n"),
        F.mean("edad").alias("edad_media"),
        F.mean("antiguedad").alias("antiguedad_media"),
        F.mean("horas_semanales").alias("horas_media"),
        F.expr("percentile_approx(salario_mensual, 0.5, 10000)").alias("salario_mediano"),
    )
)


def moda_por_cluster(columna, salida):
    ventana = Window.partitionBy("cluster").orderBy(F.desc("count"), F.asc(columna))
    return (
        df_clusters.groupBy("cluster", columna).count()
        .withColumn("orden", F.row_number().over(ventana))
        .filter(F.col("orden") == 1)
        .select("cluster", F.col(columna).alias(salida))
    )


perfil = (
    perfil_numerico
    .join(moda_por_cluster("nivel_educativo_etiqueta", "nivel_moda"), "cluster")
    .join(moda_por_cluster("categoria_ocupacional_etiqueta", "categoria_moda"), "cluster")
    .join(moda_por_cluster("dominio_etiqueta", "dominio_moda"), "cluster")
    .orderBy("cluster")
    .toPandas()
)

media_edad = df_2025_filtrado.agg(F.mean("edad")).first()[0]
media_antiguedad = df_2025_filtrado.agg(F.mean("antiguedad")).first()[0]
media_horas = df_2025_filtrado.agg(F.mean("horas_semanales")).first()[0]


def describir_cluster(fila):
    edad = "mayor edad" if fila.edad_media >= media_edad else "menor edad"
    antiguedad = "más antigüedad" if fila.antiguedad_media >= media_antiguedad else "menos antigüedad"
    jornada = "jornada más extensa" if fila.horas_media >= media_horas else "jornada más corta"
    return f"{edad}, {antiguedad} y {jornada}; predomina {fila.categoria_moda.lower()}"


perfil["descripcion"] = perfil.apply(describir_cluster, axis=1)
display(perfil)
"""
    ),
    md(
        """
K = 2 obtiene la mayor silueta (0.556), por encima de K = 5 (0.492), K = 4
(0.479) y K = 3 (0.470). El primer perfil contiene 38,533 registros: edad media
29.1 años, antigüedad 2.4 años, jornada 48.2 horas y salario mediano Q3,000.
Se describe como **personas jóvenes con menor permanencia y jornada más
extensa**. El segundo reúne 14,492 registros: edad media 51.3 años, antigüedad
13.9 años, jornada 41.3 horas y salario mediano Q3,500; se describe como
**personas mayores con permanencia larga y jornada más corta**. En ambos
predominan empleados de empresa privada del dominio urbano metropolitano.

Las etiquetas se asignan después del ajuste y no son categorías oficiales de
la ENEIC. El salario se reporta para contextualizar, pero no construye los
clústeres.
"""
    ),
    md(
        """
## 5 Regresión lineal con Pipeline

Los cuatro trimestres de 2025 se dividen una vez en entrenamiento (80 %) y
validación (20 %) con semilla 42. Esta misma partición se reutiliza para ambos
algoritmos. Los predictores son exactamente edad, antigüedad, horas semanales,
nivel educativo, categoría ocupacional y dominio. Todos los transformadores se
ajustan solo con entrenamiento.
"""
    ),
    code(
        """
train, validation = df_2025_filtrado.randomSplit([0.8, 0.2], seed=42)
train = train.cache()
validation = validation.cache()
print("Entrenamiento:", f"{train.count():,}")
print("Validación:", f"{validation.count():,}")

categoricas_modelo = ["nivel_educativo", "categoria_ocupacional", "dominio"]
numericas_modelo = ["edad", "antiguedad", "horas_semanales"]


def metricas_regresion(predicciones):
    return {
        metrica: RegressionEvaluator(
            labelCol="salario_mensual", predictionCol="prediction", metricName=metrica
        ).evaluate(predicciones)
        for metrica in ("mae", "rmse", "r2")
    }


media_train = train.agg(F.mean("salario_mensual")).first()[0]
baseline_validation = validation.withColumn("prediction", F.lit(media_train))
metricas_baseline_validation = metricas_regresion(baseline_validation)
print("Baseline: media de entrenamiento =", f"Q{media_train:,.2f}")
display(pd.DataFrame([{"modelo": "Media de entrenamiento", **metricas_baseline_validation}]))
"""
    ),
    code(
        """
def etapas_categoricas():
    indexadores = [
        StringIndexer(inputCol=columna, outputCol=f"{columna}_idx", handleInvalid="keep")
        for columna in categoricas_modelo
    ]
    codificador = OneHotEncoder(
        inputCols=[f"{columna}_idx" for columna in categoricas_modelo],
        outputCols=[f"{columna}_ohe" for columna in categoricas_modelo],
        handleInvalid="keep",
    )
    return indexadores, codificador


resultados_lineal = []
modelos_lineal = {}
for reg_param in (0.0, 0.1, 1.0, 10.0):
    indexadores, codificador = etapas_categoricas()
    ensamblador = VectorAssembler(
        inputCols=numericas_modelo + [f"{columna}_ohe" for columna in categoricas_modelo],
        outputCol="features_sin_escalar",
    )
    escalador = StandardScaler(
        inputCol="features_sin_escalar", outputCol="features", withMean=True, withStd=True
    )
    regresion = LinearRegression(
        featuresCol="features", labelCol="salario_mensual", predictionCol="prediction",
        regParam=reg_param, elasticNetParam=0.0, maxIter=100,
    )
    pipeline = Pipeline(stages=[*indexadores, codificador, ensamblador, escalador, regresion])
    modelo = pipeline.fit(train)
    metricas = metricas_regresion(modelo.transform(validation))
    resultados_lineal.append({"regParam": reg_param, **metricas})
    modelos_lineal[reg_param] = modelo

resultados_lineal = pd.DataFrame(resultados_lineal).sort_values("rmse")
display(resultados_lineal)
mejor_reg_param = float(resultados_lineal.iloc[0]["regParam"])
modelo_lineal = modelos_lineal[mejor_reg_param]
print("Configuración lineal seleccionada: regParam =", mejor_reg_param)

ruta_modelo_lineal = PROCESSED / "modelos" / "regresion_lineal"
modelo_lineal.write().overwrite().save(str(ruta_modelo_lineal))
print("Modelo guardado en", ruta_modelo_lineal.relative_to(BASE))
"""
    ),
    md(
        """
La regularización ridge con `regParam = 10` obtiene el menor RMSE de validación:
Q1,993.66, con MAE Q1,238.08 y R² 0.422. La diferencia frente a otras
regularizaciones es pequeña, pero todas mejoran claramente el baseline de la
media, cuyo RMSE es Q2,623.47. MAE resume el error absoluto típico; RMSE penaliza
con mayor fuerza los errores grandes; R² compara el modelo con una predicción
constante. Ninguna configuración consulta 2026.
"""
    ),
    md(
        """
## 6 Random Forest con Pipeline

Random Forest reutiliza los mismos registros de entrenamiento y validación y
el mismo tratamiento categórico. No se estandarizan predictores porque los
árboles dividen el espacio por umbrales. Se comparan dos configuraciones que
varían número de árboles y profundidad máxima.
"""
    ),
    code(
        """
configuraciones_rf = [
    {"numTrees": 60, "maxDepth": 8},
    {"numTrees": 120, "maxDepth": 12},
]
resultados_rf = []
modelos_rf = {}
for configuracion in configuraciones_rf:
    indexadores, codificador = etapas_categoricas()
    ensamblador = VectorAssembler(
        inputCols=numericas_modelo + [f"{columna}_ohe" for columna in categoricas_modelo],
        outputCol="features",
    )
    bosque = RandomForestRegressor(
        featuresCol="features", labelCol="salario_mensual", predictionCol="prediction",
        numTrees=configuracion["numTrees"], maxDepth=configuracion["maxDepth"],
        maxBins=64, featureSubsetStrategy="sqrt", seed=42,
    )
    pipeline = Pipeline(stages=[*indexadores, codificador, ensamblador, bosque])
    modelo = pipeline.fit(train)
    metricas = metricas_regresion(modelo.transform(validation))
    clave_configuracion = (configuracion["numTrees"], configuracion["maxDepth"])
    resultados_rf.append({**configuracion, **metricas})
    modelos_rf[clave_configuracion] = modelo

resultados_rf = pd.DataFrame(resultados_rf).sort_values("rmse")
display(resultados_rf)
mejor_rf = resultados_rf.iloc[0]
clave_mejor_rf = (int(mejor_rf["numTrees"]), int(mejor_rf["maxDepth"]))
modelo_rf = modelos_rf[clave_mejor_rf]
print("Configuración Random Forest seleccionada:", clave_mejor_rf)

ruta_modelo_rf = PROCESSED / "modelos" / "random_forest"
modelo_rf.write().overwrite().save(str(ruta_modelo_rf))
print("Modelo guardado en", ruta_modelo_rf.relative_to(BASE))
"""
    ),
    code(
        """
comparacion_validacion = pd.concat(
    [
        pd.DataFrame([{"modelo": "Baseline media", **metricas_baseline_validation}]),
        resultados_lineal.iloc[[0]].assign(modelo="Regresión lineal"),
        resultados_rf.iloc[[0]].assign(modelo="Random Forest"),
    ],
    ignore_index=True,
)[["modelo", "mae", "rmse", "r2"]]
display(comparacion_validacion)

mejor_validacion = comparacion_validacion.sort_values("rmse").iloc[0]
print(
    f"En validación, {mejor_validacion['modelo']} obtuvo el menor RMSE "
    f"(Q{mejor_validacion['rmse']:,.2f})."
)
"""
    ),
    md(
        """
Random Forest con 120 árboles y profundidad máxima 12 logra el menor RMSE de
validación: Q1,842.78, con MAE Q1,119.27 y R² 0.507. Su RMSE es 7.6 % menor que
el de la regresión lineal y 29.8 % menor que el baseline. La mejora es
compatible con relaciones no lineales e interacciones entre educación,
categoría, dominio y variables numéricas. La mayor profundidad también puede
ajustarse a particularidades de 2025; por eso la selección se hace con
validación y se comprueba una sola vez en 2026.
"""
    ),
    md(
        """
## 7 Entrenamiento final y evaluación en 2026

Las configuraciones quedan fijadas antes de abrir la prueba. Ambos pipelines se
aplican a exactamente los mismos registros elegibles de 2026, preparados con
las mismas reglas. `handleInvalid="keep"` evita eliminar categorías nuevas.
"""
    ),
    code(
        """
pred_lineal_2026 = modelo_lineal.transform(df_2026_filtrado).cache()
pred_rf_2026 = modelo_rf.transform(df_2026_filtrado).cache()

n_test = df_2026_filtrado.count()
assert pred_lineal_2026.count() == n_test
assert pred_rf_2026.count() == n_test

baseline_2026 = df_2026_filtrado.withColumn("prediction", F.lit(media_train))
metricas_test = pd.DataFrame(
    [
        {"modelo": "Baseline media", **metricas_regresion(baseline_2026)},
        {"modelo": "Regresión lineal", **metricas_regresion(pred_lineal_2026)},
        {"modelo": "Random Forest", **metricas_regresion(pred_rf_2026)},
    ]
).sort_values("rmse")
display(metricas_test)
print("Registros idénticos evaluados por modelo:", f"{n_test:,}")
"""
    ),
    md(
        """
En 13,258 registros elegibles de 2026, Random Forest conserva el primer lugar:
MAE Q1,113.19, RMSE Q1,980.91 y R² 0.523. La regresión lineal obtiene MAE
Q1,243.60, RMSE Q2,164.02 y R² 0.431; el baseline queda en RMSE Q2,871.18 y R²
-0.002. El RMSE del bosque aumenta 7.5 % respecto de validación, pero mantiene
la ventaja temporal sin usar 2026 para ajustar hiperparámetros.
"""
    ),
    md(
        """
## 8 Visualización y análisis de errores

El residuo se define como `salario real − salario predicho`: un valor positivo
indica subestimación y uno negativo, sobreestimación. Las figuras usan la misma
muestra reproducible de hasta 5,000 registros para ambos modelos; las tablas
por grupo utilizan todos los registros elegibles de 2026.
"""
    ),
    code(
        """
claves_test = ["periodo_archivo", "num_hogar", "num_persona"]
muestra_claves = (
    df_2026_filtrado.select(*claves_test)
    .orderBy(F.rand(seed=42))
    .limit(min(5_000, n_test))
)

columnas_muestra = claves_test + ["salario_mensual", "prediction"]
muestra_lineal = (
    pred_lineal_2026.select(*columnas_muestra)
    .join(muestra_claves, claves_test, "inner")
    .withColumnRenamed("prediction", "pred_lineal")
)
muestra_rf = (
    pred_rf_2026.select(*columnas_muestra)
    .join(muestra_claves, claves_test, "inner")
    .select(*claves_test, F.col("prediction").alias("pred_rf"))
)
muestra_predicciones = muestra_lineal.join(muestra_rf, claves_test).toPandas()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for columna, titulo, ax_real, ax_residuo in [
    ("pred_lineal", "Regresión lineal", axes[0, 0], axes[1, 0]),
    ("pred_rf", "Random Forest", axes[0, 1], axes[1, 1]),
]:
    real = muestra_predicciones["salario_mensual"]
    predicho = muestra_predicciones[columna]
    residuo = real - predicho
    limite = max(real.max(), predicho.max())
    ax_real.scatter(real, predicho, alpha=0.25, s=12)
    ax_real.plot([0, limite], [0, limite], "--", color="black", linewidth=1)
    ax_real.set_title(f"{titulo}: real frente a predicho")
    ax_real.set_xlabel("Salario real Q")
    ax_real.set_ylabel("Salario predicho Q")
    ax_residuo.scatter(predicho, residuo, alpha=0.25, s=12)
    ax_residuo.axhline(0, linestyle="--", color="black", linewidth=1)
    ax_residuo.set_title(f"{titulo}: residuos")
    ax_residuo.set_xlabel("Salario predicho Q")
    ax_residuo.set_ylabel("Real − predicho Q")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
def errores_por_grupo(predicciones, grupo):
    return (
        predicciones
        .withColumn("residuo", F.col("salario_mensual") - F.col("prediction"))
        .groupBy(grupo)
        .agg(
            F.count("*").alias("n"),
            F.mean(F.abs("residuo")).alias("mae"),
            F.mean("residuo").alias("error_medio"),
            F.sqrt(F.mean(F.pow("residuo", 2))).alias("rmse"),
        )
        .orderBy(grupo)
        .toPandas()
    )


for nombre, predicciones in [("Regresión lineal", pred_lineal_2026), ("Random Forest", pred_rf_2026)]:
    print(nombre, "— nivel educativo")
    display(errores_por_grupo(predicciones, "nivel_educativo_etiqueta"))
    print(nombre, "— dominio")
    display(errores_por_grupo(predicciones, "dominio_etiqueta"))
"""
    ),
    code(
        """
p50, p75, p90, p95 = df_2026_filtrado.approxQuantile(
    "salario_mensual", [0.50, 0.75, 0.90, 0.95], 0.0001
)
print(f"Percentiles de salario 2026: p50=Q{p50:,.0f}, p75=Q{p75:,.0f}, p90=Q{p90:,.0f}, p95=Q{p95:,.0f}")


def errores_por_tramo(predicciones):
    tramo = (
        F.when(F.col("salario_mensual") <= p50, "Hasta p50")
        .when(F.col("salario_mensual") <= p90, "p50 a p90")
        .otherwise("Sobre p90")
    )
    return errores_por_grupo(predicciones.withColumn("tramo_salario", tramo), "tramo_salario")


tramos_lineal = errores_por_tramo(pred_lineal_2026)
tramos_rf = errores_por_tramo(pred_rf_2026)
print("Regresión lineal por tramo de salario")
display(tramos_lineal)
print("Random Forest por tramo de salario")
display(tramos_rf)

mejor_test = metricas_test.iloc[0]
print(
    f"Mejor RMSE en 2026: {mejor_test['modelo']} con Q{mejor_test['rmse']:,.2f}; "
    f"MAE Q{mejor_test['mae']:,.2f} y R² {mejor_test['r2']:.3f}."
)
"""
    ),
    md(
        """
### Discusión final

Random Forest es el mejor modelo final, pero los residuos muestran regresión
hacia el centro. Para salarios hasta la mediana (Q3,200) su error medio es
-Q539.59: sobreestima en promedio. Sobre el percentil 90 (Q6,000), el error
medio cambia a +Q3,340.50 y el RMSE alcanza Q5,290.27: subestima los salarios
altos. La regresión lineal reproduce el mismo patrón con errores mayores.

Por dominio, el bosque obtiene RMSE Q1,074.13 en rural nacional y Q2,512.27 en
urbano metropolitano. Por educación, los grupos pequeños de doctorado (n = 12)
y maestría (n = 183) tienen errores mucho mayores; no deben compararse como si
tuvieran la estabilidad de primaria (n = 3,957) o diversificado (n = 4,117).
Estas diferencias son diagnósticos del modelo y del tamaño de cada grupo, no
recomendaciones salariales.

La evaluación conserva tres límites. Primero, solo incluye asalariados con
salario positivo y variables válidas. Segundo, no utiliza ponderadores, por lo
que describe registros y no población. Tercero, los seis predictores observados
no capturan ocupación detallada, industria, sexo, ubicación departamental ni
calidad del puesto. En consecuencia, incluso el mejor algoritmo tiene un techo
predictivo y sus asociaciones no son causales.
"""
    ),
    code(
        """
# Liberar recursos al terminar la ejecución completa.
for frame in (
    df_2025, df_2026, df_2025_filtrado, df_2026_filtrado, base_cluster,
    df_clusters, train, validation, pred_lineal_2026, pred_rf_2026,
):
    frame.unpersist()
spark.stop()
print("Ejecución completa finalizada.")
"""
    ),
]


notebook = v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Cuaderno generado: {OUTPUT}")
