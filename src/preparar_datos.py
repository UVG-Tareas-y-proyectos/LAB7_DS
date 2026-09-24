"""Lee las bases de Personas de la ENEIC (.xlsx, una por trimestre), homologa columnas y tipos,
agrega las columnas de identificacion del periodo y guarda el resultado en Parquet.

Se ejecuta una vez por archivo (para controlar memoria, tal como pide el enunciado) y despues
une los trimestres de 2025 disponibles con unionByName. El primer trimestre de 2026 se procesa y
se guarda por separado, para la evaluacion final del modelado supervisado.

Nombres de columna confirmados contra el diccionario de datos real (Dicc Personas ENEIC I-2025,
II-2025 y I-2026 -- los tres comparten la misma posicion y nombre de columna):
  - edad -> P02A03 (3. Edad)
  - nivel_educativo -> P03A03A (3.a nivel de educacion mas alto que aprobo)
  - antiguedad_anios / antiguedad_meses -> P05C07A / P05C07B (anios/meses en esta empresa,
    negocio o finca -- coincide con la definicion de antiguedad del enunciado)
  - horas_semanales -> P05H01A (horas habituales a la semana, ocupacion principal; se usa esta
    y no P05H01C porque el resto de variables -- p05c16, p05d01 -- tambien son de la ocupacion
    principal)
  - dominio -> DOMINIO, categoria ocupacional -> P05C16, salario -> P05D01, ocupados -> OCUPADOS

Los archivos de Personas de 2025 T1 y T2 vienen con las 270 columnas en el mismo orden (se
verifico contra el diccionario). El enunciado advierte que 2025-IV trae otro orden -- por eso
la seleccion siempre se hace por nombre (pandas ya selecciona por nombre de columna, nunca por
posicion), y conviene volver a verificar el diccionario de esa base en particular cuando
llegue.
"""
import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data/raw"
OUT = BASE / "data/processed"

# Un archivo .xlsx por trimestre. Ajusta los nombres de archivo a como los descargaste del INE.
# NOTA: a la fecha de este avance solo se cuenta con 2025 T1 y T2 -- T3, T4 y 2026 T1 se agregan
# aqui en cuanto esten disponibles (el pipeline no cambia, unionByName ya es por nombre).
ARCHIVOS_2025 = {
    "2025T1": RAW / "Personas_ENEIC_T1_2025.xlsx",
    "2025T2": RAW / "Personas-ENEIC-T2-2025.xlsx",
    # "2025T3": RAW / "Personas-ENEIC-T3-2025.xlsx",
    # "2025T4": RAW / "Personas-ENEIC-T4-2025.xlsx",
}
ARCHIVO_2026 = RAW / "Personas-ENEIC-I-2026.xlsx"  # evaluacion final (actividad 7) -- pendiente

# Columnas que se conservan de cada archivo. La clave es el nombre estandarizado que se usa en
# el resto del laboratorio; el valor es el nombre real en el .xlsx (confirmado con el diccionario).
COLUMNAS = {
    "num_hogar": "NUM_HOGAR",
    "num_persona": "NUM_PERSONA",
    "trimestre_original": "TRIMESTRE",
    "ocupados": "OCUPADOS",
    "p05c16": "P05C16",              # categoria ocupacional (asalariados: 1,2,3,4)
    "p05d01": "P05D01",              # salario mensual sin descuentos, ocupacion principal
    "edad": "P02A03",                # 3. Edad
    "antiguedad_anios": "P05C07A",   # 7.a anios en esta empresa, negocio o finca
    "antiguedad_meses": "P05C07B",   # 7.b meses en esta empresa, negocio o finca
    "horas_semanales": "P05H01A",    # 1.a horas habituales a la semana, ocupacion principal
    "nivel_educativo": "P03A03A",    # 3.a nivel de educacion mas alto que aprobo
    "dominio": "DOMINIO",
    "factor": "FACTOR",
}


def leer_trimestre(ruta: Path, periodo_archivo: str, anio_archivo: int, trimestre_calendario: int) -> pd.DataFrame:
    """Lee un .xlsx, selecciona por NOMBRE (nunca por posicion) y agrega las columnas de periodo."""
    df = pd.read_excel(ruta, dtype=str)  # como texto: homologamos tipos despues, ya en Spark
    faltantes = [c for c in COLUMNAS.values() if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"{ruta.name}: no se encontraron las columnas {faltantes}. "
            "Revisa el diccionario de datos y ajusta COLUMNAS."
        )
    df = df[list(COLUMNAS.values())].copy()
    df.columns = list(COLUMNAS.keys())
    # pd.read_excel(dtype=str) deja las celdas vacias como NaN (float), no como texto. df.where()
    # con other=None NO reemplaza (None ahi se interpreta como "sin reemplazo", pandas deja el
    # NaN). Hay que usar mask() explicitamente para que la celda quede como None real; si no,
    # createDataFrame de Spark convierte cada NaN en el string literal "NaN" en vez de nulo, y
    # los conteos de faltantes y los filtros quedan mal.
    df = df.astype(object).mask(df.isna(), None)
    df["archivo_origen"] = ruta.name
    df["periodo_archivo"] = periodo_archivo
    df["anio_archivo"] = anio_archivo
    df["trimestre_calendario"] = trimestre_calendario
    return df


def homologar_tipos(sdf):
    """Convierte de string a los tipos numericos/enteros que se necesitan para filtrar y modelar."""
    numericas = ["edad", "antiguedad_anios", "antiguedad_meses", "horas_semanales", "p05d01", "factor"]
    for c in numericas:
        sdf = sdf.withColumn(c, F.col(c).cast(T.DoubleType()))
    sdf = sdf.withColumn("anio_archivo", F.col("anio_archivo").cast(T.IntegerType()))
    sdf = sdf.withColumn("trimestre_calendario", F.col("trimestre_calendario").cast(T.IntegerType()))
    # codigos categoricos: se conservan como string, homologados a texto sin espacios extra
    for c in ["p05c16", "nivel_educativo", "dominio", "ocupados", "num_hogar", "num_persona"]:
        sdf = sdf.withColumn(c, F.trim(F.col(c).cast(T.StringType())))
    sdf = sdf.withColumn("antiguedad", F.col("antiguedad_anios") + F.col("antiguedad_meses") / F.lit(12.0))
    return sdf


def main():
    spark = SparkSession.builder.appName("eneic_prep").getOrCreate()
    OUT.mkdir(parents=True, exist_ok=True)

    # --- 2025: procesar cada trimestre por separado y unir con unionByName ---
    trimestres_2025 = []
    for periodo, ruta in ARCHIVOS_2025.items():
        anio, trim = int(periodo[:4]), int(periodo[-1])
        pdf = leer_trimestre(ruta, periodo, anio, trim)
        sdf = spark.createDataFrame(pdf)
        sdf = homologar_tipos(sdf)
        print(periodo, "registros leidos:", sdf.count())
        trimestres_2025.append(sdf)
        sdf.unpersist()

    df_2025 = trimestres_2025[0]
    for sdf in trimestres_2025[1:]:
        df_2025 = df_2025.unionByName(sdf)

    df_2025.write.mode("overwrite").parquet(str(OUT / "eneic_2025.parquet"))
    print("2025 total (sin filtrar):", df_2025.count())

    # --- 2026 T1: para la evaluacion final, mismo procesamiento por separado ---
    if ARCHIVO_2026.exists():
        pdf_2026 = leer_trimestre(ARCHIVO_2026, "2026T1", 2026, 1)
        df_2026 = spark.createDataFrame(pdf_2026)
        df_2026 = homologar_tipos(df_2026)
        df_2026.write.mode("overwrite").parquet(str(OUT / "eneic_2026.parquet"))
        print("2026T1 total (sin filtrar):", df_2026.count())
    else:
        print(f"AVISO: no se encontro {ARCHIVO_2026.name} en data/raw/ -- se omite 2026 "
              "(pendiente para la entrega final, no bloquea el avance).")

    spark.stop()


if __name__ == "__main__":
    main()
