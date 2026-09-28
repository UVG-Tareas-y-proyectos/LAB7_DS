"""Prepara las bases de Personas de la ENEIC para el Laboratorio 7.

Los archivos de Excel se leen uno por uno con openpyxl en modo ``read_only``.
Solo se conservan las variables solicitadas por el enunciado; luego Spark
homologa los tipos y une los trimestres por nombre de columna. Los cuatro
trimestres de 2025 y el primer trimestre de 2026 se escriben por separado en
Parquet para evitar cualquier fuga de datos hacia la evaluación final.
"""

from __future__ import annotations

import os
import sys
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from openpyxl import load_workbook
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"


@dataclass(frozen=True)
class ArchivoENEIC:
    periodo_archivo: str
    anio_archivo: int
    trimestre_calendario: int
    nombre: str
    registros_esperados: int


ARCHIVOS = (
    ArchivoENEIC("2025T1", 2025, 1, "Personas_ENEIC_T1_2025.xlsx", 51_588),
    ArchivoENEIC("2025T2", 2025, 2, "Personas-ENEIC-T2-2025.xlsx", 51_167),
    ArchivoENEIC("2025T3", 2025, 3, "Base-de-datos-Personas-ENEIC-III-2025.xlsx", 51_583),
    ArchivoENEIC("2025T4", 2025, 4, "Base-de-datos-Personas-ENEIC-IV-2025.xlsx", 49_338),
    ArchivoENEIC("2026T1", 2026, 1, "Base-de-datos-Personas-ENEIC-I-2026.xlsx", 49_843),
)


# nombre analítico -> nombre en la base oficial
COLUMNAS = {
    "num_hogar": "NUM_HOGAR",
    "num_persona": "NUM_PERSONA",
    "trimestre_original": "TRIMESTRE",
    "ocupado": "OCUPADOS",
    "categoria_ocupacional": "P05C16",
    "salario_mensual": "P05D01",
    "edad": "P02A03",
    "antiguedad_anios": "P05C07A",
    "antiguedad_meses": "P05C07B",
    "horas_semanales": "P05H01A",
    "nivel_educativo": "P03A03A",
    "dominio": "DOMINIO",
    "factor": "FACTOR",
}


ESQUEMA_LECTURA = T.StructType(
    [T.StructField(nombre, T.StringType(), True) for nombre in COLUMNAS]
    + [
        T.StructField("archivo_origen", T.StringType(), False),
        T.StructField("periodo_archivo", T.StringType(), False),
        T.StructField("anio_archivo", T.IntegerType(), False),
        T.StructField("trimestre_calendario", T.IntegerType(), False),
    ]
)


def _texto(value: object) -> str | None:
    """Normaliza una celda de Excel a texto sin convertir vacíos en ``'nan'``."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def leer_trimestre(ruta: Path, especificacion: ArchivoENEIC) -> pd.DataFrame:
    """Lee por nombre las columnas solicitadas, sin cargar 270–302 columnas en memoria."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Descargue la base oficial de Personas del INE "
            "y colóquela en data/raw/."
        )

    workbook = load_workbook(ruta, read_only=True, data_only=True)
    worksheet = workbook[workbook.sheetnames[0]]
    rows = worksheet.iter_rows(values_only=True)
    encabezado = [_texto(value) or "" for value in next(rows)]
    posiciones = {origen: encabezado.index(origen) for origen in COLUMNAS.values() if origen in encabezado}
    faltantes = sorted(set(COLUMNAS.values()) - set(posiciones))
    if faltantes:
        workbook.close()
        raise KeyError(f"{ruta.name}: faltan las columnas requeridas {faltantes}")

    datos: dict[str, list[object]] = {nombre: [] for nombre in COLUMNAS}
    for row in rows:
        for nombre, origen in COLUMNAS.items():
            posicion = posiciones[origen]
            value = row[posicion] if posicion < len(row) else None
            datos[nombre].append(_texto(value))
    workbook.close()

    frame = pd.DataFrame(datos)
    frame["archivo_origen"] = ruta.name
    frame["periodo_archivo"] = especificacion.periodo_archivo
    frame["anio_archivo"] = especificacion.anio_archivo
    frame["trimestre_calendario"] = especificacion.trimestre_calendario

    if len(frame) != especificacion.registros_esperados:
        raise ValueError(
            f"{ruta.name}: se esperaban {especificacion.registros_esperados:,} registros "
            f"y se leyeron {len(frame):,}."
        )
    return frame


def homologar_tipos(frame: DataFrame) -> DataFrame:
    """Convierte códigos y medidas a tipos consistentes antes de combinar archivos."""
    numericas = (
        "edad",
        "antiguedad_anios",
        "antiguedad_meses",
        "horas_semanales",
        "salario_mensual",
        "factor",
    )
    for columna in numericas:
        frame = frame.withColumn(columna, F.col(columna).cast(T.DoubleType()))

    categoricas = (
        "num_hogar",
        "num_persona",
        "trimestre_original",
        "ocupado",
        "categoria_ocupacional",
        "nivel_educativo",
        "dominio",
    )
    for columna in categoricas:
        frame = frame.withColumn(
            columna,
            F.when(F.trim(F.col(columna)) == "", None).otherwise(F.trim(F.col(columna))),
        )

    return frame.withColumn(
        "antiguedad",
        F.col("antiguedad_anios") + F.col("antiguedad_meses") / F.lit(12.0),
    )


def preparar_bases(
    spark: SparkSession,
    raw_dir: Path = RAW,
    output_dir: Path = OUT,
) -> pd.DataFrame:
    """Genera los Parquet de 2025 y 2026 y devuelve la auditoría de lectura."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, DataFrame] = {}
    auditoria: list[dict[str, object]] = []

    # Spark 3.5 no soporta todos los entornos Python 3.12 al crear un RDD desde
    # objetos de Python en Windows. Un CSV temporal por archivo evita ese puente:
    # openpyxl/pandas solo extraen las columnas y Spark realiza la lectura nativa.
    with TemporaryDirectory(prefix="eneic_csv_", dir=output_dir) as temp_dir:
        temp_dir = Path(temp_dir)
        for especificacion in ARCHIVOS:
            ruta = raw_dir / especificacion.nombre
            pandas_frame = leer_trimestre(ruta, especificacion)
            ruta_temporal = temp_dir / f"{especificacion.periodo_archivo}.csv"
            pandas_frame.to_csv(ruta_temporal, index=False, encoding="utf-8", na_rep="")
            spark_frame = homologar_tipos(
                spark.read.option("header", True).schema(ESQUEMA_LECTURA).csv(str(ruta_temporal))
            ).cache()
            conteo = spark_frame.count()
            auditoria.append(
                {
                    "periodo_archivo": especificacion.periodo_archivo,
                    "archivo_origen": ruta.name,
                    "registros": conteo,
                    "columnas_originales": 302 if especificacion.periodo_archivo == "2025T4" else 270,
                }
            )
            frames[especificacion.periodo_archivo] = spark_frame

        base_2025 = frames["2025T1"]
        for periodo in ("2025T2", "2025T3", "2025T4"):
            base_2025 = base_2025.unionByName(frames[periodo])

        base_2025.write.mode("overwrite").parquet(str(output_dir / "eneic_2025.parquet"))
        frames["2026T1"].write.mode("overwrite").parquet(str(output_dir / "eneic_2026.parquet"))

    for frame in frames.values():
        frame.unpersist()
    return pd.DataFrame(auditoria)


def archivos_faltantes(raw_dir: Path = RAW) -> Iterable[Path]:
    return (raw_dir / especificacion.nombre for especificacion in ARCHIVOS if not (raw_dir / especificacion.nombre).exists())


def main() -> None:
    faltantes = list(archivos_faltantes())
    if faltantes:
        nombres = "\n".join(f"- {ruta.name}" for ruta in faltantes)
        raise FileNotFoundError(f"Faltan bases oficiales en data/raw/:\n{nombres}")

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark = (
        SparkSession.builder.master("local[4]")
        .appName("laboratorio7-preparacion")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        auditoria = preparar_bases(spark)
        print(auditoria.to_string(index=False))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
