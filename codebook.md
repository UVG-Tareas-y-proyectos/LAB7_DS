# Codebook ENEIC para el Laboratorio 7

El laboratorio utiliza exclusivamente las bases de Personas de la Encuesta
Nacional de Empleo e Ingresos Continua (ENEIC) publicadas por el Instituto
Nacional de Estadística de Guatemala. Los archivos crudos se conservan sin
modificar en `data/raw/` y no se versionan por su tamaño.

Fuente oficial: <https://www.ine.gob.gt/encuesta-nacional-de-empleo-e-ingresos/>

## Cortes utilizados

| Período | Registros originales | Columnas originales | Uso |
|---|---:|---:|---|
| 2025T1 | 51,588 | 270 | Desarrollo y entrenamiento |
| 2025T2 | 51,167 | 270 | Desarrollo y entrenamiento |
| 2025T3 | 51,583 | 270 | Desarrollo y entrenamiento |
| 2025T4 | 49,338 | 302 | Validación y entrenamiento final |
| 2026T1 | 49,843 | 270 | Prueba final |

`periodo_archivo` identifica el corte publicado y no se deriva del valor de
`TRIMESTRE`. En las bases oficiales, `TRIMESTRE` toma los valores 2, 3, 4, 5 y
6 para 2025T1, 2025T2, 2025T3, 2025T4 y 2026T1, respectivamente; además,
2025T2 contiene 175 registros con valor 2. El código conserva
`trimestre_original` para auditoría y crea `trimestre_calendario` a partir del
archivo de procedencia.

## Variables conservadas

| Variable original | Nombre analítico | Tipo | Uso |
|---|---|---|---|
| `NUM_HOGAR` | `num_hogar` | texto | Clave de auditoría |
| `NUM_PERSONA` | `num_persona` | texto | Clave de auditoría |
| `TRIMESTRE` | `trimestre_original` | texto | Auditoría de la fuente |
| `OCUPADOS` | `ocupado` | categoría | Filtro (`1` = persona ocupada) |
| `P05C16` | `categoria_ocupacional` | categoría | Filtro y predictor |
| `P05D01` | `salario_mensual` | double | Variable objetivo en quetzales |
| `P02A03` | `edad` | double | Predictor y clustering |
| `P05C07A` | `antiguedad_anios` | double | Construcción de antigüedad |
| `P05C07B` | `antiguedad_meses` | double | Construcción de antigüedad |
| `P05H01A` | `horas_semanales` | double | Predictor y clustering |
| `P03A03A` | `nivel_educativo` | categoría | Predictor |
| `DOMINIO` | `dominio` | categoría | Predictor |
| `FACTOR` | `factor` | double | Diseño muestral; no se usa como peso |

La variable `antiguedad` se calcula como
`antiguedad_anios + antiguedad_meses / 12`.

## Etiquetas de variables categóricas

### Nivel educativo

| Código | Etiqueta |
|---:|---|
| 0 | Ninguno |
| 1 | Preprimaria |
| 2 | Primaria |
| 3 | Básico |
| 4 | Diversificado |
| 5 | Superior |
| 6 | Maestría |
| 7 | Doctorado |

### Categoría ocupacional incluida

| Código | Etiqueta |
|---:|---|
| 1 | Empleado de gobierno |
| 2 | Empleado de empresa privada |
| 3 | Jornalero o peón |
| 4 | Servicio doméstico |

### Dominio

| Código | Etiqueta |
|---:|---|
| 1 | Urbano metropolitano |
| 2 | Resto urbano |
| 3 | Rural nacional |

Los valores categóricos ausentes o fuera del diccionario se representan como
`DESCONOCIDO`. No se convierten a cero.

## Población analítica

El filtro se aplica en el mismo orden a 2025 y 2026:

1. edad finita y mayor o igual a 15;
2. `ocupado = 1`;
3. `categoria_ocupacional` en 1, 2, 3 o 4;
4. salario finito y estrictamente positivo;
5. años de antigüedad no negativos;
6. meses de antigüedad enteros entre 0 y 11;
7. antigüedad calculada menor o igual a la edad;
8. horas semanales mayores que 0 y menores o iguales a 168.

El salario no se imputa ni se recorta. El análisis, el clustering y las
métricas principales son no ponderados, como requiere el enunciado; `factor`
se conserva únicamente para documentar el diseño muestral.

## Limitaciones

- La ENEIC es longitudinal y una persona puede aparecer en más de un período;
  eso no constituye un duplicado si cambia `periodo_archivo`.
- Los resultados describen los registros elegibles analizados. No son
  estimaciones oficiales de la población guatemalteca porque no se aplica el
  factor de expansión.
- Las asociaciones y predicciones no demuestran causalidad ni indican cuánto
  debería ganar una persona.
