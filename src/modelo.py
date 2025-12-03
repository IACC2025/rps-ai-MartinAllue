"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

INSTRUCCIONES PARA EL ALUMNO:
-----------------------------
Este archivo contiene la plantilla para tu modelo de IA.
Debes completar las secciones marcadas con TODO.

El objetivo es crear un modelo que prediga la PROXIMA jugada del oponente
y responda con la jugada que le gana.

FORMATO DEL CSV (minimo requerido):
-----------------------------------
Tu archivo data/partidas.csv debe tener AL MENOS estas columnas:
    - numero_ronda: Numero de la ronda (1, 2, 3...)
    - jugada_j1: Jugada del jugador 1 (piedra/papel/tijera)
    - jugada_j2: Jugada del jugador 2/oponente (piedra/papel/tijera)

Ejemplo:
    numero_ronda,jugada_j1,jugada_j2
    1,piedra,papel
    2,tijera,piedra
    3,papel,papel

Si has capturado datos adicionales (tiempo_reaccion, timestamp, etc.),
puedes usarlos para crear features extra.

EVALUACION:
- 30% Extraccion de datos (documentado en DATOS.md)
- 30% Feature Engineering
- 40% Entrenamiento y funcionamiento del modelo

FLUJO:
1. Cargar datos del CSV
2. Crear features (caracteristicas predictivas)
3. Entrenar modelo(s)
4. Evaluar y seleccionar el mejor
5. Usar el modelo para predecir y jugar
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
# warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# TODO: Importa los modelos que necesites (KNN, DecisionTree, RandomForest, etc.)ok
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================
COLUMNAS_REQUERIDAS = ["ronda", "jugada_j1", "jugada_j2", "ganador"]

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.

    TODO: Implementa esta funcion ok
    - Usa pandas para leer el CSV
    - Maneja el caso de que el archivo no exista
    - Verifica que tenga las columnas necesarias

    Args:
        ruta_csv: Ruta al archivo CSV (usa RUTA_DATOS por defecto)

    Returns:
        DataFrame con los datos de las partidas
    """

    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"{ruta_csv} no existe")

    try:
        df = pd.read_csv(ruta_csv)
    except Exception as error:
        raise ValueError(f"Error al leer el CSV: {error}") from error

    columnas_faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {columnas_faltantes}")

    return df

    # TODO: Implementa la carga de datos ok
    # Pista: usa pd.read_csv()


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.

    TODO: Implementa esta funcion
    - Convierte las jugadas de texto a numeros
    - Crea la columna 'proxima_jugada_j2' (el target a predecir)
    - Elimina filas con valores nulos

    Args:
        df: DataFrame con los datos crudos

    Returns:
        DataFrame preparado para feature engineering
    """
    # TODO: Implementa la preparacion de datos
    # Pistas:
    # - Usa map() con JUGADA_A_NUM para convertir jugadas a numeros
    # - Usa shift(-1) para crear la columna de proxima jugada
    # - Usa dropna() para eliminar filas con NaN

    df["jugada_j1_num"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["jugada_j2"].map(JUGADA_A_NUM)

    df["proxima_jugada_j2"] = df["jugada_j2_num"].shift(-1)

    df = df.dropna()

    df["jugada_j1_num"] = df["jugada_j1_num"].astype(int)
    df["jugada_j2_num"] = df["jugada_j2_num"].astype(int)
    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)

    return df

# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.

    TODO: Implementa al menos 3 tipos de features diferentes.

    Ideas de features:
    1. Frecuencia de cada jugada del oponente (j2)
    2. Ultimas N jugadas (lag features)
    3. Resultado de la ronda anterior
    4. Racha actual (victorias/derrotas consecutivas)
    5. Patron despues de ganar/perder
    6. Fase del juego (inicio/medio/final)

    Cuantas mas features relevantes crees, mejor podra predecir tu modelo.

    Args:
        df: DataFrame con datos preparados

    Returns:
        DataFrame con todas las features creadas
    """
    df = df.copy()

    # ------------------------------------------
    # TODO: Feature 1 - Frecuencia de jugadas
    # ------------------------------------------
    # Calcula que porcentaje de veces j2 juega cada opcion
    # Pista: usa expanding().mean() o rolling()

    df["freq_j2_piedra"] = (df["jugada_j2_num"] == 0).expanding().mean()
    df["freq_j2_papel"] = (df["jugada_j2_num"] == 1).expanding().mean()
    df["freq_j2_tijera"] = (df["jugada_j2_num"] == 2).expanding().mean()

    df["freq_j2_piedra_5"] = (df["jugada_j2_num"] == 0).rolling(5).mean()
    df["freq_j2_papel_5"] = (df["jugada_j2_num"] == 1).rolling(5).mean()
    df["freq_j2_tijera_5"] = (df["jugada_j2_num"] == 2).rolling(5).mean()

    # ------------------------------------------
    # TODO: Feature 2 - Lag features (jugadas anteriores)
    # ------------------------------------------
    # Crea columnas con las ultimas 1, 2, 3 jugadas
    # Pista: usa shift(1), shift(2), etc.

    df["j2_lag1"] = df["jugada_j2_num"].shift(1)
    df["j2_lag2"] = df["jugada_j2_num"].shift(2)
    df["j2_lag3"] = df["jugada_j2_num"].shift(3)

    df["j1_lag1"] = df["jugada_j1_num"].shift(1)
    df["j1_lag2"] = df["jugada_j1_num"].shift(2)

    # ------------------------------------------
    # TODO: Feature 3 - Resultado anterior
    # ------------------------------------------
    # Crea una columna con el resultado de la ronda anterior
    # Esto puede revelar patrones (ej: siempre cambia despues de perder)

    def calcular_resultado(fila):
        a, b = fila["jugada_j1_num"], fila["jugada_j2_num"]
        if a == b:
            return 0
        if (a == 0 and b == 2) or (a == 1 and b == 0) or (a == 2 and b == 1):
            return 1
        return -1
    df["resultado"] = df.apply(calcular_resultado, axis=1)
    df["resultado_anterior"] = df["resultado"].shift(1)

    # ------------------------------------------
    # TODO: Mas features (opcional pero recomendado)
    # ------------------------------------------
    # Agrega mas features que creas utiles
    # Recuerda: mas features relevantes = mejor prediccion

    # FEATURE 4 RACHA ACTUAL

    racha = []
    actual = 0
    for r in df["resultado"]:
        if r == 1:
            actual = actual + 1 if actual >= 0 else 1
        elif r == -1:
            actual = actual - 1 if actual <= 0 else -1
        else:
            actual = 0
        racha.append(actual)

    df["racha"] = racha

    # FEATURE 5 TENDENCIA RECIENTE DE OPONENTE

    df["var_lag1"] = df["jugada_j2_num"] - df["jugada_j2_num"].shift(1)
    df["var_lag2"] = df["jugada_j2_num"] - df["jugada_j2_num"].shift(2)
    df["var_lag3"] = df["jugada_j2_num"] - df["jugada_j2_num"].shift(3)

    # Hago una limpieza eliminando filas con NaN y reordeono con reset_index

    df = df.dropna()
    df = df.reset_index(drop=True)


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.

    TODO: Implementa esta funcion
    - Define que columnas usar como features (X)
    - Define la columna target (y) - debe ser 'proxima_jugada_j2'
    - Elimina filas con valores nulos

    Returns:
        (X, y) - Features y target como arrays/DataFrames
    """

    df = df.copy()

    target_col = ["proxima_jugada_j2"]

    # TODO: Selecciona las columnas de features
    # feature_cols = ['feature1', 'feature2', ...]

    feature_cols = [
        "freq_j2_piedra",
        "freq_j2_papel",
        "freq_j2_tijera",

        "freq_j2_piedra_5",
        "freq_j2_papel_5",
        "freq_j2_tijera_5",

        "j2_lag1",
        "j2_lag2",
        "j2_lag3",

        "j1_lag1",
        "j1_lag2",

        "resultado_anterior",

        "racha",

        "var_lag1",
        "var_lag2",
        "var_lag3",
    ]



    # TODO: Crea X (features) e y (target)
    # X = df[feature_cols]
    # y = df['proxima_jugada_j2']

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X = X.dropna()
    y = y.loc[X.index]

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de prediccion.

    TODO: Implementa esta funcion
    - Divide los datos en train/test
    - Entrena al menos 2 modelos diferentes
    - Evalua cada modelo y selecciona el mejor
    - Muestra metricas de evaluacion

    Args:
        X: Features
        y: Target (proxima jugada del oponente)
        test_size: Proporcion de datos para test

    Returns:
        El mejor modelo entrenado
    """
    # TODO: Divide los datos
    # X_train, X_test, y_train, y_test = train_test_split(...)

    # TODO: Entrena varios modelos
    # modelos = {
    #     'KNN': KNeighborsClassifier(n_neighbors=5),
    #     'DecisionTree': DecisionTreeClassifier(),
    #     'RandomForest': RandomForestClassifier()
    # }

    # TODO: Evalua cada modelo
    # Para cada modelo:
    #   - Entrena con fit()
    #   - Predice con predict()
    #   - Calcula accuracy con accuracy_score()
    #   - Muestra classification_report()

    # TODO: Selecciona y retorna el mejor modelo

    pass  # Elimina esta linea cuando implementes


def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado en un archivo."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 4: PREDICCION Y JUEGO
# =============================================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.

    TODO: Completa esta clase para que pueda:
    - Cargar un modelo entrenado
    - Mantener historial de la partida actual
    - Predecir la proxima jugada del oponente
    - Decidir que jugada hacer para ganar
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2)

        # TODO: Carga el modelo si existe
        # try:
        #     self.modelo = cargar_modelo(ruta_modelo)
        # except FileNotFoundError:
        #     print("Modelo no encontrado. Entrena primero.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.

        TODO: Implementa esta funcion
        - Usa el historial para calcular las mismas features que usaste en entrenamiento
        - Retorna un array con las features

        Returns:
            Array con las features para la prediccion
        """
        # TODO: Calcula las features basadas en self.historial
        # Deben ser LAS MISMAS features que usaste para entrenar

        pass  # Elimina esta linea cuando implementes

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.

        TODO: Implementa esta funcion
        - Usa obtener_features_actuales() para obtener las features
        - Usa el modelo para predecir
        - Convierte la prediccion numerica a texto

        Returns:
            Jugada predicha del oponente (piedra/papel/tijera)
        """
        if self.modelo is None:
            # Si no hay modelo, juega aleatorio
            return np.random.choice(["piedra", "papel", "tijera"])

        # TODO: Implementa la prediccion
        # features = self.obtener_features_actuales()
        # prediccion = self.modelo.predict([features])[0]
        # return NUM_A_JUGADA[prediccion]

        pass  # Elimina esta linea cuando implementes

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        # Juega lo que le gana a la prediccion
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Funcion principal para entrenar el modelo.

    Ejecuta: python src/modelo.py
    """
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    # TODO: Implementa el flujo completo:
    # 1. Cargar datos
    # 2. Preparar datos
    # 3. Crear features
    # 4. Seleccionar features
    # 5. Entrenar modelo
    # 6. Guardar modelo

    print("\n[!] Implementa las funciones marcadas con TODO")
    print("[!] Luego ejecuta este script para entrenar tu modelo")


if __name__ == "__main__":
    main()
