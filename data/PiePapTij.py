import csv
import time
import keyboard
import os

# =========================================================
# CONFIGURACIÓN
# =========================================================

# Mapeo de teclas a jugadas
tecla_a_jugada_j1 = {
    'a': 'piedra',
    's': 'papel',
    'd': 'tijera'
}

tecla_a_jugada_j2 = {
    '1': 'piedra',
    '2': 'papel',
    '3': 'tijera'
}

# Configuración del archivo de datos
NOMBRE_ARCHIVO = "partidas.csv"
ENCABEZADO_DATOS = ["numero_ronda", "jugada_j1", "jugada_j2"]


# =========================================================
# FUNCIONES DE LÓGICA
# =========================================================

def esperar_jugadas():
    """
    Espera hasta que ambos jugadores pulsen una tecla válida.
    """
    jugada_j1 = None
    jugada_j2 = None

    print("\n--- NUEVA RONDA ---")
    print("Jugadores preparados... (Jugador1: a/s/d — Jugador2: 1/2/3)")

    while jugada_j1 is None or jugada_j2 is None:
        evento = keyboard.read_event()
        if evento.event_type == keyboard.KEY_DOWN:
            if evento.name in tecla_a_jugada_j1 and jugada_j1 is None:
                jugada_j1 = tecla_a_jugada_j1[evento.name]
            elif evento.name in tecla_a_jugada_j2 and jugada_j2 is None:
                jugada_j2 = tecla_a_jugada_j2[evento.name]

    return jugada_j1, jugada_j2


def obtener_ultima_ronda(nombre_archivo):
    """
    Lee el archivo CSV y devuelve el número de la última ronda guardada,
    o 0 si el archivo no existe o está vacío.
    """
    if not os.path.exists(nombre_archivo):
        return 0

    try:
        with open(nombre_archivo, 'r', newline='') as f:
            lector = csv.reader(f)
            try:
                next(lector)  # Saltar el encabezado
            except StopIteration:
                return 0

            ultima_fila = None
            for fila in lector:
                ultima_fila = fila

            if ultima_fila and ultima_fila[0].isdigit():
                return int(ultima_fila[0])
            return 0
    except Exception as e:
        print(f"⚠️ Error al leer el número de ronda en {nombre_archivo}: {e}. Reiniciando contador.")
        return 0


def guardar_datos(nombre_archivo, fila):
    """
    Guarda la fila de datos en el CSV, creando el archivo con el encabezado si no existe.
    """
    # Si el archivo NO existe O si existe pero está vacío (st_size == 0)
    if not os.path.exists(nombre_archivo) or os.stat(nombre_archivo).st_size == 0:
        try:
            # Usamos 'w' para crear y escribir el encabezado
            with open(nombre_archivo, 'w', newline='') as f:
                escritor = csv.writer(f)
                escritor.writerow(ENCABEZADO_DATOS)
        except Exception as e:
            print(f"❌ ERROR CRÍTICO: No se pudo crear/escribir el encabezado. Verifique permisos. Error: {e}")
            return

    # Escribe la nueva fila al final del archivo
    try:
        with open(nombre_archivo, 'a', newline='') as f:
            escritor = csv.writer(f)
            escritor.writerow(fila)
    except Exception as e:
        print(f"❌ ERROR: No se pudo añadir datos a {nombre_archivo}. Error: {e}")


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def jugar_y_recopilar(nombre_archivo=NOMBRE_ARCHIVO):
    """
    Función principal para jugar y recopilar datos.
    """
    # La ruta de guardado es el NOMBRE_ARCHIVO, que se resuelve en la carpeta actual.
    csv_path = os.path.join(os.path.dirname(__file__), nombre_archivo)

    print("=========================================================")
    print("Sistema de Recopilación: Piedra, Papel o Tijera")
    print(f"Guardando datos en la carpeta actual: {csv_path}")
    print("=========================================================")
    print("Jugador1: a,s,d — Jugador2: 1,2,3. Pulsa ESC para salir.")

    # Obtenemos la última ronda del archivo *en la misma carpeta*
    ronda_actual = obtener_ultima_ronda(csv_path) + 1
    print(f"▶️ Iniciando recopilación desde Ronda {ronda_actual}")

    while True:
        if keyboard.is_pressed('esc'):
            print("\nSaliendo... ¡Datos recopilados con éxito!")
            break

        jugada_j1, jugada_j2 = esperar_jugadas()

        # Usamos csv_path para que sepa exactamente dónde guardar
        fila_datos = [ronda_actual, jugada_j1, jugada_j2]
        guardar_datos(csv_path, fila_datos)

        print(f"Ronda {ronda_actual}: J1 -> {jugada_j1} | J2 -> {jugada_j2} (Guardado)")

        ronda_actual += 1
        time.sleep(0.1)


if __name__ == "__main__":
    jugar_y_recopilar()