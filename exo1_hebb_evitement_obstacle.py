import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
from serial.tools import list_ports

try:
    from thymiodirect import Thymio
except ImportError:
    Thymio = None


# Periode d'echantillonnage de l'algorithme 1
SAMPLE_PERIOD_S = 0.1

# Les entrees sont normalisees dans [0, 100]
THYMIO_PROX_MAX = 4500.0

# Les sorties du reseau sont saturees dans [-100, 100]
MAX_OUTPUT_SPEED = 100.0


@dataclass
class ObstacleAvoidanceNetwork:
    """
    Reseau 5 entrees -> 2 sorties.

    x = [x1, x2, x3, x4, x5]
    y = [y_gauche, y_droite]

    Equation (4) du cours:
        y = W x
    """

    W: np.ndarray  # forme (2, 5)

    @staticmethod
    def create_tp_default() -> "ObstacleAvoidanceNetwork":
        """
        Poids heuristiques symetriques pour l'evitement d'obstacle.

        Convention:
        - x1 = capteur frontal gauche
        - x2 = capteur central
        - x3 = capteur frontal droit
        - x4 = capteur arriere gauche
        - x5 = capteur arriere droit

        - y[0] = vitesse roue gauche
        - y[1] = vitesse roue droite

        Idee:
        - obstacle a gauche  -> roue gauche plus rapide, roue droite plus lente
        - obstacle a droite  -> roue droite plus rapide, roue gauche plus lente
        - obstacle devant    -> freinage des deux roues
        """
        w = np.array(
            [
                [0.90, -1.00, -0.90, 0.55, -0.55],   # roue gauche
                [-0.90, -1.00, 0.90, -0.55, 0.55],   # roue droite
            ],
            dtype=float,
        )
        return ObstacleAvoidanceNetwork(W=w)

    @staticmethod
    def saturate(value: np.ndarray, low: float, high: float) -> np.ndarray:
        return np.clip(value, low, high)

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Calcule la sortie du reseau:
            y = W x
        puis saturation dans [-100, 100].
        """
        y = self.W @ x
        y = self.saturate(y, -MAX_OUTPUT_SPEED, MAX_OUTPUT_SPEED)
        return y


def clip_sensors_0_100(raw_x: np.ndarray) -> np.ndarray:
    """
    Sature les capteurs dans [0, 100], comme dans l'enonce.
    """
    return np.clip(raw_x, 0.0, 100.0)


def normalize_prox_horizontal(value: float) -> float:
    """
    Convertit une valeur prox.horizontal du Thymio vers [0, 100].
    """
    return float(np.clip((value / THYMIO_PROX_MAX) * 100.0, 0.0, 100.0))


def sensor_stream_example() -> list[np.ndarray]:
    """
    Quelques situations d'exemple pour tester l'algorithme sans robot reel.
    """
    return [
        np.array([80, 15, 20, 10, 15], dtype=float),  # obstacle a gauche
        np.array([20, 15, 85, 15, 10], dtype=float),  # obstacle a droite
        np.array([35, 90, 40, 20, 20], dtype=float),  # obstacle devant
        np.array([15, 10, 15, 10, 10], dtype=float),  # peu d'obstacles
        np.array([75, 20, 25, 15, 15], dtype=float),
        np.array([25, 20, 80, 15, 10], dtype=float),
        np.array([30, 88, 35, 20, 20], dtype=float),
        np.array([10, 10, 10, 10, 10], dtype=float),
    ]


def run_control_loop_simulation(network: ObstacleAvoidanceNetwork, cycles: int = 16) -> None:
    """
    Version simulation de l'algorithme 1.

    A chaque periode de 100 ms:
    1. lire x
    2. calculer y = W x
    3. appliquer y[0] et y[1]
    """
    stream = sensor_stream_example()

    print("\n--- Boucle de controle simulation (100 ms) ---")
    for t in range(cycles):
        raw_x = stream[t % len(stream)]
        x = clip_sensors_0_100(raw_x)
        y = network.infer(x)

        motor_left = float(y[0])
        motor_right = float(y[1])

        print(
            f"t={t:02d}  "
            f"x={x.astype(int).tolist()}  "
            f"y=[{motor_left:7.2f}, {motor_right:7.2f}]"
        )

        time.sleep(SAMPLE_PERIOD_S)


def clamp_motor_target(speed: float) -> int:
    """
    Le Thymio attend des entiers.
    Ici on applique directement les sorties du reseau,
    bornees dans [-100, 100] comme dans le cours.
    """
    return int(np.clip(speed, -MAX_OUTPUT_SPEED, MAX_OUTPUT_SPEED))


def find_thymio_serial_port() -> str:
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("Aucun port serie detecte.")

    thymio_ports = []
    for port in ports:
        info = f"{port.description} {port.manufacturer} {port.hwid}".lower()
        if "thymio" in info:
            thymio_ports.append(port)

    candidate_ports = thymio_ports if thymio_ports else ports
    return candidate_ports[0].device


def read_x_from_thymio(th, node_id: int) -> np.ndarray:
    """
    Mapping demande par l'enonce:
    x1 <- frontal gauche
    x2 <- central
    x3 <- frontal droit
    x4 <- arriere gauche
    x5 <- arriere droit
    """
    prox = th[node_id]["prox.horizontal"]

    x = np.array(
        [
            normalize_prox_horizontal(prox[0]),  # frontal gauche
            normalize_prox_horizontal(prox[2]),  # central
            normalize_prox_horizontal(prox[4]),  # frontal droit
            normalize_prox_horizontal(prox[5]),  # arriere gauche
            normalize_prox_horizontal(prox[6]),  # arriere droit
        ],
        dtype=float,
    )

    return clip_sensors_0_100(x)


def run_control_loop_thymio(network: ObstacleAvoidanceNetwork) -> None:
    """
    Version robot reel de l'algorithme 1.
    """
    if Thymio is None:
        raise RuntimeError("Le package 'thymiodirect' n'est pas installe.")

    serial_port = find_thymio_serial_port()
    print(f"Port detecte: {serial_port}")

    th = Thymio(
        use_tcp=False,
        serial_port=serial_port,
        refreshing_coverage={"prox.horizontal", "button.center"},
    )
    th.connect()
    node_id = th.first_node()

    print("\n--- Boucle de controle Thymio (100 ms) ---")
    print("Appuie sur le bouton central pour arreter.")

    t = 0
    try:
        while True:
            # 1) lecture des entrees
            x = read_x_from_thymio(th, node_id)

            # 2) calcul des sorties
            y = network.infer(x)

            # 3) application aux moteurs
            motor_left = clamp_motor_target(y[0])
            motor_right = clamp_motor_target(y[1])

            th[node_id]["motor.left.target"] = motor_left
            th[node_id]["motor.right.target"] = motor_right

            print(
                f"t={t:04d}  "
                f"x={x.astype(int).tolist()}  "
                f"y=[{motor_left:4d}, {motor_right:4d}]"
            )
            t += 1

            if th[node_id]["button.center"]:
                print("Arret demande par bouton central.")
                break

            time.sleep(SAMPLE_PERIOD_S)

    finally:
        th[node_id]["motor.left.target"] = 0
        th[node_id]["motor.right.target"] = 0

        try:
            if getattr(th, "thymio_proxy", None) is not None:
                th.thymio_proxy.loop.call_soon_threadsafe(th.thymio_proxy.loop.stop)
            if hasattr(th, "thread"):
                th.thread.join(timeout=1.0)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exo 1 - Algorithme de reseau pour evitement d'obstacle"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Execute la boucle sur le vrai robot Thymio.",
    )
    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=True)

    net = ObstacleAvoidanceNetwork.create_tp_default()

    print("Matrice W (2x5) :")
    print(net.W)

    if args.real:
        try:
            run_control_loop_thymio(net)
        except Exception as error:
            print(f"Erreur mode robot: {error}")
            sys.exit(1)
    else:
        run_control_loop_simulation(net, cycles=16)


if __name__ == "__main__":
    main()