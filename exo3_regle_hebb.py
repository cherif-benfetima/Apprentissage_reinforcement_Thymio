import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from serial.tools import list_ports

try:
    from thymiodirect import Thymio
except ImportError:
    Thymio = None


# ------------------------------------------------------------
# Constantes TP
# ------------------------------------------------------------

SAMPLE_PERIOD_S = 0.1
THYMIO_PROX_MAX = 4500.0
MAX_OUTPUT_SPEED = 100.0

# Exo 3 : vitesse d'apprentissage Hebb
ALPHA = 0.05

MIN_WEIGHT = -1.5
MAX_WEIGHT = 1.5

MIN_ACTIVE_SENSOR = 5.0
LAST_X_MEMORY_S = 2.0


# ------------------------------------------------------------
# Reseau + Regle de Hebb (Algorithme 3)
# ------------------------------------------------------------

@dataclass
class ObstacleAvoidanceNetwork:
    """
    Reseau 5 entrees -> 2 sorties
    x = [x1, x2, x3, x4, x5]
    y = [y1, y2]
    """

    W: np.ndarray  # matrice (2, 5)

    @staticmethod
    def create_zero() -> "ObstacleAvoidanceNetwork":
        return ObstacleAvoidanceNetwork(W=np.zeros((2, 5), dtype=float))

    @staticmethod
    def create_small_random(seed: int = 7, scale: float = 0.02) -> "ObstacleAvoidanceNetwork":
        rng = np.random.default_rng(seed)
        w = rng.normal(loc=0.0, scale=scale, size=(2, 5))
        return ObstacleAvoidanceNetwork(W=w)

    @staticmethod
    def saturate(value: np.ndarray, low: float, high: float) -> np.ndarray:
        return np.clip(value, low, high)

    def infer(self, x: np.ndarray) -> np.ndarray:
        y = self.W @ x
        return self.saturate(y, -MAX_OUTPUT_SPEED, MAX_OUTPUT_SPEED)

    def hebb_update(self, x: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> None:
        """
        Algorithme 3 (forme implementee, stable num.)

        Pour j dans {1..5}:
            w_1j <- w_1j + alpha * y1 * xj
            w_2j <- w_2j + alpha * y2 * xj

        x et y sont normalises dans [-1,1]/[0,1] pour eviter divergence.
        """
        x_norm = x / 100.0
        y_norm = y / 100.0

        y1 = float(y_norm[0])
        y2 = float(y_norm[1])

        for j in range(5):
            xj = float(x_norm[j])
            self.W[0, j] = self.W[0, j] + alpha * y1 * xj
            self.W[1, j] = self.W[1, j] + alpha * y2 * xj

        self.W = self.saturate(self.W, MIN_WEIGHT, MAX_WEIGHT)


# ------------------------------------------------------------
# Outils capteurs / moteurs
# ------------------------------------------------------------

def clip_sensors_0_100(raw_x: np.ndarray) -> np.ndarray:
    return np.clip(raw_x, 0.0, 100.0)


def normalize_prox_horizontal(value: float) -> float:
    return float(np.clip((value / THYMIO_PROX_MAX) * 100.0, 0.0, 100.0))


def clamp_motor_target(speed: float) -> int:
    return int(np.clip(speed, -MAX_OUTPUT_SPEED, MAX_OUTPUT_SPEED))


def is_informative_x(x: np.ndarray, threshold: float = MIN_ACTIVE_SENSOR) -> bool:
    return bool(np.max(x) >= threshold)


# ------------------------------------------------------------
# Action enseignee (reprend Exo 2)
# ------------------------------------------------------------

def reward_action_from_button_states(
    forward_pressed: bool,
    backward_pressed: bool,
    left_pressed: bool,
    right_pressed: bool,
) -> Optional[np.ndarray]:
    if forward_pressed:
        return np.array([100.0, 100.0], dtype=float)

    if backward_pressed:
        return np.array([-100.0, -100.0], dtype=float)

    if left_pressed:
        return np.array([-100.0, 100.0], dtype=float)

    if right_pressed:
        return np.array([100.0, -100.0], dtype=float)

    return None


# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

def sensor_stream_example() -> List[np.ndarray]:
    return [
        np.array([80, 15, 20, 10, 15], dtype=float),
        np.array([20, 15, 85, 15, 10], dtype=float),
        np.array([35, 90, 40, 20, 20], dtype=float),
        np.array([15, 10, 15, 10, 10], dtype=float),
        np.array([75, 20, 25, 15, 15], dtype=float),
        np.array([25, 20, 80, 15, 10], dtype=float),
        np.array([30, 88, 35, 20, 20], dtype=float),
        np.array([10, 10, 10, 10, 10], dtype=float),
    ]


def reward_button_stream_example() -> List[np.ndarray]:
    return [
        np.array([0, 0, 0, 1]),
        np.array([0, 0, 1, 0]),
        np.array([0, 1, 0, 0]),
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 0, 0]),
        np.array([0, 0, 0, 0]),
        np.array([0, 0, 0, 1]),
        np.array([0, 0, 1, 0]),
    ]


def run_hebb_loop_simulation(
    network: ObstacleAvoidanceNetwork,
    cycles: int = 20,
    alpha: float = ALPHA,
) -> None:
    """
    Exo 3 simulation:
    - calcul y_net = W x
    - si action enseignee (bouton), applique y_teach
    - mise a jour Hebb suivant Algorithme 3 avec (x, y_teach)
    """
    x_stream = sensor_stream_example()
    b_stream = reward_button_stream_example()

    last_informative_x: Optional[np.ndarray] = None
    last_informative_t: float = -1.0

    print("\n--- Exo 3 : simulation regle de Hebb ---")
    for t in range(cycles):
        now = time.monotonic()
        x = clip_sensors_0_100(x_stream[t % len(x_stream)])

        if is_informative_x(x):
            last_informative_x = x.copy()
            last_informative_t = now

        y_net = network.infer(x)

        b = b_stream[t % len(b_stream)]
        y_teach = reward_action_from_button_states(
            forward_pressed=bool(b[0]),
            backward_pressed=bool(b[1]),
            left_pressed=bool(b[2]),
            right_pressed=bool(b[3]),
        )

        if y_teach is not None:
            x_train = x
            if not is_informative_x(x):
                if (
                    last_informative_x is not None
                    and last_informative_t > 0
                    and (now - last_informative_t) <= LAST_X_MEMORY_S
                ):
                    x_train = last_informative_x
                else:
                    x_train = None

            if x_train is not None:
                network.hebb_update(x=x_train, y=y_teach, alpha=alpha)
                mode = "HEBB"
            else:
                mode = "HEBB_SKIP"

            y_apply = y_teach
        else:
            y_apply = y_net
            mode = "AUTO"

        motor_left = clamp_motor_target(y_apply[0])
        motor_right = clamp_motor_target(y_apply[1])

        print(
            f"t={t:03d} mode={mode} "
            f"x={x.astype(int).tolist()} "
            f"y_net={[round(v, 1) for v in y_net.tolist()]} "
            f"y_apply=[{motor_left:4d},{motor_right:4d}]"
        )

        time.sleep(SAMPLE_PERIOD_S)

    print("\nMatrice W finale Exo 3 (simulation):")
    print(network.W)


# ------------------------------------------------------------
# Thymio reel
# ------------------------------------------------------------

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
    prox = th[node_id]["prox.horizontal"]

    x = np.array(
        [
            normalize_prox_horizontal(prox[0]),
            normalize_prox_horizontal(prox[2]),
            normalize_prox_horizontal(prox[4]),
            normalize_prox_horizontal(prox[5]),
            normalize_prox_horizontal(prox[6]),
        ],
        dtype=float,
    )

    return clip_sensors_0_100(x)


def read_reward_buttons_from_thymio(th, node_id: int) -> Tuple[bool, bool, bool, bool]:
    return (
        bool(th[node_id]["button.forward"]),
        bool(th[node_id]["button.backward"]),
        bool(th[node_id]["button.left"]),
        bool(th[node_id]["button.right"]),
    )


def run_hebb_loop_thymio(
    network: ObstacleAvoidanceNetwork,
    alpha: float = ALPHA,
) -> None:
    if Thymio is None:
        raise RuntimeError("Le package 'thymiodirect' n'est pas installe.")

    serial_port = find_thymio_serial_port()
    print(f"Port detecte : {serial_port}")

    th = Thymio(
        use_tcp=False,
        serial_port=serial_port,
        refreshing_coverage={
            "prox.horizontal",
            "button.forward",
            "button.backward",
            "button.left",
            "button.right",
            "button.center",
        },
    )
    th.connect()
    node_id = th.first_node()

    print("\n--- Exo 3 : Thymio reel regle de Hebb ---")
    print("Boutons directionnels = action enseignee, bouton centre = arret")

    t = 0
    last_informative_x: Optional[np.ndarray] = None
    last_informative_t: float = -1.0

    try:
        while True:
            now = time.monotonic()
            x = read_x_from_thymio(th, node_id)

            if is_informative_x(x):
                last_informative_x = x.copy()
                last_informative_t = now

            y_net = network.infer(x)

            forward, backward, left, right = read_reward_buttons_from_thymio(th, node_id)
            y_teach = reward_action_from_button_states(
                forward_pressed=forward,
                backward_pressed=backward,
                left_pressed=left,
                right_pressed=right,
            )

            if y_teach is not None:
                x_train = x
                if not is_informative_x(x):
                    if (
                        last_informative_x is not None
                        and last_informative_t > 0
                        and (now - last_informative_t) <= LAST_X_MEMORY_S
                    ):
                        x_train = last_informative_x
                    else:
                        x_train = None

                if x_train is not None:
                    network.hebb_update(x=x_train, y=y_teach, alpha=alpha)
                    mode = "HEBB"
                else:
                    mode = "HEBB_SKIP"

                y_apply = y_teach
            else:
                y_apply = y_net
                mode = "AUTO"

            motor_left = clamp_motor_target(y_apply[0])
            motor_right = clamp_motor_target(y_apply[1])

            th[node_id]["motor.left.target"] = motor_left
            th[node_id]["motor.right.target"] = motor_right

            print(
                f"t={t:04d} mode={mode} "
                f"x={x.astype(int).tolist()} "
                f"y_net={[round(v, 1) for v in y_net.tolist()]} "
                f"y_apply=[{motor_left:4d},{motor_right:4d}]"
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
            thymio_proxy = getattr(th, "thymio_proxy", None)
            if thymio_proxy is not None and getattr(thymio_proxy, "loop", None) is not None:
                thymio_proxy.loop.call_soon_threadsafe(thymio_proxy.loop.stop)
            if hasattr(th, "thread"):
                th.thread.join(timeout=1.0)
        except Exception:
            pass

    print("\nMatrice W finale Exo 3 (reel):")
    print(network.W)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exo 3 - Mise a jour des poids avec la regle de Hebb"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Execute Exo 3 sur le vrai robot Thymio.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="Vitesse d apprentissage Hebb (defaut: 0.05).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=20,
        help="Nombre de cycles en mode simulation.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Initialise W avec petits poids aleatoires au lieu de zeros.",
    )
    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=True)

    if args.random_init:
        net = ObstacleAvoidanceNetwork.create_small_random()
        print("Initialisation : petits poids aleatoires")
    else:
        net = ObstacleAvoidanceNetwork.create_zero()
        print("Initialisation : poids nuls")

    print("Matrice W initiale (2x5) :")
    print(net.W)

    if args.real:
        try:
            run_hebb_loop_thymio(net, alpha=args.alpha)
        except Exception as error:
            print(f"Erreur mode robot : {error}")
            sys.exit(1)
    else:
        run_hebb_loop_simulation(net, cycles=args.cycles, alpha=args.alpha)


if __name__ == "__main__":
    main()
