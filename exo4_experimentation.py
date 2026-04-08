import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
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

ALPHA = 0.05
MIN_WEIGHT = -1.5
MAX_WEIGHT = 1.5

MIN_ACTIVE_SENSOR = 5.0
LAST_X_MEMORY_S = 2.0
NO_OBS_THRESHOLD = 12.0


# ------------------------------------------------------------
# Reseau + Hebb
# ------------------------------------------------------------

@dataclass
class ObstacleAvoidanceNetwork:
    W: np.ndarray  # (2, 6) = [bias, x1, x2, x3, x4, x5]

    @staticmethod
    def create_zero() -> "ObstacleAvoidanceNetwork":
        return ObstacleAvoidanceNetwork(W=np.zeros((2, 6), dtype=float))

    @staticmethod
    def create_small_random(seed: int = 7, scale: float = 0.02) -> "ObstacleAvoidanceNetwork":
        rng = np.random.default_rng(seed)
        return ObstacleAvoidanceNetwork(W=rng.normal(0.0, scale, size=(2, 6)))

    @staticmethod
    def saturate(value: np.ndarray, low: float, high: float) -> np.ndarray:
        return np.clip(value, low, high)

    def augment_with_bias(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.array([100.0], dtype=float), x])

    def infer(self, x: np.ndarray) -> np.ndarray:
        x_aug = self.augment_with_bias(x)
        y = self.W @ x_aug
        return self.saturate(y, -MAX_OUTPUT_SPEED, MAX_OUTPUT_SPEED)

    def hebb_update(self, x: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> None:
        # Algorithme 3: w_ij <- w_ij + alpha * y_i * x_j
        x_aug = self.augment_with_bias(x)
        x_norm = x_aug / 100.0
        y_norm = y / 100.0

        y1 = float(y_norm[0])
        y2 = float(y_norm[1])
        for j in range(6):
            xj = float(x_norm[j])
            self.W[0, j] += alpha * y1 * xj
            self.W[1, j] += alpha * y2 * xj

        self.W = self.saturate(self.W, MIN_WEIGHT, MAX_WEIGHT)


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def clip_sensors_0_100(raw_x: np.ndarray) -> np.ndarray:
    return np.clip(raw_x, 0.0, 100.0)


def normalize_prox_horizontal(value: float) -> float:
    return float(np.clip((value / THYMIO_PROX_MAX) * 100.0, 0.0, 100.0))


def clamp_motor_target(speed: float) -> int:
    return int(np.clip(speed, -MAX_OUTPUT_SPEED, MAX_OUTPUT_SPEED))


def is_informative_x(x: np.ndarray, threshold: float = MIN_ACTIVE_SENSOR) -> bool:
    return bool(np.max(x) >= threshold)


def is_no_obstacle_x(x: np.ndarray, threshold: float = NO_OBS_THRESHOLD) -> bool:
    return bool(np.max(x) <= threshold)


def save_weights(network: ObstacleAvoidanceNetwork, output_prefix: str) -> Tuple[Path, Path]:
    out_prefix = Path(output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    npy_path = out_prefix.with_suffix(".npy")
    txt_path = out_prefix.with_suffix(".txt")

    np.save(npy_path, network.W)
    np.savetxt(txt_path, network.W, fmt="%.6f")
    return npy_path, txt_path


# ------------------------------------------------------------
# Enseignement
# ------------------------------------------------------------

def action_from_buttons(
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


def teacher_action_for_task(task: str, x: np.ndarray) -> Optional[np.ndarray]:
    x1, x2, x3, x4, x5 = [float(v) for v in x]

    if task == "avoid":
        if x2 > 60:
            return np.array([-100.0, -100.0], dtype=float)
        if x1 > 45 and x1 >= x3:
            return np.array([100.0, -100.0], dtype=float)
        if x3 > 45 and x3 > x1:
            return np.array([-100.0, 100.0], dtype=float)
        if x4 > 45:
            return np.array([100.0, -100.0], dtype=float)
        if x5 > 45:
            return np.array([-100.0, 100.0], dtype=float)
        return None

    if task == "forward":
        if is_no_obstacle_x(x):
            return np.array([100.0, 100.0], dtype=float)
        return None

    if task == "both":
        # Priorite: eviter obstacle, sinon apprendre a avancer hors obstacle.
        avoid = teacher_action_for_task("avoid", x)
        if avoid is not None:
            return avoid
        return teacher_action_for_task("forward", x)

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


def run_experiment_simulation(
    network: ObstacleAvoidanceNetwork,
    task: str,
    cycles: int,
    alpha: float,
    auto_teach: bool,
) -> None:
    stream = sensor_stream_example()
    last_informative_x: Optional[np.ndarray] = None
    last_informative_t: float = -1.0

    print(f"\n--- Exo 4 simulation: task={task} ---")
    for t in range(cycles):
        now = time.monotonic()
        x = clip_sensors_0_100(stream[t % len(stream)])

        if is_informative_x(x):
            last_informative_x = x.copy()
            last_informative_t = now

        y_net = network.infer(x)
        y_teach = teacher_action_for_task(task, x) if auto_teach else None

        if y_teach is not None:
            x_train = x
            using_forward_teacher = bool(np.allclose(y_teach, np.array([100.0, 100.0])))

            # Important Exo 4: ne pas reutiliser la memoire pour apprendre l'avance
            # en absence d'obstacle.
            if (not using_forward_teacher) and (not is_informative_x(x)):
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

        print(
            f"t={t:03d} mode={mode} x={x.astype(int).tolist()} "
            f"y_net={[round(v, 1) for v in y_net.tolist()]} "
            f"y_apply=[{clamp_motor_target(y_apply[0]):4d},{clamp_motor_target(y_apply[1]):4d}]"
        )
        time.sleep(SAMPLE_PERIOD_S)


# ------------------------------------------------------------
# Thymio
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


def read_buttons(th, node_id: int) -> Tuple[bool, bool, bool, bool, bool]:
    return (
        bool(th[node_id]["button.forward"]),
        bool(th[node_id]["button.backward"]),
        bool(th[node_id]["button.left"]),
        bool(th[node_id]["button.right"]),
        bool(th[node_id]["button.center"]),
    )


def run_experiment_thymio(
    network: ObstacleAvoidanceNetwork,
    task: str,
    alpha: float,
    auto_teach: bool,
) -> None:
    if Thymio is None:
        raise RuntimeError("Le package thymiodirect n est pas installe.")

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

    print(f"\n--- Exo 4 reel: task={task} ---")
    print("Boutons directionnels: enseignement manuel | centre: arret")
    if auto_teach:
        print("Auto-teach actif: le script genere aussi un enseignement selon le task.")

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

            fwd, bwd, left, right, center = read_buttons(th, node_id)
            y_teach_manual = action_from_buttons(fwd, bwd, left, right)
            y_teach_auto = teacher_action_for_task(task, x) if auto_teach else None
            y_teach = y_teach_manual if y_teach_manual is not None else y_teach_auto

            if y_teach is not None:
                x_train = x
                using_forward_teacher = bool(np.allclose(y_teach, np.array([100.0, 100.0])))

                # Important Exo 4: ne pas reutiliser la memoire pour apprendre l'avance
                # en absence d'obstacle.
                if (not using_forward_teacher) and (not is_informative_x(x)):
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
                mode = "AUTO"
                y_apply = y_net

            motor_left = clamp_motor_target(y_apply[0])
            motor_right = clamp_motor_target(y_apply[1])
            th[node_id]["motor.left.target"] = motor_left
            th[node_id]["motor.right.target"] = motor_right

            print(
                f"t={t:04d} mode={mode} x={x.astype(int).tolist()} "
                f"y_net={[round(v, 1) for v in y_net.tolist()]} y_apply=[{motor_left:4d},{motor_right:4d}]"
            )
            t += 1

            if center:
                print("Arret demande par bouton central.")
                break

            time.sleep(SAMPLE_PERIOD_S)

    finally:
        th[node_id]["motor.left.target"] = 0
        th[node_id]["motor.right.target"] = 0
        try:
            proxy = getattr(th, "thymio_proxy", None)
            if proxy is not None and getattr(proxy, "loop", None) is not None:
                proxy.loop.call_soon_threadsafe(proxy.loop.stop)
            if hasattr(th, "thread"):
                th.thread.join(timeout=1.0)
        except Exception:
            pass


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exo 4 - Experimentation (evitement + avance sans obstacle)"
    )
    parser.add_argument("--real", action="store_true", help="Execution sur robot reel")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Taux Hebb")
    parser.add_argument("--cycles", type=int, default=40, help="Cycles simulation")
    parser.add_argument(
        "--task",
        type=str,
        choices=["avoid", "forward", "both"],
        default="both",
        help="Scenario d experimentation Exo 4",
    )
    parser.add_argument(
        "--auto-teach",
        action="store_true",
        help="Active un enseignant automatique selon task (utile pour demo).",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Initialise W aleatoirement (sinon W=0).",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="exo4_poids_finaux",
        help="Prefix du fichier de sauvegarde des poids (.npy et .txt)",
    )
    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=True)

    if args.random_init:
        net = ObstacleAvoidanceNetwork.create_small_random()
        print("Initialisation : poids aleatoires")
    else:
        net = ObstacleAvoidanceNetwork.create_zero()
        print("Initialisation : poids nuls")

    print("Matrice W initiale (2x6, biais inclus):")
    print(net.W)

    if args.real:
        try:
            run_experiment_thymio(net, task=args.task, alpha=args.alpha, auto_teach=args.auto_teach)
        except Exception as error:
            print(f"Erreur mode reel: {error}")
            sys.exit(1)
    else:
        run_experiment_simulation(
            net,
            task=args.task,
            cycles=args.cycles,
            alpha=args.alpha,
            auto_teach=args.auto_teach,
        )

    print("\nMatrice W finale:")
    print(net.W)

    npy_path, txt_path = save_weights(net, args.save_prefix)
    print(f"Poids sauvegardes: {npy_path}")
    print(f"Poids sauvegardes: {txt_path}")


if __name__ == "__main__":
    main()
