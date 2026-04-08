# TP Reinforcement Learning - Thymio

## Overview

This project contains a full practical workflow for reinforcement learning on Thymio, split into 4 exercises:

1. Exo 1: network model and periodic control (y = W x)
2. Exo 2: reward by buttons to teach actions
3. Exo 3: explicit Hebb rule update of weights
4. Exo 4: experimentation and validation on real robot

The project supports both:

- simulation mode
- real robot mode (Thymio via serial)

## Project Files

- exo1_hebb_evitement_obstacle.py
- exo1_evitement_obstacle.md
- exo2_recompense.py
- exo2_recompense.md
- exo3_regle_hebb.py
- exo3_regle_hebb.md
- exo4_experimentation.py
- exo4_experimentation.md

Generated outputs (examples):

- exo4_poids_finaux.npy
- exo4_poids_finaux.txt

## Prerequisites

- Python 3.10+
- Thymio robot (for real mode)
- USB serial connection

Python packages:

- numpy
- pyserial
- thymiodirect

Install example:

```powershell
pip install numpy pyserial thymiodirect
```

## Exercise 1

Goal:

- Build and run the base network controller with 5 inputs and 2 outputs.

Run:

```powershell
python exo1_hebb_evitement_obstacle.py
python exo1_hebb_evitement_obstacle.py --real
```

## Exercise 2

Goal:

- Add reward with 4 buttons and update W from taught action.

Button mapping:

- forward -> [100, 100]
- backward -> [-100, -100]
- left -> [-100, 100]
- right -> [100, -100]

Run:

```powershell
python exo2_recompense.py --cycles 20
python exo2_recompense.py --real
python exo2_recompense.py --real --alpha 0.03
```

## Exercise 3

Goal:

- Apply explicit Hebb rule from algorithm 3:
  - w1j <- w1j + alpha * y1 * xj
  - w2j <- w2j + alpha * y2 * xj

Run:

```powershell
python exo3_regle_hebb.py --cycles 20
python exo3_regle_hebb.py --real
python exo3_regle_hebb.py --real --alpha 0.03
```

## Exercise 4

Goal:

1. Teach obstacle avoidance
2. Teach forward motion when no obstacle is present

Important model change:

- Bias input added to solve no-obstacle forward learning
- W shape is (2, 6): [bias, x1, x2, x3, x4, x5]

Run (simulation):

```powershell
python exo4_experimentation.py --task both --auto-teach --cycles 40 --save-prefix exo4_both
python exo4_experimentation.py --task avoid --auto-teach --cycles 40 --save-prefix exo4_avoid
python exo4_experimentation.py --task forward --auto-teach --cycles 40 --save-prefix exo4_forward
```

Run (real robot):

```powershell
python exo4_experimentation.py --real --task both --alpha 0.03 --save-prefix exo4_both_real
python exo4_experimentation.py --real --task both --alpha 0.03 --auto-teach --save-prefix exo4_both_real
```

Manual teaching in real mode:

- direction buttons: teach action
- center button: stop

## What To Submit

For each exercise (as required by your TP):

1. Script file
2. Demonstration video
3. Final model weights

For Exo 4, include:

- exo4_experimentation.py
- exo4_experimentation.md
- weight files (.npy and .txt)
- video for obstacle avoidance
- video for forward-in-no-obstacle behavior

## Troubleshooting

If the robot is not detected:

1. Check USB cable and robot power
2. Close other apps using serial port
3. Re-run script and verify detected COM port in console

If learning seems ineffective:

1. Reduce alpha (example 0.03)
2. Teach with clear sensor situations
3. In manual mode, use short button presses
4. Validate learned behavior in AUTO mode without pressing buttons
