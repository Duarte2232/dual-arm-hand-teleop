#!/usr/bin/env python3
"""Offline check of the IK stage (no ROS needed): how well does ik_node reach the teleop workspace?

Uses the same URDF, chain, home pose, 0.85/0.15 warm start and 10 iterations per tick as
ik_node.py. Hand positions are swept over a grid (as the operator moving a hand), mapped to
targets exactly as teleop_node.py does (fixed depth x = 0.4 m, y and z from the hand), and the
distance between the target and the forward-kinematics position of the solution is reported:
  - after 1 tick  (a jump to a new target)
  - after 20 ticks (about 1 s at 20 Hz, hand held still)

    python3 scripts/ik_workspace_check.py
"""
import os
import time
import numpy as np
import ikpy.chain

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'src', 'dual_arm_hand_teleop', 'urdf', 'dual_arm.urdf')
MASK = [False, False, False, True, True, True, True, True, True, False]
HOME = np.array([0, 0, 0, 0.0, -1.0, 1.4, -1.7, 0.0, 0.0, 0.0])
X, Z_MIN, Z_MAX = 0.4, 0.1, 0.7
Y_RANGE = {'left': (-0.5, 0.0), 'right': (0.0, 0.5)}


def chain(side):
    return ikpy.chain.Chain.from_urdf_file(
        URDF, base_elements=['world', f'{side}_base_joint', f'{side}_base_link'],
        active_links_mask=MASK)


def tick(ch, angles, target):
    t0 = time.perf_counter()
    new = ch.inverse_kinematics(target, target_orientation=[0, 0, 1], orientation_mode='Z',
                                initial_position=0.85 * angles + 0.15 * HOME, max_iter=10)
    dt = time.perf_counter() - t0
    return np.clip(new, -2 * np.pi, 2 * np.pi), dt


def err(ch, angles, target):
    return float(np.linalg.norm(ch.forward_kinematics(angles)[:3, 3] - np.array(target)))


for side in ('left', 'right'):
    ch = chain(side)
    y0 = -0.5 if side == 'left' else 0.5
    angles = np.clip(ch.inverse_kinematics([0.3, y0, 0.5], target_orientation=[0, 0, 1],
                                           orientation_mode='Z', initial_position=HOME,
                                           max_iter=200), -2 * np.pi, 2 * np.pi)
    ymin, ymax = Y_RANGE[side]
    e1, e20, times = [], [], []
    for hx in np.linspace(0, 1, 11):
        for hy in np.linspace(0, 1, 13):
            target = [X, ymin + hx * (ymax - ymin), Z_MAX - hy * (Z_MAX - Z_MIN)]
            angles, dt = tick(ch, angles, target)
            times.append(dt)
            e1.append(err(ch, angles, target))
            for _ in range(19):
                angles, dt = tick(ch, angles, target)
                times.append(dt)
            e20.append(err(ch, angles, target))
    e1, e20, times = np.array(e1), np.array(e20), np.array(times) * 1000
    print(f'{side:5s} arm, {len(e1)} targets | IK time per tick: median {np.median(times):.1f} ms, '
          f'p95 {np.percentile(times, 95):.1f} ms (tick budget at 20 Hz: 50 ms)')
    for name, e in (('after 1 tick ', e1), ('after 20 ticks', e20)):
        print(f'      {name}: median error {np.median(e) * 1000:6.2f} mm, max {np.max(e) * 1000:6.2f} mm | '
              f'within 1 cm: {np.mean(e < 0.01) * 100:3.0f}%')
