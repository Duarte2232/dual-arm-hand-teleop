#!/usr/bin/env python3
"""Headless end-to-end test: no camera, no RViz.

Plays a fake hand through /hand_state and checks that teleop_node + ik_node move the
arms, close the grippers, freeze on the dead-man switch and trigger the collision guard.

    source /opt/ros/humble/setup.bash && source install/setup.bash
    ros2 run dual_arm_hand_teleop ik_node.py &
    ros2 run dual_arm_hand_teleop teleop_node.py &
    python3 scripts/headless_pipeline_test.py
"""
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import JointState


class Probe(Node):
    def __init__(self):
        super().__init__('probe')
        self.pub = self.create_publisher(Float32MultiArray, '/hand_state', 10)
        self.js = None
        self.n_js = 0
        self.collision = False
        self.collision_seen = False
        self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        self.create_subscription(Bool, '/collision_state', self.on_col, 10)

    def on_js(self, m):
        self.js = m
        self.n_js += 1

    def on_col(self, m):
        self.collision = m.data
        self.collision_seen = self.collision_seen or m.data

    def send(self, lx, ly, lp, la, rx, ry, rp, ra, lact=1.0, ract=1.0):
        m = Float32MultiArray()
        m.data = [lx, ly, lp, la, rx, ry, rp, ra, lact, ract]
        self.pub.publish(m)

    def run_for(self, seconds, **kw):
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.send(**kw)
            rclpy.spin_once(self, timeout_sec=0.03)


def arm(js, side):
    d = dict(zip(js.name, js.position))
    return [d[f'{side}_{j}'] for j in ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')], \
        d[f'{side}_robotiq_85_left_knuckle_joint']


def dist(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


rclpy.init()
p = Probe()
ok = True


def check(name, cond, detail=''):
    global ok
    ok = ok and bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))


# let the nodes come up and the IK startup solve finish
t0 = time.time()
while p.js is None and time.time() - t0 < 25:
    rclpy.spin_once(p, timeout_sec=0.2)
check('ik_node publishes /joint_states', p.js is not None)
if p.js is None:
    raise SystemExit(1)
check('joint_states has 24 joints (2 arms x (6 + 6 gripper))', len(p.js.name) == 24, f'{len(p.js.name)}')

# pose A: hands mid-left / mid-right, pinch open
open_p, closed_p = 0.36, 0.056
p.run_for(4.0, lx=0.3, ly=0.4, lp=open_p, la=0.1, rx=0.7, ry=0.4, rp=open_p, ra=0.1)
A_l, gA_l = arm(p.js, 'left'); A_r, gA_r = arm(p.js, 'right')
# pose B: move both hands, close pinch
p.run_for(4.0, lx=0.6, ly=0.8, lp=closed_p, la=0.1, rx=0.4, ry=0.8, rp=closed_p, ra=0.1)
B_l, gB_l = arm(p.js, 'left'); B_r, gB_r = arm(p.js, 'right')
check('left arm joints move when the hand moves', dist(A_l, B_l) > 0.05, f'max dq = {dist(A_l, B_l):.2f} rad')
check('right arm joints move when the hand moves', dist(A_r, B_r) > 0.05, f'max dq = {dist(A_r, B_r):.2f} rad')
check('gripper closes with the pinch (left)', gB_l > gA_l + 0.3, f'{gA_l:.2f} -> {gB_l:.2f}')
check('gripper closes with the pinch (right)', gB_r > gA_r + 0.3, f'{gA_r:.2f} -> {gB_r:.2f}')

# dead-man: left hand inactive, right keeps moving
p.run_for(1.0, lx=0.6, ly=0.8, lp=closed_p, la=0.1, rx=0.4, ry=0.8, rp=closed_p, ra=0.1)
F0_l, _ = arm(p.js, 'left')
p.run_for(3.0, lx=0.1, ly=0.1, lp=open_p, la=0.1, rx=0.9, ry=0.1, rp=open_p, ra=0.1, lact=0.0, ract=1.0)
F1_l, _ = arm(p.js, 'left'); F1_r, _ = arm(p.js, 'right')
check('dead-man switch freezes the inactive left arm', dist(F0_l, F1_l) < 0.01, f'max dq = {dist(F0_l, F1_l):.4f} rad')
check('active right arm still moves', dist(B_r, F1_r) > 0.05, f'max dq = {dist(B_r, F1_r):.2f} rad')

# collision: push both hands to the middle, where the end-effectors meet
p.collision_seen = False
p.run_for(5.0, lx=1.0, ly=0.5, lp=open_p, la=0.1, rx=0.0, ry=0.5, rp=open_p, ra=0.1)
check('collision guard triggers when both end-effectors meet', p.collision_seen)

print('\nRESULT: ' + ('ALL PASS' if ok else 'SOME FAILED') + f'  ({p.n_js} joint_state messages)')
p.destroy_node()
rclpy.shutdown()
raise SystemExit(0 if ok else 1)
