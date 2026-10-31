"""Check the SOMA77 name table of vrm_map.py against a generated .npz.

The npz carries no joint names, so the check is structural: bone lengths
from frame 0 under the assumed names must be anatomically plausible
(thigh and shin about 40 cm, upper arm and forearm about 25 cm, finger
segments a few cm), each child must sit where its parent chain says, and
the relaxed-hand finger rotations must be identical on every frame and
across takes. Run with ``python -I check_skeleton.py take.npz [take.npz ...]``.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vrm_map import SOMA77  # noqa: E402

NAMES = [n for n, _ in SOMA77]
PARENT = dict(SOMA77)
J = {n: i for i, n in enumerate(NAMES)}


def bone_lengths(P):
    out = {}
    for name, parent in SOMA77:
        if parent is None:
            continue
        out[name] = float(np.linalg.norm(P[J[name]] - P[J[parent]]) * 100)
    return out


def main():
    first_fingers = None
    for path in sys.argv[1:]:
        z = np.load(path, allow_pickle=False)
        P0 = z["posed_joints"][0].astype(np.float64)
        G0 = z["global_rot_mats"][0].astype(np.float64)
        L = z["local_rot_mats"].astype(np.float64)
        lengths = bone_lengths(P0)
        print(f"== {path}: {P0.shape[0]} joints, {z['fps']} fps, {L.shape[0]} frames")
        show = ["Spine1", "Chest", "Neck1", "Head", "HeadEnd", "LeftShoulder", "LeftArm", "LeftForeArm",
                "LeftHand", "LeftHandThumb1", "LeftHandThumb2", "LeftHandIndex1", "LeftHandIndex2",
                "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd", "LeftHandMiddle1",
                "LeftHandPinky1", "LeftHandPinky2", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
                "LeftToeEnd", "RightLeg", "RightShin"]
        print("  bone lengths (cm):", ", ".join(f"{n} {lengths[n]:.1f}" for n in show))
        # Child offsets in the parent's frame for the first finger joints: a
        # metacarpal joint sits inside the palm near the wrist; a knuckle sits
        # at the far edge of the palm.
        for f in ("Index", "Middle", "Ring", "Pinky"):
            j1, j2 = J[f"LeftHand{f}1"], J[f"LeftHand{f}2"]
            hand = J["LeftHand"]
            d1 = np.linalg.norm(P0[j1] - P0[hand]) * 100
            d2 = np.linalg.norm(P0[j2] - P0[hand]) * 100
            print(f"  Left {f}: joint1 {d1:.1f} cm from wrist, joint2 {d2:.1f} cm from wrist")
        # Vertical order sanity under Y up.
        checks = [("Head", "Hips"), ("Hips", "LeftShin"), ("LeftShin", "LeftFoot"), ("LeftArm", "LeftHand")]
        for a, b in checks:
            print(f"  {a} above {b}: {P0[J[a], 1] > P0[J[b], 1]}")
        # Finger rotations: constant over frames and identical across takes.
        fingers = [J[n] for n in NAMES if "Hand" in n and n[-4:] != "Hand" and not n.endswith("End")]
        spread = np.abs(L[:, fingers] - L[0, fingers]).max()
        print(f"  finger local rotations max |delta| over frames: {spread:.2e}")
        if first_fingers is None:
            first_fingers = L[0, fingers]
        else:
            print(f"  finger rotations differ from first take by max {np.abs(L[0, fingers] - first_fingers).max():.2e}")
        # Hip height and the heading of frame 0 (facing +Z).
        fwd = G0[J["Hips"]] @ np.array([0, 0, 1.0])
        print(f"  hips height {P0[J['Hips'], 1]:.3f} m, head height {P0[J['Head'], 1]:.3f} m, hips forward xz=({fwd[0]:+.2f},{fwd[2]:+.2f})")


if __name__ == "__main__":
    main()
