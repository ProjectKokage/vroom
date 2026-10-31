"""Measure ARDY SOMA generations (.npz from scripts/generate.py).

Per clip: finger joint angular range, upper-arm abduction and forward angle,
elbow bend, thigh outward angle, knee/ankle/hip-joint separation.
Run with ``python -I measure.py label=path.npz ...`` (numpy only, no torch).
Prints one line per clip and writes measurements.json next to the clips.

Definitions (matching the arms.py and measure.py of Kokage's ignored
``output/motion-clips/stance`` tools used for the v3 sets, so the numbers
compare):
- arm from down: angle between the upper-arm bone (Arm -> ForeArm) and
  straight down, from posed joint positions; 0 = arm hanging, 90 = T-pose.
- arm fwd: elevation of the same bone in the sagittal plane, + = forward.
- elbow (local): rotation angle of ForeArm's local rotation from the
  skeleton rest (what arms.py reports on VRMA as "elbow").
- elbow (bend): angle between upper-arm and forearm bones from positions.
- thigh out: mean outward angle of each thigh (Leg -> Shin) from vertical in
  the frontal plane; + = knee outward.
- knee/ankle/hip-joint separation: distance between the left and right
  Shin, Foot and Leg joints, in centimetres (SOMA output is in metres).
- fingers: for every joint of the finger chains, the rotation angle between
  its local rotation at frame t and at frame 0; the reported range is the
  largest over joints and frames. 0 means the fingers never move.
"""
import json
import os
import sys

import numpy as np

SOMA77 = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye",
    "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumb1",
    "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd", "LeftHandIndex1", "LeftHandIndex2",
    "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd", "LeftHandMiddle1", "LeftHandMiddle2",
    "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd", "LeftHandRing1", "LeftHandRing2",
    "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd", "LeftHandPinky1", "LeftHandPinky2",
    "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd", "RightShoulder", "RightArm",
    "RightForeArm", "RightHand", "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
    "RightHandThumbEnd", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandIndex4", "RightHandIndexEnd", "RightHandMiddle1", "RightHandMiddle2",
    "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd", "RightHandRing1",
    "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd", "RightHandPinky1",
    "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd", "LeftLeg",
    "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd", "RightLeg", "RightShin", "RightFoot",
    "RightToeBase", "RightToeEnd",
]
FINGER_JOINTS = [n for n in SOMA77 if "Hand" in n and n[-4:] != "Hand"]
FINGER_CHAIN_JOINTS = [n for n in FINGER_JOINTS if not n.endswith("End")]
HAND_END_MARKERS = [n for n in FINGER_JOINTS if n.endswith("End")]


def rot_angle_deg(R):
    """Rotation angle of a batch of 3x3 matrices, degrees."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def measure(path):
    z = np.load(path, allow_pickle=False)
    P = z["posed_joints"].astype(np.float64)  # [T, 77, 3], metres, Y up
    L = z["local_rot_mats"].astype(np.float64)  # [T, 77, 3, 3]
    if P.shape[1] != len(SOMA77):
        raise SystemExit(f"{path}: expected 77 joints, got {P.shape[1]}")
    J = {n: i for i, n in enumerate(SOMA77)}
    out = {"frames": int(P.shape[0]), "fps": float(z["fps"]), "text": str(z["text"])}

    # Fingers: local rotation at frame t relative to frame 0, largest angle.
    def finger_range(names):
        idx = [J[n] for n in names]
        R0 = L[0, idx]  # [n, 3, 3]
        rel = np.einsum("tnij,nkj->tnik", L[:, idx], R0)  # L_t * L_0^T
        ang = rot_angle_deg(rel)  # [T, n]
        per_joint = ang.max(axis=0)
        return float(per_joint.max()), {names[k]: round(float(per_joint[k]), 2) for k in range(len(names))}

    chain_max, chain_per = finger_range(FINGER_CHAIN_JOINTS)
    end_max, _ = finger_range(HAND_END_MARKERS)
    # Rotation of finger chains against the whole clip mean, in case frame 0 is
    # an outlier: mean over frames of angle to frame 0 is a second summary.
    out["fingerChainMaxDeg"] = round(chain_max, 3)
    out["handEndMarkerMaxDeg"] = round(end_max, 3)
    out["fingerChainPerJointMaxDeg"] = chain_per

    def bone(a, b):
        return unit(P[:, J[b]] - P[:, J[a]])

    down = np.array([0.0, -1.0, 0.0])
    arms = {}
    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        up_arm = bone(f"{side}Arm", f"{side}ForeArm")
        fore = bone(f"{side}ForeArm", f"{side}Hand")
        abd = np.degrees(np.arccos(np.clip(up_arm @ down, -1, 1)))
        fwd = np.degrees(np.arcsin(np.clip(up_arm[:, 2], -1, 1)))
        bend = np.degrees(np.arccos(np.clip(np.sum(up_arm * fore, axis=-1), -1, 1)))
        local = rot_angle_deg(L[:, J[f"{side}ForeArm"]])
        clav = rot_angle_deg(L[:, J[f"{side}Shoulder"]])
        thigh = bone(f"{side}Leg", f"{side}Shin")
        thigh_out = np.degrees(np.arcsin(np.clip(thigh[:, 0] * sx, -1, 1)))
        arms[side.lower()] = {
            "armFromDownDeg": round(float(abd.mean()), 1),
            "armForwardDeg": round(float(fwd.mean()), 1),
            "elbowLocalDeg": round(float(local.mean()), 1),
            "elbowBendDeg": round(float(bend.mean()), 1),
            "clavicleDeg": round(float(clav.mean()), 1),
            "thighOutDeg": round(float(thigh_out.mean()), 1),
        }
    out["sides"] = arms

    def sep(a, b):
        d = np.linalg.norm(P[:, J[a]] - P[:, J[b]], axis=-1) * 100
        return {"mean": round(float(d.mean()), 1), "min": round(float(d.min()), 1), "max": round(float(d.max()), 1)}

    out["kneeSeparationCm"] = sep("LeftShin", "RightShin")
    out["ankleSeparationCm"] = sep("LeftFoot", "RightFoot")
    out["hipJointSeparationCm"] = sep("LeftLeg", "RightLeg")
    out["hipsHeightM"] = round(float(P[:, J["Hips"]][:, 1].mean()), 3)
    out["headHeightM"] = round(float(P[:, J["Head"]][:, 1].mean()), 3)
    # Head motion, for the one-shots: pitch/yaw range of the head's global rotation.
    G = z["global_rot_mats"].astype(np.float64)[:, J["Head"]]
    fwd_vec = G @ np.array([0.0, 0.0, 1.0])
    out["headYawRangeDeg"] = round(float(np.ptp(np.degrees(np.arctan2(fwd_vec[:, 0], fwd_vec[:, 2])))), 1)
    out["headPitchRangeDeg"] = round(float(np.ptp(np.degrees(np.arcsin(np.clip(fwd_vec[:, 1], -1, 1))))), 1)
    return out


def main():
    results = {}
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        m = measure(path)
        results[label] = m
        l, r = m["sides"]["left"], m["sides"]["right"]
        print(
            f"{label:16} fingers {m['fingerChainMaxDeg']:6.3f}° (end markers {m['handEndMarkerMaxDeg']:6.3f}°) | "
            f"arm L {l['armFromDownDeg']:4.1f}° R {r['armFromDownDeg']:4.1f}° from down, "
            f"fwd L {l['armForwardDeg']:+5.1f}° R {r['armForwardDeg']:+5.1f}° | "
            f"elbow local L {l['elbowLocalDeg']:5.1f}° R {r['elbowLocalDeg']:5.1f}°, "
            f"bend L {l['elbowBendDeg']:5.1f}° R {r['elbowBendDeg']:5.1f}° | "
            f"thigh out L {l['thighOutDeg']:4.1f}° R {r['thighOutDeg']:4.1f}° | "
            f"knees {m['kneeSeparationCm']['mean']:4.1f} cm (hip joints {m['hipJointSeparationCm']['mean']:4.1f}, "
            f"ankles {m['ankleSeparationCm']['mean']:4.1f}) | head yaw {m['headYawRangeDeg']:5.1f}° pitch {m['headPitchRangeDeg']:5.1f}°"
        )
    if results:
        first = sys.argv[1].split("=", 1)[1]
        out = os.path.join(os.path.dirname(os.path.abspath(first)), "measurements.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
