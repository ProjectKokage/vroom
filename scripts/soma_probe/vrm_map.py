"""SOMA 77-joint skeleton -> VRM humanoid bone map.

Joint names and parents are copied from nv-tlabs/ardy e706e27
ardy/skeleton/definitions.py (SOMASkeleton77 and SOMASkeleton30). The map
is static: it needs no generated motion. Run with ``python -I vrm_map.py``.
Prints a markdown table and writes vrm_map.json into the current directory.
"""
import json
import os
import sys

SOMA77 = [
    ("Hips", None), ("Spine1", "Hips"), ("Spine2", "Spine1"), ("Chest", "Spine2"),
    ("Neck1", "Chest"), ("Neck2", "Neck1"), ("Head", "Neck2"), ("HeadEnd", "Head"),
    ("Jaw", "Head"), ("LeftEye", "Head"), ("RightEye", "Head"),
    ("LeftShoulder", "Chest"), ("LeftArm", "LeftShoulder"), ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("LeftHandThumb1", "LeftHand"), ("LeftHandThumb2", "LeftHandThumb1"),
    ("LeftHandThumb3", "LeftHandThumb2"), ("LeftHandThumbEnd", "LeftHandThumb3"),
    ("LeftHandIndex1", "LeftHand"), ("LeftHandIndex2", "LeftHandIndex1"),
    ("LeftHandIndex3", "LeftHandIndex2"), ("LeftHandIndex4", "LeftHandIndex3"),
    ("LeftHandIndexEnd", "LeftHandIndex4"),
    ("LeftHandMiddle1", "LeftHand"), ("LeftHandMiddle2", "LeftHandMiddle1"),
    ("LeftHandMiddle3", "LeftHandMiddle2"), ("LeftHandMiddle4", "LeftHandMiddle3"),
    ("LeftHandMiddleEnd", "LeftHandMiddle4"),
    ("LeftHandRing1", "LeftHand"), ("LeftHandRing2", "LeftHandRing1"),
    ("LeftHandRing3", "LeftHandRing2"), ("LeftHandRing4", "LeftHandRing3"),
    ("LeftHandRingEnd", "LeftHandRing4"),
    ("LeftHandPinky1", "LeftHand"), ("LeftHandPinky2", "LeftHandPinky1"),
    ("LeftHandPinky3", "LeftHandPinky2"), ("LeftHandPinky4", "LeftHandPinky3"),
    ("LeftHandPinkyEnd", "LeftHandPinky4"),
    ("RightShoulder", "Chest"), ("RightArm", "RightShoulder"), ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
    ("RightHandThumb1", "RightHand"), ("RightHandThumb2", "RightHandThumb1"),
    ("RightHandThumb3", "RightHandThumb2"), ("RightHandThumbEnd", "RightHandThumb3"),
    ("RightHandIndex1", "RightHand"), ("RightHandIndex2", "RightHandIndex1"),
    ("RightHandIndex3", "RightHandIndex2"), ("RightHandIndex4", "RightHandIndex3"),
    ("RightHandIndexEnd", "RightHandIndex4"),
    ("RightHandMiddle1", "RightHand"), ("RightHandMiddle2", "RightHandMiddle1"),
    ("RightHandMiddle3", "RightHandMiddle2"), ("RightHandMiddle4", "RightHandMiddle3"),
    ("RightHandMiddleEnd", "RightHandMiddle4"),
    ("RightHandRing1", "RightHand"), ("RightHandRing2", "RightHandRing1"),
    ("RightHandRing3", "RightHandRing2"), ("RightHandRing4", "RightHandRing3"),
    ("RightHandRingEnd", "RightHandRing4"),
    ("RightHandPinky1", "RightHand"), ("RightHandPinky2", "RightHandPinky1"),
    ("RightHandPinky3", "RightHandPinky2"), ("RightHandPinky4", "RightHandPinky3"),
    ("RightHandPinkyEnd", "RightHandPinky4"),
    ("LeftLeg", "Hips"), ("LeftShin", "LeftLeg"), ("LeftFoot", "LeftShin"),
    ("LeftToeBase", "LeftFoot"), ("LeftToeEnd", "LeftToeBase"),
    ("RightLeg", "Hips"), ("RightShin", "RightLeg"), ("RightFoot", "RightShin"),
    ("RightToeBase", "RightFoot"), ("RightToeEnd", "RightToeBase"),
]

# The 30 joints the SOMA checkpoints actually predict (SOMASkeleton30).
SOMA30 = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye",
    "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd",
    "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot",
    "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]

# VRM 1.0 humanoid bone names. A SOMA joint maps to one VRM bone, to None
# when it has no counterpart, or to "fold:<bone>" when its rotation is
# composed into that bone (as Core's Spine1 is folded into Spine2 today).
VRM = {
    "Hips": "hips", "Spine1": "spine", "Spine2": "chest", "Chest": "upperChest",
    "Neck1": "neck", "Neck2": "fold:neck", "Head": "head",
    "HeadEnd": None, "Jaw": "jaw", "LeftEye": "leftEye", "RightEye": "rightEye",
    "LeftShoulder": "leftShoulder", "LeftArm": "leftUpperArm", "LeftForeArm": "leftLowerArm",
    "LeftHand": "leftHand",
    "RightShoulder": "rightShoulder", "RightArm": "rightUpperArm",
    "RightForeArm": "rightLowerArm", "RightHand": "rightHand",
    "LeftLeg": "leftUpperLeg", "LeftShin": "leftLowerLeg", "LeftFoot": "leftFoot",
    "LeftToeBase": "leftToes", "LeftToeEnd": None,
    "RightLeg": "rightUpperLeg", "RightShin": "rightLowerLeg", "RightFoot": "rightFoot",
    "RightToeBase": "rightToes", "RightToeEnd": None,
}
# Fingers. SOMA Thumb1..3 -> VRM thumbMetacarpal/Proximal/Distal. SOMA
# Index..Pinky have four joints each; VRM has three (proximal, intermediate,
# distal). Joint 1 is taken as the metacarpal (inside the palm), folded into
# the hand, and 2..4 map to proximal..distal. Whether joint 1 is really a
# metacarpal must be read from the rest offsets when a retarget is built.
for side in ("Left", "Right"):
    s = side.lower()
    VRM[f"{side}HandThumb1"] = f"{s}ThumbMetacarpal"
    VRM[f"{side}HandThumb2"] = f"{s}ThumbProximal"
    VRM[f"{side}HandThumb3"] = f"{s}ThumbDistal"
    VRM[f"{side}HandThumbEnd"] = None
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        f = finger if finger != "Pinky" else "Little"
        VRM[f"{side}Hand{finger}1"] = f"fold:{s}Hand"
        VRM[f"{side}Hand{finger}2"] = f"{s}{f}Proximal"
        VRM[f"{side}Hand{finger}3"] = f"{s}{f}Intermediate"
        VRM[f"{side}Hand{finger}4"] = f"{s}{f}Distal"
        VRM[f"{side}Hand{finger}End"] = None

NOTES = {
    "Neck2": "VRM has one neck bone: compose Neck1*Neck2 onto neck",
    "Jaw": "VRM 1.0 optional bone; lip sync owns the mouth in the app, track dropped",
    "LeftEye": "VRM optional bone; the gaze owner drives eyes in the app, track dropped",
    "RightEye": "VRM optional bone; the gaze owner drives eyes in the app, track dropped",
}


def main():
    rows = []
    predicted = set(SOMA30)
    for name, parent in SOMA77:
        target = VRM[name]
        kind = "none" if target is None else ("folded" if target.startswith("fold:") else "bone")
        note = NOTES.get(name)
        if name.endswith("End") and note is None:
            note = "end marker, no VRM bone"
        if name not in predicted and not name.endswith("End"):
            note = (note + "; " if note else "") + "not predicted: constant relaxed-hand rest pose"
        rows.append({"soma": name, "parent": parent, "vrm": target, "kind": kind,
                     "predicted": name in predicted, "note": note})
    out = os.path.join(os.getcwd(), "vrm_map.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    bones = sum(r["kind"] == "bone" for r in rows)
    folded = sum(r["kind"] == "folded" for r in rows)
    none = sum(r["kind"] == "none" for r in rows)
    print(f"{len(rows)} SOMA joints: {bones} map to a VRM bone, {folded} fold into a parent bone, "
          f"{none} have no VRM bone; {len(predicted)} are predicted by the model")
    print("\n| SOMA joint | VRM bone | Note |\n| --- | --- | --- |")
    for r in rows:
        if r["kind"] == "bone" and r["predicted"] and r["note"] is None:
            continue
        if r["kind"] == "none" and r["note"] == "end marker, no VRM bone":
            continue
        print(f"| {r['soma']} | {r['vrm'] or '-'} | {r['note'] or ''} |")
    ends = [r["soma"] for r in rows if r["kind"] == "none" and r["note"] == "end marker, no VRM bone"]
    print(f"\nEnd markers without a VRM bone ({len(ends)}): {', '.join(ends)}")
    if "--all" in sys.argv:
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()
