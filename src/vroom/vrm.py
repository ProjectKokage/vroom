"""Purpose-built VRM humanoid metadata reader for VRM 0.x and VRM 1.0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import VrmError
from .glb import Glb

REQUIRED_HUMANOID_BONES = frozenset(
    {
        "hips",
        "spine",
        "head",
        "leftUpperLeg",
        "leftLowerLeg",
        "leftFoot",
        "rightUpperLeg",
        "rightLowerLeg",
        "rightFoot",
        "leftUpperArm",
        "leftLowerArm",
        "leftHand",
        "rightUpperArm",
        "rightLowerArm",
        "rightHand",
    }
)

HUMANOID_PARENTS: dict[str, str | None] = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
    "upperChest": "chest",
    "neck": "upperChest",
    "head": "neck",
    "leftEye": "head",
    "rightEye": "head",
    "jaw": "head",
    "leftUpperLeg": "hips",
    "leftLowerLeg": "leftUpperLeg",
    "leftFoot": "leftLowerLeg",
    "leftToes": "leftFoot",
    "rightUpperLeg": "hips",
    "rightLowerLeg": "rightUpperLeg",
    "rightFoot": "rightLowerLeg",
    "rightToes": "rightFoot",
    "leftShoulder": "upperChest",
    "leftUpperArm": "leftShoulder",
    "leftLowerArm": "leftUpperArm",
    "leftHand": "leftLowerArm",
    "rightShoulder": "upperChest",
    "rightUpperArm": "rightShoulder",
    "rightLowerArm": "rightUpperArm",
    "rightHand": "rightLowerArm",
}
for side in ("left", "right"):
    HUMANOID_PARENTS.update(
        {
            f"{side}ThumbMetacarpal": f"{side}Hand",
            f"{side}ThumbProximal": f"{side}ThumbMetacarpal",
            f"{side}ThumbDistal": f"{side}ThumbProximal",
        }
    )
    for finger in ("Index", "Middle", "Ring", "Little"):
        HUMANOID_PARENTS.update(
            {
                f"{side}{finger}Proximal": f"{side}Hand",
                f"{side}{finger}Intermediate": f"{side}{finger}Proximal",
                f"{side}{finger}Distal": f"{side}{finger}Intermediate",
            }
        )

REQUIRED_OPTIONAL_PARENTS = frozenset(
    {
        "upperChest",
        *(
            f"{side}{finger}{segment}"
            for side in ("left", "right")
            for finger, segment in (
                ("Thumb", "Proximal"),
                ("Thumb", "Distal"),
                ("Index", "Intermediate"),
                ("Index", "Distal"),
                ("Middle", "Intermediate"),
                ("Middle", "Distal"),
                ("Ring", "Intermediate"),
                ("Ring", "Distal"),
                ("Little", "Intermediate"),
                ("Little", "Distal"),
            )
        ),
    }
)

_VRM0_THUMB_NAMES = {
    "leftThumbProximal": "leftThumbMetacarpal",
    "leftThumbIntermediate": "leftThumbProximal",
    "rightThumbProximal": "rightThumbMetacarpal",
    "rightThumbIntermediate": "rightThumbProximal",
}


@dataclass(frozen=True, slots=True)
class VrmAvatar:
    """The humanoid portion of a VRM GLB."""

    glb: Glb
    spec_version: str
    humanoid_bones: dict[str, int]

    @classmethod
    def read(cls, path: str | Path) -> VrmAvatar:
        return cls.from_glb(Glb.read(path))

    @classmethod
    def from_glb(cls, glb: Glb) -> VrmAvatar:
        extensions = glb.document.get("extensions", {})
        if not isinstance(extensions, dict):
            raise VrmError("glTF extensions must be an object")

        if "VRMC_vrm" in extensions:
            extension = extensions["VRMC_vrm"]
            if not isinstance(extension, dict):
                raise VrmError("VRMC_vrm extension must be an object")
            spec_version = str(extension.get("specVersion", "1.0"))
            humanoid = extension.get("humanoid", {})
            if not isinstance(humanoid, dict):
                raise VrmError("VRM 1.0 humanoid must be an object")
            raw_bones = humanoid.get("humanBones", {})
            if not isinstance(raw_bones, dict):
                raise VrmError("VRM 1.0 humanBones must be an object")
            mapping = {
                name: value.get("node")
                for name, value in raw_bones.items()
                if isinstance(value, dict)
            }
        elif "VRM" in extensions:
            extension = extensions["VRM"]
            if not isinstance(extension, dict):
                raise VrmError("legacy VRM extension must be an object")
            spec_version = str(extension.get("specVersion", "0.0"))
            humanoid = extension.get("humanoid", {})
            if not isinstance(humanoid, dict):
                raise VrmError("VRM 0.x humanoid must be an object")
            raw_bones = humanoid.get("humanBones", [])
            if not isinstance(raw_bones, list):
                raise VrmError("VRM 0.x humanBones must be an array")
            mapping = {}
            for value in raw_bones:
                if not isinstance(value, dict) or not isinstance(value.get("bone"), str):
                    continue
                legacy_name = value["bone"]
                mapping[_VRM0_THUMB_NAMES.get(legacy_name, legacy_name)] = value.get("node")
        else:
            raise VrmError("GLB contains neither VRMC_vrm nor legacy VRM metadata")

        nodes = glb.document.get("nodes", [])
        if not isinstance(nodes, list):
            raise VrmError("glTF nodes must be an array")
        clean_mapping: dict[str, int] = {}
        used_nodes: set[int] = set()
        for bone_name, node_index in mapping.items():
            if (
                not isinstance(bone_name, str)
                or not isinstance(node_index, int)
                or isinstance(node_index, bool)
            ):
                raise VrmError(f"humanoid bone {bone_name!r} has an invalid node")
            if not 0 <= node_index < len(nodes):
                raise VrmError(f"humanoid bone {bone_name!r} points to node {node_index}")
            if node_index in used_nodes:
                raise VrmError(f"multiple humanoid bones point to node {node_index}")
            clean_mapping[bone_name] = node_index
            used_nodes.add(node_index)

        missing = REQUIRED_HUMANOID_BONES - clean_mapping.keys()
        if missing:
            raise VrmError(f"VRM humanoid is missing required bones: {', '.join(sorted(missing))}")
        return cls(glb=glb, spec_version=spec_version, humanoid_bones=clean_mapping)

    @property
    def generation(self) -> str:
        return "1.0" if self.spec_version.startswith("1.") else "0.x"

    def node(self, bone_name: str) -> dict[str, Any]:
        try:
            return self.glb.document["nodes"][self.humanoid_bones[bone_name]]
        except KeyError as exc:
            raise VrmError(f"avatar has no {bone_name!r} humanoid bone") from exc

    def node_name(self, bone_name: str) -> str | None:
        name = self.node(bone_name).get("name")
        return name if isinstance(name, str) else None

    def summary(self) -> dict[str, Any]:
        return {
            "format": "VRM",
            "generation": self.generation,
            "specVersion": self.spec_version,
            "nodes": len(self.glb.document.get("nodes", [])),
            "humanoidBones": len(self.humanoid_bones),
            "requiredBonesPresent": True,
        }
