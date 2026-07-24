from __future__ import annotations

from vroom.glb import Glb
from vroom.vrm import REQUIRED_HUMANOID_BONES, VrmAvatar


def test_vrm1_humanoid_metadata_reader() -> None:
    bone_names = sorted(REQUIRED_HUMANOID_BONES)
    document = {
        "asset": {"version": "2.0"},
        "nodes": [{"name": f"Avatar_{name}"} for name in bone_names],
        "extensions": {
            "VRMC_vrm": {
                "specVersion": "1.0",
                "humanoid": {
                    "humanBones": {name: {"node": index} for index, name in enumerate(bone_names)}
                },
            }
        },
    }

    avatar = VrmAvatar.from_glb(Glb(document))

    assert avatar.generation == "1.0"
    assert avatar.humanoid_bones.keys() == set(bone_names)
    assert avatar.node_name("hips") == "Avatar_hips"
