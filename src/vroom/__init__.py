"""VRMA generation, curation, inspection, and validation."""

from .curation import (
    HIPS_TRANSLATION_MODES,
    MAX_ROTATION_GAIN,
    ROTATION_PROJECTIONS,
    BoneCurationPolicy,
    CurationOptions,
    HipsTranslationPolicy,
    curate_vrma,
    write_curated_vrma,
)
from .glb import Glb
from .motion import MotionClip, Skeleton, load_hy_motion_skeleton
from .vrm import VrmAvatar
from .vrma import ValidationReport, read_vrma, validate_vrma, write_vrma

__all__ = [
    "HIPS_TRANSLATION_MODES",
    "MAX_ROTATION_GAIN",
    "ROTATION_PROJECTIONS",
    "BoneCurationPolicy",
    "CurationOptions",
    "Glb",
    "HipsTranslationPolicy",
    "MotionClip",
    "Skeleton",
    "ValidationReport",
    "VrmAvatar",
    "curate_vrma",
    "load_hy_motion_skeleton",
    "read_vrma",
    "validate_vrma",
    "write_curated_vrma",
    "write_vrma",
]

__version__ = "0.1.0"
