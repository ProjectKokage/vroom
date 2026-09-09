"""Project-specific exceptions."""


class VroomError(Exception):
    """Base class for errors that should be presented cleanly by the CLI."""


class GlbError(VroomError):
    """A GLB container or accessor is malformed."""


class VrmError(VroomError):
    """A file does not contain usable VRM humanoid metadata."""


class VrmaError(VroomError):
    """A VRM Animation document or motion clip is invalid."""


class HyMotionError(VroomError):
    """HY-Motion could not be configured or executed."""


class ArdyError(VroomError):
    """ARDY input, configuration, or generation failed."""


class DeviceError(HyMotionError):
    """The requested inference device is unavailable."""
