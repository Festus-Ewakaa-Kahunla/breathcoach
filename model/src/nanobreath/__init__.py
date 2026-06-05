"""nanobreath — real-time joint VAD + breath-event detection for singing voice.

The top-level package is intentionally light: it does not eagerly import the
torch-dependent model code, so utility scripts (data loaders, the baseline,
evaluation) can be used without a torch install. Import the model directly:

    from nanobreath.model.breath_head import BreathHead
    from nanobreath.model.joint import JointModel, load_backbone_frozen
"""

from nanobreath._version import __version__

__all__ = ["__version__"]
