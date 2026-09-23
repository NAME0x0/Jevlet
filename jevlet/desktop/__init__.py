"""Windows desktop context and explicitly approved actions."""

from .harness import ActionProposal, ActionRequest, ActionResult, DesktopHarness
from .native import ControlInfo, DesktopContext, NativeDesktop, WindowInfo

__all__ = [
    "ActionProposal",
    "ActionRequest",
    "ActionResult",
    "ControlInfo",
    "DesktopContext",
    "DesktopHarness",
    "NativeDesktop",
    "WindowInfo",
]
