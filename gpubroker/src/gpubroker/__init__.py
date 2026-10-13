"""Shared GPU lease broker for wingpu and its sibling bridges.

The WSL-side program is ``broker.py`` (stdlib only). wingpu uploads that file
to the GPU host. Siblings import ``client``, ``ssh``, ``lease``, and ``config``
and do not install or upgrade the remote broker.
"""

from .broker import PROTOCOL, VERSION

__version__ = VERSION

__all__ = ["PROTOCOL", "VERSION", "__version__"]
