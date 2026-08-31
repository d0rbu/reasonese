"""Print the four experimental axes."""

from __future__ import annotations

import json

from beartype import beartype

from reasonese.axes import axis_manifest


@beartype
def main() -> None:
    """Print the axis values as JSON."""
    print(json.dumps(axis_manifest(), indent=2))
