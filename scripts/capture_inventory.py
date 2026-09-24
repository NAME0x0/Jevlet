"""Record the controls of every open window as grounding data (read-only, local).

Open the apps you use, then run:  python -m scripts.capture_inventory
Run it again later with other apps open; inventories merge per process.
"""

from __future__ import annotations

import json

from jevlet.grounding_live import INVENTORIES, capture, inventory_apps


def main() -> None:
    counts = capture()
    apps = inventory_apps()
    print(
        json.dumps(
            {
                "captured_controls": counts,
                "usable_apps": {app.name: len(app.tasks) for app in apps},
                "stored_in": str(INVENTORIES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
