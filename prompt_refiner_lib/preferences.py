"""Persist an explicit allowlist of UI preferences without saving prompt text or secrets."""

import json
import math
import os
from pathlib import Path
import tempfile
import threading


class SettingsStore:
    """Validate saved controls and atomically replace their JSON file on each change."""

    def __init__(self, path, defaults, choices=None, ranges=None):
        self.path = Path(path)
        self.defaults = defaults
        self.choices = choices or {}
        self.ranges = ranges or {}
        self.values = dict(defaults)
        self.message = ""
        self.lock = threading.Lock()
        try:
            if self.path.exists():
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(saved, dict):
                    raise ValueError("expected a JSON object")
                for key, value in saved.items():
                    if key in defaults and self.valid(key, value):
                        self.values[key] = value
        except (OSError, ValueError) as exc:
            self.message = f"Could not read settings.json; using defaults: {exc}"
            print(f"[Prompt Refiner] {self.message}", flush=True)

    def valid(self, key, value):
        """Reject stale choices, wrong types, non-finite numbers, and invalid ranges."""
        if key not in self.defaults:
            return False
        if key in self.choices:
            return (value is None or isinstance(value, str)) and value in self.choices[
                key
            ]
        default = self.defaults[key]
        if isinstance(default, bool):
            return isinstance(value, bool)
        if isinstance(default, (int, float)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return False
            if isinstance(default, int) and int(value) != value:
                return False
            lower, upper = self.ranges.get(key, (-math.inf, math.inf))
            return lower <= value <= upper
        return isinstance(value, str)

    def update(self, key, value):
        """Save only the changed allowed preference; return nonfatal error text."""
        if not self.valid(key, value):
            return "This setting could not be saved because its value is invalid."
        temporary = None
        try:
            with self.lock:
                updated = {**self.values, key: value}
                with tempfile.NamedTemporaryFile(
                    dir=self.path.parent,
                    prefix=self.path.name + ".",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(
                        json.dumps(
                            updated, indent=2, ensure_ascii=False, allow_nan=False
                        ).encode("utf-8")
                    )
                os.replace(temporary, self.path)
                self.values = updated
            return ""
        except (OSError, ValueError) as exc:
            message = f"Could not save settings.json: {exc}. Current controls still work; check folder write permissions."
            print(f"[Prompt Refiner] {message}", flush=True)
            return message
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
