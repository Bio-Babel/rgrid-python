"""biobabel manifest factory for grid_py (rgrid-python).

`get_manifest()` is the `biobabel.manifest` entry point declared in pyproject.toml.

The manifest body lives in YAML files alongside this module so non-Python
maintainers can read and edit it directly. This factory just walks the
directory and hands the merged dict to Pydantic.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from biobabel.manifest_api import PackageManifest

_HERE = Path(__file__).parent


def get_manifest() -> PackageManifest:
    package_yaml = yaml.safe_load((_HERE / "package.yaml").read_text(encoding="utf-8")) or {}

    for subdir, field in (
        ("concepts", "concepts"),
        ("idioms", "idioms"),
        ("anti_patterns", "anti_patterns"),
        ("compositions", "compositions"),
    ):
        items: list[dict] = list(package_yaml.get(field, []) or [])
        for yfile in sorted((_HERE / subdir).glob("*.yaml")):
            loaded = yaml.safe_load(yfile.read_text(encoding="utf-8"))
            if loaded is None:
                continue
            if isinstance(loaded, list):
                items.extend(loaded)
            else:
                items.append(loaded)
        if items:
            package_yaml[field] = items

    return PackageManifest.model_validate(package_yaml)
