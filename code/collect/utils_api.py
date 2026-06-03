"""Shared utilities for the longitudinal metric collection pipeline."""

import asyncio
import json
import re
import time
from functools import wraps
from pathlib import Path

CACHE_DIR = Path("data/collected/cache")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path: Path, completed: set) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(list(completed), f)


def load_checkpoint(path: Path) -> set:
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    return set()


# ── Async retry decorator ─────────────────────────────────────────────────────

def retry_async(max_retries: int = 4, base_delay: float = 1.0):
    """Exponential backoff on HTTP 500/503; re-raise on other errors."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    msg = str(exc)
                    retriable = any(code in msg for code in ("500", "503", "502", "429"))
                    if not retriable or attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(base_delay * (2 ** attempt))
        return wrapper
    return decorator


# ── Pin / float detection per ecosystem ──────────────────────────────────────

_NPM_RANGE_RE = re.compile(r'[\^~><*|]| - ')
_NPM_VERSION_RE = re.compile(r'^\d+(\.\d+)*(-[\w.]+)?(\+[\w.]+)?$')

def is_pinned_npm(requirement: str) -> bool:
    """True if the npm version requirement is an exact version (no range operators)."""
    req = requirement.strip()
    if not req or req in ('*', 'latest', 'x'):
        return False
    # Bare version like "1.2.3" with no operators
    return bool(_NPM_VERSION_RE.match(req)) and not _NPM_RANGE_RE.search(req)


def is_pinned_pypi(version_specifier: str) -> bool:
    """True if the PyPI specifier is exact equality (==X.Y.Z with no wildcard)."""
    spec = version_specifier.strip()
    if not spec:
        return False
    # Must be purely "==X.Y.Z", no comma (combined specifiers count as float)
    return spec.startswith('==') and '*' not in spec and ',' not in spec


_MAVEN_RANGE_RE = re.compile(r'[,()\[\]]')
_MAVEN_DYNAMIC_RE = re.compile(r'(\$\{|LATEST|RELEASE|\+)')

def is_pinned_maven(version: str) -> bool:
    """True if the Maven version is a plain exact version string."""
    v = version.strip()
    if not v:
        return False
    if _MAVEN_DYNAMIC_RE.search(v):
        return False
    # Range notation uses [ ] ( ) and commas
    if '[' in v and ',' not in v:
        # [1.2.3] is an exact range requirement → pinned
        return True
    if _MAVEN_RANGE_RE.search(v):
        return False
    return True


def is_pinned_go(version: str) -> bool:
    """Go module versions are always exact (vX.Y.Z) → always pinned."""
    return bool(version.strip())


def is_pinned_cargo(requirement: str) -> bool:
    """True if the Cargo requirement is an exact version (=X.Y.Z)."""
    req = requirement.strip()
    if not req:
        return False
    # Cargo exact: "=1.2.3" (single =, not >=)
    if req.startswith('=') and not req.startswith('>=') and not req.startswith('<='):
        return True
    # Bare version without operator also counts as caret-range in Cargo → float
    return False


def count_pins_floats(ecosystem: str, requirements_payload: dict) -> tuple[int, int]:
    """
    Given the parsed JSON response from deps.dev GetRequirements,
    return (pins, floats) for the package's dependencies.
    """
    eco = ecosystem.lower()
    pins = floats = 0

    if eco == 'npm':
        deps = requirements_payload.get('npm', {}).get('dependencies', [])
        for dep in deps:
            req = dep.get('requirement', '')
            if is_pinned_npm(req):
                pins += 1
            else:
                floats += 1

    elif eco == 'pypi':
        deps = requirements_payload.get('pypi', {}).get('dependencies', [])
        for dep in deps:
            spec = dep.get('versionSpecifier', '')
            env_marker = dep.get('environmentMarker', '')
            # Skip optional extras deps (they apply conditionally)
            if is_pinned_pypi(spec):
                pins += 1
            else:
                floats += 1

    elif eco == 'maven':
        deps = requirements_payload.get('maven', {}).get('dependencies', [])
        for dep in deps:
            ver = dep.get('version', '')
            if is_pinned_maven(ver):
                pins += 1
            else:
                floats += 1

    elif eco == 'go':
        direct = requirements_payload.get('go', {}).get('directDependencies', [])
        for dep in direct:
            ver = dep.get('version', '')
            if is_pinned_go(ver):
                pins += 1
            else:
                floats += 1

    elif eco == 'cargo':
        deps = requirements_payload.get('cargo', {}).get('dependencies', [])
        for dep in deps:
            req = dep.get('requirement', dep.get('versionReq', ''))
            if is_pinned_cargo(req):
                pins += 1
            else:
                floats += 1

    else:
        # Unknown ecosystem — count any non-empty requirement as float
        for key in requirements_payload:
            for dep in requirements_payload[key].get('dependencies', []):
                floats += 1

    return pins, floats


# ── System name normalization ─────────────────────────────────────────────────

ECOSYSTEM_TO_DEPSDEV = {
    'npm':   ('NPM',   'npm'),
    'pypi':  ('PYPI',  'pypi'),
    'maven': ('MAVEN', 'maven'),
    'go':    ('GO',    'go'),
    'cargo': ('CARGO', 'cargo'),
    'nuget': ('NUGET', 'nuget'),
}

def system_names(ecosystem: str) -> tuple[str, str]:
    """Return (UPPER, lower) system name for deps.dev API calls."""
    return ECOSYSTEM_TO_DEPSDEV.get(ecosystem.lower(), (ecosystem.upper(), ecosystem.lower()))
