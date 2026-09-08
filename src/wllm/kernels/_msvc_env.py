"""Auto-bootstraps the MSVC build environment (PATH/INCLUDE/LIB for cl.exe)
so building the custom CUDA kernels works from an ordinary terminal, not
just a "Developer Command Prompt for VS" or after manually running
vcvarsall.bat. This is the actual portability gap on Windows -- there are no
hardcoded paths anywhere in this project (every path is resolved relative to
__file__ or via torch's own CUDA_HOME auto-detection), but MSVC's compiler
genuinely isn't on PATH by default on any machine, including the one this
project was developed on. Every path used here (vswhere.exe, vcvarsall.bat)
is Visual Studio's own well-known, version-independent install location, so
this works unmodified on any machine with VS + the CUDA Toolkit installed
the normal way.

Import and call ensure_msvc_on_path() before the first
torch.utils.cpp_extension.load() call -- kernels/paged_attention.py and
scripts/check_toolchain.py both do this automatically, so nothing else in
this codebase needs to know this exists.
"""
from __future__ import annotations

import os
import shutil
import subprocess

_VSWHERE = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
_done = False


def ensure_msvc_on_path() -> None:
    """No-op if cl.exe is already reachable (e.g. already running inside a
    Developer Command Prompt, or the user set PATH up themselves) -- only
    does the vcvarsall.bat dance when it's actually needed, and only once
    per process.
    """
    global _done
    if _done or shutil.which("cl") is not None:
        _done = True
        return

    vcvarsall = _find_vcvarsall()
    if vcvarsall is None:
        _done = True  # nothing more we can do -- let the real compiler error surface
        return

    env_text = subprocess.run(
        f'"{vcvarsall}" x64 && set',
        shell=True, capture_output=True, text=True, check=False,
    ).stdout
    for line in env_text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key] = value

    _done = True


def _find_vcvarsall() -> str | None:
    if not os.path.exists(_VSWHERE):
        return None
    result = subprocess.run(
        [_VSWHERE, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, text=True, check=False,
    )
    install_path = result.stdout.strip()
    if not install_path:
        return None
    candidate = os.path.join(install_path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
    return candidate if os.path.exists(candidate) else None
