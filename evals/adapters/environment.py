"""Facts about where a run happened (git state, machine, CI)."""

from __future__ import annotations

import os
import platform
import subprocess


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def git_sha() -> str:
    """`EVALS_GIT_SHA` when set (CI passes the PR head commit, not the merge
    commit it checks out), otherwise the current commit."""
    return os.environ.get("EVALS_GIT_SHA") or _git("rev-parse", "HEAD")


def git_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def environment() -> dict[str, str]:
    return {
        "ci": "true" if os.environ.get("CI") == "true" else "false",
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "python": platform.python_version(),
    }
