from functools import cache
from pathlib import Path


@cache
def repo_name(cwd: str) -> str:
    """Name of the git repo containing cwd; a worktree resolves to its main repo.

    Falls back to the folder name when cwd is outside any repo or no longer exists.
    """
    if not cwd:
        return ""

    start = Path(cwd)
    for folder in (start, *start.parents):
        git = folder / ".git"
        if git.is_dir():
            return folder.name
        if git.is_file():
            gitdir = git.read_text().removeprefix("gitdir:").strip()
            main = Path(gitdir).parent.parent.parent if "/worktrees/" in gitdir else None
            return main.name if main else folder.name

    return start.name
