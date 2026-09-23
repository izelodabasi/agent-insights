from pathlib import Path

from agent_insights.repos import repo_name


def test_subfolder_resolves_to_repo(tmp_path: Path):
    (tmp_path / "app" / ".git").mkdir(parents=True)
    (tmp_path / "app" / "Processing" / "deep").mkdir(parents=True)
    assert repo_name(str(tmp_path / "app" / "Processing" / "deep")) == "app"


def test_worktree_resolves_to_main_repo(tmp_path: Path):
    (tmp_path / "app" / ".git" / "worktrees" / "app-develop").mkdir(parents=True)
    wt = tmp_path / "app-develop"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {tmp_path}/app/.git/worktrees/app-develop\n")
    assert repo_name(str(wt / "src")) == "app"


def test_missing_folder_falls_back_to_name(tmp_path: Path):
    assert repo_name(str(tmp_path / "gone" / "shared")) == "shared"
