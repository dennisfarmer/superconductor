"""Thin CLI wrappers for process orchestration scripts."""

from pathlib import Path
import subprocess
import click


@click.command()
@click.option("--start", "start", is_flag=True)
@click.option("--stop", "stop", is_flag=True)
@click.option("--status", "status", is_flag=True)
@click.option("--restart", "restart", is_flag=True)
def main(start: bool, stop: bool, status: bool, restart: bool):
    """Python wrapper for bin/superconductor"""

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "bin" / "superconductor"
    if not script.exists():
        raise SystemExit(f"Missing script: {script}")


    command = ""
    if start:
        command = "start"
    elif stop:
        command = "stop"
    elif status:
        command = "status"
    elif restart:
        command = "restart"
    else:
        command = "start"

    result = subprocess.run([str(script), command], cwd=str(repo_root), check=False)
    raise SystemExit(result.returncode)

if __name__ == "__main__":
    main()