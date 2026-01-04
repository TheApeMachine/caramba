"""Docker workspace

Runs all development actions inside a disposable Docker container so agentic
processes can safely clone, edit, test, and open PRs without touching the host
checkout.
"""

from __future__ import annotations

import io
import shlex
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import docker
from docker.errors import APIError, ImageNotFound
from docker.client import DockerClient
from docker.models.containers import Container, ExecResult


@dataclass(frozen=True)
class CommandOutput:
    """Command output

    Captures stdout/stderr/exit code from a single command executed in the workspace
    container, allowing verification and auditing from outside the container.
    """

    command: str
    exit_code: int
    stdout: str
    stderr: str

    def ok(self) -> bool:
        """Success flag

        Used to gate workflow progress on deterministic command results.
        """

        return self.exit_code == 0


class DockerWorkspace:
    """Docker workspace

    A long-lived container that accepts exec commands. The container is created
    without mounting the host repository, so code changes occur only in the
    container filesystem.
    """

    def __init__(self, *, image: str, workdir: str, name_prefix: str) -> None:
        self.image: str = str(image)
        self.workdir: str = str(workdir)
        self.name_prefix: str = str(name_prefix)
        self.client: DockerClient = docker.from_env()
        self.container: Container | None = None

    def start(self) -> None:
        """Start container

        Creates a container that stays alive and can be used for multiple exec calls.
        """

        self.ensureImageExists()
        name = self.name_prefix
        container = self.client.containers.run(
            self.image,
            command=["bash", "-lc", "sleep infinity"],
            detach=True,
            tty=False,
            stdin_open=False,
            working_dir=self.workdir,
            name=name,
        )
        self.container = container
        _ = self.run(["bash", "-lc", f"mkdir -p {self.workdir}"], cwd="/")

    def close(self) -> None:
        """Close container

        Stops and removes the workspace container. This is required to avoid
        leaking writable environments.
        """

        if self.container is None:
            raise RuntimeError("DockerWorkspace.close called before start")
        self.container.remove(force=True)
        self.container = None

    def run(self, command: list[str], *, cwd: str | None = None, timeout_sec: int = 900) -> CommandOutput:
        """Run command

        Executes a command inside the container and returns stdout/stderr and exit code.
        """

        if self.container is None:
            raise RuntimeError("DockerWorkspace.run called before start")
        if int(timeout_sec) <= 0:
            raise ValueError("timeout_sec must be positive")
        workdir = cwd or self.workdir
        shell = " ".join(shlex.quote(x) for x in command)
        wrapped = ["bash", "-lc", f"timeout {int(timeout_sec)}s {shell}"]
        exec_result: ExecResult = self.container.exec_run(wrapped, workdir=workdir, demux=True, tty=False)
        output = cast(tuple[bytes | None, bytes | None] | None, exec_result.output)
        stdout_bytes, stderr_bytes = output if output is not None else (None, None)
        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
        exit_code = int(exec_result.exit_code or 0)
        joined = " ".join(command)
        return CommandOutput(command=joined, exit_code=exit_code, stdout=stdout, stderr=stderr)

    def writeText(self, *, path: str, text: str) -> None:
        """Write file

        Writes a text file inside the container using a tar archive upload.
        """

        if self.container is None:
            raise RuntimeError("DockerWorkspace.writeText called before start")
        target = Path(path)
        if not target.is_absolute():
            raise ValueError(f"path must be absolute inside container, got {path!r}")
        parent = str(target.parent)
        _ = self.run(["bash", "-lc", f"mkdir -p {parent}"], cwd="/")
        payload = self.buildTarForTextFile(file_path=target.name, text=text)
        ok = bool(self.container.put_archive(parent, payload))
        if not ok:
            raise RuntimeError(f"Failed to write file into container at {path}")

    def readText(self, *, path: str, max_chars: int) -> str:
        """Read file

        Reads a file from the container and truncates to a maximum character budget.
        """

        out = self.run(["bash", "-lc", f"python3 -c \"print(open('{path}','r',encoding='utf-8').read())\""])
        if not out.ok():
            raise RuntimeError(
                f"Failed to read file in container: {path}\nstdout:\n{out.stdout}\nstderr:\n{out.stderr}"
            )
        content = out.stdout
        if max_chars > 0 and len(content) > max_chars:
            return content[:max_chars] + "\n\n[...truncated...]\n"
        return content

    def ensureImageExists(self) -> None:
        """Image validation

        Ensures the configured image exists locally. This avoids implicit pulls and
        makes execution deterministic.
        """

        try:
            _image = self.client.images.get(self.image)
        except ImageNotFound as e:
            raise RuntimeError(
                f"Docker image not found locally: {self.image}. Build or pull it explicitly."
            ) from e
        except APIError as e:
            raise RuntimeError(f"Failed to query Docker images: {e}") from e

    def buildTarForTextFile(self, *, file_path: str, text: str) -> bytes:
        """Tar builder

        Docker put_archive accepts a tar stream. This creates an in-memory tar
        containing a single UTF-8 text file.
        """

        buf = io.BytesIO()
        data = text.encode("utf-8")
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=file_path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        return buf.getvalue()

    def ensureRepoCloned(self, *, repo_url: str, repo_dir: str, base_branch: str) -> None:
        """Clone repository

        Creates a fresh clone inside the container and checks out the base branch.
        """

        _ = self.run(["bash", "-lc", f"rm -rf {repo_dir}"], cwd=self.workdir)
        clone = self.run(["bash", "-lc", f"git clone {repo_url} {repo_dir}"], cwd=self.workdir)
        if not clone.ok():
            raise RuntimeError(f"git clone failed:\n{clone.stderr}")
        checkout = self.run(["bash", "-lc", f"git checkout {base_branch}"], cwd=f"{self.workdir}/{repo_dir}")
        if not checkout.ok():
            raise RuntimeError(f"git checkout failed:\n{checkout.stderr}")

    def createBranch(self, *, repo_dir: str, branch_name: str) -> None:
        """Create branch

        Creates a new branch for changes inside the container clone.
        """

        out = self.run(["bash", "-lc", f"git checkout -B {branch_name}"], cwd=f"{self.workdir}/{repo_dir}")
        if not out.ok():
            raise RuntimeError(f"git checkout -B failed:\n{out.stderr}")

    def applyPatch(self, *, repo_dir: str, patch_text: str) -> None:
        """Apply patch

        Applies a git diff patch inside the container clone.
        """

        patch_path = "/tmp/agent.patch"
        self.writeText(path=patch_path, text=patch_text)
        out = self.run(["bash", "-lc", f"git apply {patch_path}"], cwd=f"{self.workdir}/{repo_dir}")
        if not out.ok():
            raise RuntimeError(f"git apply failed:\n{out.stderr}\n{out.stdout}")

    def diff(self, *, repo_dir: str) -> str:
        """Diff

        Returns git diff from inside the clone.
        """

        out = self.run(["bash", "-lc", "git diff"], cwd=f"{self.workdir}/{repo_dir}")
        if not out.ok():
            raise RuntimeError(f"git diff failed:\n{out.stderr}")
        return out.stdout

    def commitAll(self, *, repo_dir: str, message: str) -> None:
        """Commit changes

        Stages all changes and commits them with the provided message.
        """

        add = self.run(["bash", "-lc", "git add -A"], cwd=f"{self.workdir}/{repo_dir}")
        if not add.ok():
            raise RuntimeError(f"git add failed:\n{add.stderr}")
        commit = self.run(["bash", "-lc", f"git commit -m {message!r}"], cwd=f"{self.workdir}/{repo_dir}")
        if not commit.ok():
            raise RuntimeError(f"git commit failed:\n{commit.stderr}\n{commit.stdout}")

    def pushBranch(self, *, repo_dir: str, remote: str, branch_name: str) -> None:
        """Push branch

        Pushes the current branch to the configured remote.
        """

        out = self.run(["bash", "-lc", f"git push -u {remote} {branch_name}"], cwd=f"{self.workdir}/{repo_dir}")
        if not out.ok():
            raise RuntimeError(f"git push failed:\n{out.stderr}\n{out.stdout}")

    def ghAuthWithToken(self, *, token_text: str) -> None:
        """Authenticate GitHub CLI

        Logs in using a token passed through a file, avoiding environment variables.
        """

        token_path = "/tmp/gh_token"
        self.writeText(path=token_path, text=token_text.strip() + "\n")
        out = self.run(["bash", "-lc", f"gh auth login --with-token < {token_path}"], cwd="/")
        if not out.ok():
            raise RuntimeError(f"gh auth login failed:\n{out.stderr}\n{out.stdout}")

    def ghCreatePr(self, *, repo_dir: str, base_branch: str, title: str, body: str) -> str:
        """Create PR

        Creates a GitHub pull request and returns the resulting URL.
        """

        self.writeText(path="/tmp/pr_body.md", text=body)
        out = self.run(
            [
                "bash",
                "-lc",
                f"gh pr create --base {base_branch} --title {title!r} --body-file /tmp/pr_body.md",
            ],
            cwd=f"{self.workdir}/{repo_dir}",
        )
        if not out.ok():
            raise RuntimeError(f"gh pr create failed:\n{out.stderr}\n{out.stdout}")
        return out.stdout.strip()

