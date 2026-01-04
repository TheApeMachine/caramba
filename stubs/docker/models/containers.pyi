from typing import Protocol, Tuple

class ExecResult:
    exit_code: int
    output: Tuple[bytes | None, bytes | None] | None

class Container:
    def exec_run(
        self,
        cmd: str | list[str],
        stdout: bool = True,
        stderr: bool = True,
        stdin: bool = False,
        tty: bool = False,
        privileged: bool = False,
        user: str = "",
        detach: bool = False,
        stream: bool = False,
        socket: bool = False,
        environment: object | None = None,
        workdir: str | None = None,
        demux: bool = False,
    ) -> ExecResult: ...

    def put_archive(self, path: str, data: bytes) -> bool: ...
    def remove(self, force: bool = False) -> None: ...

class ContainerCollection:
    def run(
        self,
        image: str,
        command: object = ...,
        detach: bool = ...,
        tty: bool = ...,
        stdin_open: bool = ...,
        working_dir: str = ...,
        name: str = ...,
    ) -> Container: ...

