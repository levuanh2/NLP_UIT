from app.cli.main import configure_utf8_console


class ReconfigurableStream:
    def __init__(self) -> None:
        self.encoding: str | None = None

    def reconfigure(self, *, encoding: str) -> None:
        self.encoding = encoding


def test_configure_utf8_console_updates_supported_streams() -> None:
    stdout = ReconfigurableStream()
    stderr = ReconfigurableStream()

    configure_utf8_console(stdout, stderr)  # type: ignore[arg-type]

    assert stdout.encoding == "utf-8"
    assert stderr.encoding == "utf-8"
