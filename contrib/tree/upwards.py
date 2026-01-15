# contrib/tree/upwards.py
from __future__ import annotations

from sys import stdout
from io import StringIO
from typing import Generic, Iterable, TypeVar
from shutil import get_terminal_size
from time import sleep

T = TypeVar("T")

EOL_NORMAL = "\r\n"
EOL_UPWARDS = "\r\x1bM"


class PrintableUpwards(list[T], Generic[T]):
    """Lines with rendering and print direction options for interactive terminal sessions
    - upwards=False: join with CRLF.
    - upwards=True:  join with CR+UP, and ALWAYS wrap the entire output with enough LF BEFORE and AFTER.
    The cursor always finally lands below the text.
    """

    def __init__(self, iterable: Iterable[T] = (), *, reverse: bool = False, upwards: bool = False) -> None:
        super().__init__(iterable)
        self.upwards = upwards
        self.reverse = reverse

    def __str__(self) -> str:
        if not self:
            return ""

        top_to_bottom_order = list(reversed(self)) if self.reverse else self
        if not self.upwards:
            return EOL_NORMAL.join(map(str, top_to_bottom_order)) + EOL_NORMAL

        terminal_rows = get_terminal_size(fallback=(80, 24)).lines
        reserve_n = min(len(self) - 1, max(terminal_rows - 1, 0))
        reserve = "\n" * reserve_n
        bottom_to_top_order = self if self.reverse else list(reversed(self))
        scroll_up = EOL_NORMAL + EOL_NORMAL.join(top_to_bottom_order[reserve_n + 1:])
        if reserve_n < len(self) - 1:
            scroll_up += EOL_NORMAL
        return reserve + EOL_UPWARDS.join(map(str, bottom_to_top_order)) + reserve + scroll_up

    def __mul__(self, n: int) -> PrintableUpwards[T]:
        return type(self)(super().__mul__(n), reverse=self.reverse, upwards=self.upwards)

    def __rmul__(self, n: int) -> PrintableUpwards[T]:
        return self.__mul__(n)

    def __add__(self, other: Iterable) -> PrintableUpwards[T]:
        if isinstance(other, PrintableUpwards) and other.reverse ^ self.reverse:
            other = list(reversed(other))

        return type(self)(super().__add__(other), reverse=self.reverse, upwards=self.upwards)

    def __radd__(self, other: Iterable) -> PrintableUpwards[T]:
        return self.__add__(other)


def scroll(*args, sep=' ', end='\n', file=stdout):
    buf = StringIO()
    print(*args, sep=sep, end=end, file=buf)
    captured = buf.getvalue()
    lines = captured.split("\n")
    for i, line in enumerate(lines):
        ups = line.split(EOL_UPWARDS)
        for j, up in enumerate(ups):
            file.write(up)
            if j < len(ups) - 1:
                file.write(EOL_UPWARDS)
                sleep(.1)

        if i < len(lines) - 1:
            file.write("\n")
            sleep(.01)

if __name__ == "__main__":
    terminal_lines = get_terminal_size(fallback=(80, 24)).lines
    buf = StringIO()
    full_height_upwards = PrintableUpwards(map(str, range(terminal_lines)), reverse=True, upwards=True)
    empty_upwards = PrintableUpwards(upwards=True)
    normal = PrintableUpwards("Hello world".split())
    reverse = PrintableUpwards("Reverse order".split(), reverse=True)
    upwards = PrintableUpwards("Printed upwards".split(), upwards=True)
    print(empty_upwards, 2 * full_height_upwards, reverse + normal, upwards + reverse, sep="---\n", end="---\n", file=buf)
    captured = buf.getvalue()
    print(repr(captured))
    print(captured)
    scroll(captured)
