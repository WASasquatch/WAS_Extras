import importlib
import pkgutil
import time
import traceback

# Rich (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
    from rich.traceback import install as rich_traceback_install
    from rich.theme import Theme
    THEME = Theme({
        "was.blue": "#7cc7ff",
        "was.orange": "orange1",
        "was.muted": "grey70",
        "was.good": "#7cc7ff",
        "was.bad": "orange1",
        "rule.line": "#7cc7ff",
    })
    console = Console(theme=THEME)
    rich_traceback_install(show_locals=False, word_wrap=True)
except Exception:
    console = None

PREFIX = "[WAS Extras] "

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class NodeLoader:
    def __init__(self, package_name: str, prefix: str = PREFIX):
        self.package_name = package_name
        self.prefix = prefix
        self.timings: dict[str, tuple[float, bool, Exception | None]] = {}

    def module_path(self, module) -> str:
        spec = getattr(module, "__spec__", None)
        if spec and getattr(spec, "origin", None):
            return spec.origin
        return getattr(module, "__file__", repr(module))

    def record(self, module, elapsed: float, ok: bool, err: Exception | None) -> None:
        self.timings[self.module_path(module)] = (elapsed, ok, err)
        if ok:
            NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
            NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

    def import_module(self, fullname: str, package: str | None = None) -> tuple[object | None, bool]:
        t0 = time.time()
        ok = True
        err = None
        mod = None
        try:
            mod = importlib.import_module(fullname, package=package)
        except Exception as e:
            ok = False
            err = e
            if console is None:
                traceback.print_exc()
            else:
                console.print_exception()
        elapsed = time.time() - t0
        if mod is not None:
            self.record(mod, elapsed, ok, err)
        return mod, ok

    def print_intro(self) -> None:
        msg = (
            "[was.muted]Nodes in this repo solve specific problems and may not fit every workflow.[/was.muted]"
        )
        if console:
            console.rule(f"[was.blue]{self.prefix.strip()}[/was.blue]", style="rule.line")
            console.print(Panel(msg, title=f"[was.orange]{self.prefix.strip()}[/was.orange]", border_style="was.blue"))
        else:
            print(f"{self.prefix}Nodes in this repo solve specific problems and may not fit every workflow.")

    def print_no_nodes_pkg(self) -> None:
        msg = "No ./nodes package found or import failed."
        if console:
            console.print(f"[was.orange]{self.prefix}{msg}[/was.orange]")
        else:
            print(f"{self.prefix}{msg}")

    def print_summary(self) -> None:
        total = len(self.timings)
        ok_count = sum(1 for _, (_, ok, _) in self.timings.items() if ok)
        fail_count = total - ok_count
        if console:
            table = Table(
                title=f"[was.blue]{self.prefix}Import Summary[/was.blue]",
                expand=False,
                header_style="was.blue",
                border_style="was.blue",
            )
            table.add_column("Module/File", overflow="fold")
            table.add_column("Time (s)", justify="right")
            table.add_column("Status", justify="center")
            table.add_column("Error", overflow="fold")
            for path, (timing, success, err) in self.timings.items():
                status = "[was.good]OK[/was.good]" if success else "[was.bad]FAILED[/was.bad]"
                err_text = "" if err is None else f"{type(err).__name__}: {err}"
                table.add_row(str(path), f"{timing:.2f}", status, err_text)
            console.print(table)
            console.rule(style="rule.line")
            console.print(
                f"[was.blue]{self.prefix}Totals:[/was.blue] "
                f"[was.good]{ok_count} ok[/was.good], [was.bad]{fail_count} failed[/was.bad], {total} modules."
            )
            console.rule(style="rule.line")
        else:
            print(f"{self.prefix} Import times:")
            for path, (timing, success, err) in self.timings.items():
                print(f"   {timing:.1f} seconds{('' if success else ' (IMPORT FAILED)')}: {path}")
                if err:
                    print("Error:", err)
            print(f"{self.prefix}Totals: {total} modules, {ok_count} ok, {fail_count} failed.")

    def load_all(self) -> None:
        self.print_intro()
        nodes_pkg, ok = self.import_module(".nodes", package=self.package_name)
        if not ok or nodes_pkg is None:
            self.print_no_nodes_pkg()
            self.print_summary()
            return
        if console:
            with Progress(
                SpinnerColumn(style="was.blue"),
                TextColumn("[progress.description]{task.description}", style="was.muted"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"[was.orange]{self.prefix}Loading nodes from ./nodes ...[/was.orange]", total=None)
                for _, name, _ in pkgutil.walk_packages(nodes_pkg.__path__, prefix=nodes_pkg.__name__ + "."):
                    self.import_module(name)
                progress.remove_task(task)
        else:
            print(f"{self.prefix}Loading nodes from ./nodes ...")
            for _, name, _ in pkgutil.walk_packages(nodes_pkg.__path__, prefix=nodes_pkg.__name__ + "."):
                self.import_module(name)
        self.print_summary()


_loader = NodeLoader(package_name=__name__, prefix=PREFIX)
_loader.load_all()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
