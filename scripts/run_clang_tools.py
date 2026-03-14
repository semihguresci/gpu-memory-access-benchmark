#!/usr/bin/env python3
"""Run clang-format and clang-tidy for project C++ sources."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HEADER_EXTENSIONS = {".h", ".hh", ".hpp", ".hxx"}
SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx"}
DEFAULT_BUILD_DIR = Path("build-clang-ninja-make")


def discover_cpp_files(repo_root: Path) -> tuple[list[Path], list[Path]]:
    format_files: list[Path] = []
    tidy_files: list[Path] = []

    for relative_dir in ("include", "src"):
        directory = repo_root / relative_dir
        if not directory.exists():
            continue

        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            if path.suffix in HEADER_EXTENSIONS or path.suffix in SOURCE_EXTENSIONS:
                format_files.append(path)
            if path.suffix in SOURCE_EXTENSIONS:
                tidy_files.append(path)

    return format_files, tidy_files


def run_command(command: list[str], cwd: Path) -> bool:
    print(f"[cmd] {' '.join(command)}")
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)

    return completed.returncode == 0


def run_clang_format(repo_root: Path, files: list[Path], check_only: bool) -> bool:
    if not files:
        print("[clang-format] no files found.")
        return True

    print(f"[clang-format] files: {len(files)}")
    ok = True
    for file_path in files:
        command = ["clang-format"]
        if check_only:
            command.extend(["--dry-run", "--Werror"])
        else:
            command.append("-i")
        command.append(str(file_path))
        if not run_command(command, cwd=repo_root):
            ok = False

    return ok


def run_clang_tidy(
    repo_root: Path, files: list[Path], build_dir: Path, config_file: Path, apply_fixes: bool
) -> bool:
    if not files:
        print("[clang-tidy] no files found.")
        return True

    compile_commands = build_dir / "compile_commands.json"
    if not compile_commands.exists():
        print(
            f"[clang-tidy] missing compile database: {compile_commands}\n"
            "Generate it first (example):\n"
            f'  cmake -S . -B {build_dir.name} -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug '
            '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe"',
            file=sys.stderr,
        )
        return False

    print(f"[clang-tidy] files: {len(files)}")
    ok = True
    for file_path in files:
        command = [
            "clang-tidy",
            str(file_path),
            "-p",
            str(build_dir),
            f"--config-file={config_file}",
            "--quiet",
        ]
        if apply_fixes:
            command.insert(1, "-fix")
        if not run_command(command, cwd=repo_root):
            ok = False

    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clang-format and clang-tidy in this repository.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root directory (default: auto-detected from script location).",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="Build directory containing compile_commands.json for clang-tidy.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path(".clang-tidy"),
        help="Path to .clang-tidy configuration file.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run clang-format in check mode (--dry-run --Werror).",
    )
    parser.add_argument(
        "--fix-tidy",
        action="store_true",
        help="Run clang-tidy with -fix.",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Run only clang-format.",
    )
    parser.add_argument(
        "--tidy-only",
        action="store_true",
        help="Run only clang-tidy.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    build_dir = (repo_root / args.build_dir).resolve()
    config_file = (repo_root / args.config_file).resolve()

    format_files, tidy_files = discover_cpp_files(repo_root)

    run_format = not args.tidy_only
    run_tidy = not args.format_only

    format_ok = True
    tidy_ok = True

    if run_format:
        format_ok = run_clang_format(repo_root, format_files, check_only=args.check)

    if run_tidy:
        tidy_ok = run_clang_tidy(
            repo_root=repo_root,
            files=tidy_files,
            build_dir=build_dir,
            config_file=config_file,
            apply_fixes=args.fix_tidy,
        )

    if format_ok and tidy_ok:
        print("[done] clang tooling passed.")
        return 0

    print("[done] clang tooling completed with failures.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
