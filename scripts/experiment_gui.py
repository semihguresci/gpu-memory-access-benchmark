#!/usr/bin/env python3

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import ttk
import tkinter as tk

from experiment_manifest import ROOT
from experiment_workflow import ExperimentWorkflowConfig, load_enabled_experiment_configs, resolve_binary


WINDOW_TITLE = "GPU Experiment Runner"
LOG_POLL_INTERVAL_MS = 100


class ExperimentRunnerGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1560x980")
        self.root.minsize(1280, 760)

        self.configs = load_enabled_experiment_configs()
        self.display_order = sorted(self.configs.values(), key=lambda config: config.id)
        self.status_by_experiment = {config.id: "idle" for config in self.display_order}

        self.binary_var = tk.StringVar(value=self._auto_binary_path())
        self.iterations_var = tk.IntVar(value=5)
        self.warmup_var = tk.IntVar(value=2)
        self.size_override_var = tk.StringVar(value="")
        self.label_var = tk.StringVar(value="")
        self.validation_var = tk.BooleanVar(value=False)
        self.verbose_progress_var = tk.BooleanVar(value=False)
        self.collect_run_var = tk.BooleanVar(value=True)
        self.collect_current_before_artifacts_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")

        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_requested = threading.Event()
        self.process_lock = threading.Lock()
        self.active_process: subprocess.Popen[str] | None = None

        self._build_layout()
        self._populate_tree()
        self._update_selected_details()
        self.root.after(LOG_POLL_INTERVAL_MS, self._drain_event_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(paned, padding=10)
        right = ttk.Frame(paned, padding=10)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        left.rowconfigure(2, weight=0)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        paned.add(left, weight=2)
        paned.add(right, weight=3)

        selection_frame = ttk.LabelFrame(left, text="Experiments", padding=10)
        selection_frame.grid(row=0, column=0, sticky="ew")
        for column_index in range(6):
            selection_frame.columnconfigure(column_index, weight=1 if column_index in (0, 1, 2, 3) else 0)

        ttk.Button(selection_frame, text="Select All", command=self._select_all).grid(row=0, column=0, sticky="ew")
        ttk.Button(selection_frame, text="Clear", command=self._clear_selection).grid(row=0, column=1, sticky="ew")
        ttk.Button(selection_frame, text="Core", command=lambda: self._select_docs_group("core")).grid(
            row=0, column=2, sticky="ew"
        )
        ttk.Button(selection_frame, text="Extension", command=lambda: self._select_docs_group("extension")).grid(
            row=0, column=3, sticky="ew"
        )
        ttk.Button(selection_frame, text="Advanced", command=lambda: self._select_docs_group("advanced")).grid(
            row=0, column=4, sticky="ew"
        )
        ttk.Button(selection_frame, text="Open Results", command=self._open_selected_results).grid(
            row=0, column=5, sticky="ew"
        )

        tree_frame = ttk.Frame(left)
        tree_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 10))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("id", "name", "category", "group", "default_size", "status"),
            show="headings",
            selectmode="extended",
        )
        self.tree.heading("id", text="ID")
        self.tree.heading("name", text="Name")
        self.tree.heading("category", text="Category")
        self.tree.heading("group", text="Group")
        self.tree.heading("default_size", text="Default Size")
        self.tree.heading("status", text="Status")
        self.tree.column("id", width=190, stretch=False)
        self.tree.column("name", width=320, stretch=True)
        self.tree.column("category", width=180, stretch=False)
        self.tree.column("group", width=110, stretch=False)
        self.tree.column("default_size", width=100, stretch=False, anchor=tk.CENTER)
        self.tree.column("status", width=120, stretch=False, anchor=tk.CENTER)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._handle_tree_selection)

        tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

        details_frame = ttk.LabelFrame(left, text="Selection Details", padding=10)
        details_frame.grid(row=2, column=0, sticky="ew")
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)

        self.details_text = tk.Text(details_frame, wrap="word", height=10, state=tk.DISABLED)
        self.details_text.grid(row=0, column=0, sticky="nsew")
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        details_scrollbar.grid(row=0, column=1, sticky="ns")
        self.details_text.configure(yscrollcommand=details_scrollbar.set)

        settings_frame = ttk.LabelFrame(right, text="Run Settings", padding=10)
        settings_frame.grid(row=0, column=0, sticky="ew")
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Benchmark Binary").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(settings_frame, textvariable=self.binary_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(settings_frame, text="Browse", command=self._browse_binary).grid(row=0, column=2, sticky="ew", pady=4)
        ttk.Button(settings_frame, text="Auto", command=self._use_auto_binary).grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(settings_frame, text="Timed Iterations").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Spinbox(settings_frame, from_=1, to=999, textvariable=self.iterations_var, width=10).grid(
            row=1, column=1, sticky="w", pady=4
        )

        ttk.Label(settings_frame, text="Warmup Iterations").grid(row=1, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Spinbox(settings_frame, from_=0, to=999, textvariable=self.warmup_var, width=10).grid(
            row=1, column=3, sticky="w", pady=4
        )

        ttk.Label(settings_frame, text="Size Override").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(settings_frame, textvariable=self.size_override_var).grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(settings_frame, text="Run Label").grid(row=2, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(settings_frame, textvariable=self.label_var).grid(row=2, column=3, sticky="ew", pady=4)

        ttk.Checkbutton(settings_frame, text="Validation Layers", variable=self.validation_var).grid(
            row=3, column=0, sticky="w", pady=4
        )
        ttk.Checkbutton(settings_frame, text="Verbose Progress", variable=self.verbose_progress_var).grid(
            row=3, column=1, sticky="w", pady=4
        )
        ttk.Checkbutton(settings_frame, text="Collect Run After Benchmark", variable=self.collect_run_var).grid(
            row=3, column=2, sticky="w", pady=4
        )
        ttk.Checkbutton(
            settings_frame,
            text="Collect Current benchmark_results.json Before Artifacts",
            variable=self.collect_current_before_artifacts_var,
        ).grid(row=3, column=3, sticky="w", pady=4)

        actions_frame = ttk.LabelFrame(right, text="Actions", padding=10)
        actions_frame.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        for column_index in range(5):
            actions_frame.columnconfigure(column_index, weight=1)

        self.build_button = ttk.Button(actions_frame, text="Build Binary", command=self._start_build)
        self.build_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.run_button = ttk.Button(actions_frame, text="Run Benchmarks", command=self._start_run_only)
        self.run_button.grid(row=0, column=1, sticky="ew", padx=6)

        self.run_artifacts_button = ttk.Button(
            actions_frame, text="Run + Artifacts", command=self._start_run_and_artifacts
        )
        self.run_artifacts_button.grid(row=0, column=2, sticky="ew", padx=6)

        self.artifacts_button = ttk.Button(actions_frame, text="Artifacts Only", command=self._start_artifacts_only)
        self.artifacts_button.grid(row=0, column=3, sticky="ew", padx=6)

        self.stop_button = ttk.Button(actions_frame, text="Stop", command=self._request_stop, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=4, sticky="ew", padx=(6, 0))

        progress_frame = ttk.Frame(right)
        progress_frame.grid(row=2, column=0, sticky="nsew")
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)

        ttk.Label(progress_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.progressbar = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.progressbar.grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=(0, 6))

        log_frame = ttk.LabelFrame(progress_frame, text="Live Log", padding=10)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

    def _populate_tree(self) -> None:
        for config in self.display_order:
            self.tree.insert(
                "",
                tk.END,
                iid=config.id,
                values=(
                    config.id,
                    config.display_name,
                    config.category,
                    config.docs_group,
                    config.default_size,
                    self.status_by_experiment[config.id],
                ),
            )

    def _selected_experiment_ids(self) -> list[str]:
        return list(self.tree.selection())

    def _auto_binary_path(self) -> str:
        try:
            return str(resolve_binary(None))
        except FileNotFoundError:
            return ""

    def _browse_binary(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Select benchmark executable",
            initialdir=str(ROOT),
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
        )
        if selected:
            self.binary_var.set(selected)

    def _use_auto_binary(self) -> None:
        self.binary_var.set(self._auto_binary_path())

    def _select_all(self) -> None:
        self.tree.selection_set([config.id for config in self.display_order])
        self._update_selected_details()

    def _clear_selection(self) -> None:
        self.tree.selection_remove(self.tree.selection())
        self._update_selected_details()

    def _select_docs_group(self, docs_group: str) -> None:
        matching_ids = [config.id for config in self.display_order if config.docs_group == docs_group]
        self.tree.selection_set(matching_ids)
        self._update_selected_details()

    def _handle_tree_selection(self, _event: object) -> None:
        self._update_selected_details()

    def _update_selected_details(self) -> None:
        selected_ids = self._selected_experiment_ids()
        if not selected_ids:
            lines = [
                "No experiments selected.",
                "",
                "Use the selection buttons or multi-select rows in the table.",
            ]
        elif len(selected_ids) == 1:
            config = self.configs[selected_ids[0]]
            lines = [
                f"{config.id} - {config.display_name}",
                f"Category: {config.category}",
                f"Group: {config.docs_group}",
                f"Default size: {config.default_size}",
                f"Plan title: {config.plan_title}",
                f"Experiment folder: {config.experiment_root}",
                "",
                config.plan_description,
            ]
        else:
            selected_configs = [self.configs[experiment_id] for experiment_id in selected_ids]
            groups = sorted({config.docs_group for config in selected_configs})
            categories = sorted({config.category for config in selected_configs})
            lines = [
                f"Selected experiments: {len(selected_configs)}",
                f"Groups: {', '.join(groups)}",
                f"Categories: {', '.join(categories)}",
                "",
                "Selection order:",
            ]
            lines.extend(f"- {config.id} {config.display_name}" for config in selected_configs)

        self.details_text.configure(state=tk.NORMAL)
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert("1.0", "\n".join(lines))
        self.details_text.configure(state=tk.DISABLED)

    def _open_selected_results(self) -> None:
        selected_ids = self._selected_experiment_ids()
        if not selected_ids:
            messagebox.showerror("Open Results", "Select at least one experiment.")
            return

        results_path = self.configs[selected_ids[0]].experiment_root / "results"
        self._open_path(results_path)

    def _open_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror("Open Path", f"Path does not exist:\n{path}")
            return

        if os.name == "nt":
            os.startfile(str(path))
            return

        subprocess.Popen(["xdg-open", str(path)], cwd=str(ROOT))

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_experiment_status(self, experiment_id: str, status: str) -> None:
        self.status_by_experiment[experiment_id] = status
        if not self.tree.exists(experiment_id):
            return

        values = list(self.tree.item(experiment_id, "values"))
        values[-1] = status
        self.tree.item(experiment_id, values=values)

    def _set_busy_state(self, busy: bool) -> None:
        widget_state = tk.DISABLED if busy else tk.NORMAL
        for button in (
            self.build_button,
            self.run_button,
            self.run_artifacts_button,
            self.artifacts_button,
        ):
            button.configure(state=widget_state)
        self.stop_button.configure(state=tk.NORMAL if busy else tk.DISABLED)
        if busy:
            self.progressbar.start(10)
        else:
            self.progressbar.stop()

    def _validate_run_settings(self) -> bool:
        if self.iterations_var.get() <= 0:
            messagebox.showerror("Invalid settings", "Timed iterations must be at least 1.")
            return False
        if self.warmup_var.get() < 0:
            messagebox.showerror("Invalid settings", "Warmup iterations cannot be negative.")
            return False
        return True

    def _start_build(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showerror("Busy", "Another action is already running.")
            return

        self.status_var.set("Building benchmark binary...")
        self._set_busy_state(True)
        self.stop_requested.clear()
        self.worker_thread = threading.Thread(target=self._run_build_worker, daemon=True)
        self.worker_thread.start()

    def _start_run_only(self) -> None:
        self._start_experiment_worker(action="run")

    def _start_run_and_artifacts(self) -> None:
        self._start_experiment_worker(action="run_and_artifacts")

    def _start_artifacts_only(self) -> None:
        self._start_experiment_worker(action="artifacts")

    def _start_experiment_worker(self, action: str) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showerror("Busy", "Another action is already running.")
            return
        if not self._validate_run_settings():
            return

        selected_ids = self._selected_experiment_ids()
        if not selected_ids:
            messagebox.showerror("No selection", "Select at least one experiment.")
            return

        self.status_var.set(f"Preparing {action.replace('_', ' ')}...")
        self._set_busy_state(True)
        self.stop_requested.clear()
        self.worker_thread = threading.Thread(target=self._run_experiment_worker, args=(action, selected_ids), daemon=True)
        self.worker_thread.start()

    def _request_stop(self) -> None:
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return

        self.stop_requested.set()
        self.status_var.set("Stopping current action...")
        self._append_log("[stop] Stop requested. Waiting for the current process to terminate.\n")
        self._terminate_active_process()

    def _run_build_worker(self) -> None:
        try:
            self._run_streaming_command(["cmake", "--preset", "windows-tests-vs"])
            if self.stop_requested.is_set():
                self.event_queue.put(("finished", False, "Build stopped."))
                return

            self._run_streaming_command(
                ["cmake", "--build", "--preset", "tests-vs-release", "--target", "gpu_memory_layout_experiments"]
            )
            if self.stop_requested.is_set():
                self.event_queue.put(("finished", False, "Build stopped."))
                return

            try:
                self.event_queue.put(("binary", str(resolve_binary(None))))
            except FileNotFoundError:
                pass
            self.event_queue.put(("finished", True, "Build completed successfully."))
        except Exception as exc:
            self.event_queue.put(("finished", False, str(exc)))

    def _run_experiment_worker(self, action: str, selected_ids: list[str]) -> None:
        current_experiment_id = ""
        try:
            binary = resolve_binary(self.binary_var.get().strip() or None)
            self.event_queue.put(("binary", str(binary)))
        except FileNotFoundError as exc:
            self.event_queue.put(("finished", False, str(exc)))
            return

        try:
            total = len(selected_ids)
            for index, experiment_id in enumerate(selected_ids, start=1):
                current_experiment_id = experiment_id
                if self.stop_requested.is_set():
                    self.event_queue.put(("finished", False, "Action stopped by user."))
                    return

                config = self.configs[experiment_id]
                self.event_queue.put(("experiment_status", experiment_id, "running"))
                self.event_queue.put(
                    ("status", f"{index}/{total} {experiment_id} {config.display_name}: {action.replace('_', ' ')}")
                )

                if action in ("run", "run_and_artifacts"):
                    run_command = self._build_collection_command(config, Path(binary))
                    self._run_streaming_command(run_command)
                    if self.stop_requested.is_set():
                        self.event_queue.put(("experiment_status", experiment_id, "stopped"))
                        self.event_queue.put(("finished", False, "Action stopped by user."))
                        return

                if action in ("artifacts", "run_and_artifacts"):
                    artifact_command = self._build_artifact_command(config, action)
                    self._run_streaming_command(artifact_command)
                    if self.stop_requested.is_set():
                        self.event_queue.put(("experiment_status", experiment_id, "stopped"))
                        self.event_queue.put(("finished", False, "Action stopped by user."))
                        return

                self.event_queue.put(("experiment_status", experiment_id, "done"))

            self.event_queue.put(("finished", True, f"Completed {action.replace('_', ' ')} for {total} experiment(s)."))
        except Exception as exc:
            if current_experiment_id:
                self.event_queue.put(("experiment_status", current_experiment_id, "failed"))
            self.event_queue.put(("finished", False, str(exc)))

    def _build_collection_command(self, config: ExperimentWorkflowConfig, binary: Path) -> list[str]:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment_data_collection.py"),
            "--experiment",
            config.id,
            "--binary",
            str(binary),
            "--iterations",
            str(self.iterations_var.get()),
            "--warmup",
            str(self.warmup_var.get()),
        ]

        size_override = self.size_override_var.get().strip()
        if size_override:
            command.extend(["--size", size_override])
        if self.validation_var.get():
            command.append("--validation")
        if self.verbose_progress_var.get():
            command.append("--verbose-progress")
        label = self.label_var.get().strip()
        if label:
            command.extend(["--label", label])
        if not self.collect_run_var.get():
            command.append("--no-collect")

        return command

    def _build_artifact_command(self, config: ExperimentWorkflowConfig, action: str) -> list[str]:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "generate_experiment_artifacts.py"),
            "--experiment",
            config.id,
        ]
        should_collect_current = self.collect_current_before_artifacts_var.get() and (
            action == "artifacts" or not self.collect_run_var.get()
        )
        if should_collect_current:
            command.append("--collect-run")
        return command

    def _run_streaming_command(self, command: list[str]) -> None:
        display = subprocess.list2cmdline(command)
        self.event_queue.put(("log", f"[run] {display}\n"))

        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            # The stop button needs to terminate the full subprocess tree, not
            # just the top-level Python wrapper process.
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )

        with self.process_lock:
            self.active_process = process

        assert process.stdout is not None
        try:
            for line in process.stdout:
                self.event_queue.put(("log", line))
                if self.stop_requested.is_set():
                    self._terminate_active_process()
            return_code = process.wait()
        finally:
            with self.process_lock:
                self.active_process = None

        if self.stop_requested.is_set():
            return
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {display}")

    def _terminate_active_process(self) -> None:
        with self.process_lock:
            process = self.active_process

        if process is None or process.poll() is not None:
            return

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                cwd=str(ROOT),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        process.terminate()

    def _drain_event_queue(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            event_type = event[0]
            if event_type == "log":
                self._append_log(str(event[1]))
            elif event_type == "status":
                self.status_var.set(str(event[1]))
            elif event_type == "experiment_status":
                _, experiment_id, status = event
                self._set_experiment_status(str(experiment_id), str(status))
            elif event_type == "binary":
                self.binary_var.set(str(event[1]))
            elif event_type == "finished":
                _, success, message = event
                self.status_var.set(str(message))
                self._set_busy_state(False)
                if not bool(success):
                    self._append_log(f"[error] {message}\n")

        self.root.after(LOG_POLL_INTERVAL_MS, self._drain_event_queue)

    def _on_close(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            if not messagebox.askyesno("Exit", "An action is still running. Stop it and exit?"):
                return
            self.stop_requested.set()
            self._terminate_active_process()

        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    ExperimentRunnerGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
