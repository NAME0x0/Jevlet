"""Small Tk desktop panel. Suggestions never execute actions automatically."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

from .feedback import DesktopFeedbackStore
from .harness import ActionProposal, ActionRequest, DesktopHarness
from .native import DesktopContext

DecideCallback = Callable[[str, DesktopContext, tuple[str, ...]], object]
RateCallback = Callable[[int, str | None], None]
ROUTES = ("Local", "Codex", "Claude", "Gemini", "Retrieval", "Human")


class DesktopApp:
    def __init__(
        self,
        harness: DesktopHarness,
        feedback: DesktopFeedbackStore,
        decide: DecideCallback | None = None,
        on_rate: RateCallback | None = None,
    ) -> None:
        self.harness = harness
        self.feedback = feedback
        self.decide = decide
        self.on_rate = on_rate
        self.root = tk.Tk()
        self.root.title("Jevlet Desktop")
        self.root.geometry("560x570")
        self.root.resizable(True, True)
        self.context: DesktopContext | None = None
        self.proposal: ActionProposal | None = None
        self.last_suggestion = ""
        self.task = tk.StringVar()
        self.action = tk.StringVar(value="launch")
        self.target = tk.StringVar(value="notepad")
        self.text = tk.StringVar()
        self.correction = tk.StringVar()
        self.status = tk.StringVar(
            value="Screen access needs a signed-in desktop session. Preview and confirm actions."
        )
        self._build()

    def _build(self) -> None:
        panel = ttk.Frame(self.root, padding=12)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text="What do you want to do?").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Entry(panel, textvariable=self.task).grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(panel, text="Suggest route", command=self._suggest).grid(
            row=2, column=0, sticky="w", pady=8
        )
        ttk.Button(panel, text="Inspect other window", command=self._inspect).grid(
            row=2, column=1, sticky="e", pady=8
        )
        self.context_box = tk.Text(panel, height=10, wrap="word", state="disabled")
        self.context_box.grid(row=3, column=0, columnspan=2, sticky="nsew")
        panel.rowconfigure(3, weight=1)
        ttk.Button(panel, text="Screen preview", command=self._screenshot).grid(
            row=4, column=0, sticky="w", pady=8
        )
        ttk.Label(panel, text="Action").grid(row=5, column=0, sticky="w")
        ttk.Combobox(
            panel,
            textvariable=self.action,
            values=("launch", "switch", "focus", "type"),
            state="readonly",
        ).grid(row=5, column=1, sticky="ew")
        ttk.Label(panel, text="App / window title / automation ID").grid(
            row=6, column=0, sticky="w"
        )
        ttk.Entry(panel, textvariable=self.target).grid(row=6, column=1, sticky="ew")
        ttk.Label(panel, text="Text to type (never saved)").grid(row=7, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.text).grid(row=7, column=1, sticky="ew")
        ttk.Button(panel, text="Preview action", command=self._preview).grid(
            row=8, column=0, sticky="w", pady=8
        )
        ttk.Button(panel, text="Confirm action", command=self._execute).grid(
            row=8, column=1, sticky="e", pady=8
        )
        ttk.Label(panel, text="Was the suggestion useful?").grid(row=9, column=0, sticky="w")
        buttons = ttk.Frame(panel)
        buttons.grid(row=9, column=1, sticky="e")
        ttk.Button(buttons, text="👍", command=lambda: self._rate(1)).pack(side="left")
        ttk.Button(buttons, text="👎", command=lambda: self._rate(-1)).pack(side="left")
        ttk.Label(panel, text="If 👎, choose the right route (optional)").grid(
            row=10, column=0, sticky="w"
        )
        self.correction_picker = ttk.Combobox(
            panel, textvariable=self.correction, values=(), state="readonly"
        )
        self.correction_picker.grid(row=10, column=1, sticky="ew")
        ttk.Label(panel, textvariable=self.status, wraplength=525).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=8
        )

    def _set_context(self, context: DesktopContext) -> None:
        self.context = context
        title = context.window.title if context.window else "No foreground window"
        lines = [f"Window: {title}", f"Accessible controls: {len(context.controls)}"]
        lines.extend(
            f"• {control.control_type}: {control.name} [{control.automation_id}]"
            for control in context.controls[:30]
        )
        if not context.controls:
            lines.append("Install pywinauto to inspect accessibility controls.")
        self.context_box.configure(state="normal")
        self.context_box.delete("1.0", "end")
        self.context_box.insert("1.0", "\n".join(lines))
        self.context_box.configure(state="disabled")
        self.status.set("Context refreshed. Nothing was uploaded or saved.")

    def _inspect(self) -> None:
        self.root.iconify()

        def capture() -> None:
            try:
                self._set_context(self.harness.observe())
            except Exception as exc:
                self.status.set(str(exc))
            finally:
                self.root.deiconify()

        self.root.after(1200, capture)

    def _screenshot(self) -> None:
        self.root.iconify()

        def capture() -> None:
            try:
                from PIL import ImageTk

                image = self.harness.backend.capture_screen()
                image.thumbnail((1000, 650))
                preview = tk.Toplevel(self.root)
                preview.title("Screen preview — not saved")
                photo = ImageTk.PhotoImage(image)
                label = ttk.Label(preview, image=photo)
                label.image = photo
                label.pack()
                self.status.set("Screen preview captured locally; no image was stored.")
            except Exception as exc:
                self.status.set(str(exc))
            finally:
                self.root.deiconify()

        self.root.after(1200, capture)

    def _suggest(self) -> None:
        if self.decide is None:
            self.status.set("No decision model configured. Manual action controls still work.")
            return
        task = self.task.get().strip()
        if not task:
            self.status.set("Describe the task first.")
            return
        try:
            self.last_suggestion = ""
            context = self.context or self.harness.observe()
            result = self.decide(task, context, ROUTES)
            choice = getattr(result, "choice", result)
            self.last_suggestion = str(getattr(choice, "selected", choice))
            options = tuple(getattr(choice, "options", ROUTES))
            self.correction.set("")
            self.correction_picker.configure(
                values=tuple(option for option in options if option != self.last_suggestion)
            )
            confidence = getattr(choice, "confidence", None)
            gate = getattr(result, "gate", None)
            risk = getattr(result, "risk", None)
            detail = f" ({confidence:.0%} confidence)" if isinstance(confidence, float) else ""
            detail += f"; {gate}" if gate else ""
            detail += f"; risk {risk:.0%}" if isinstance(risk, float) else ""
            self.status.set(
                f"Suggested route: {self.last_suggestion}{detail}. No action was taken."
            )
        except Exception as exc:
            self.status.set(f"Could not suggest a route: {exc}")

    def _preview(self) -> None:
        try:
            request = ActionRequest(self.action.get(), self.target.get().strip(), self.text.get())
            if request.kind in {"focus", "type"} and (
                self.context is None or self.context.window is None
            ):
                raise ValueError("Inspect the target window before focusing or typing")
            expected = self.context.window.handle if self.context and self.context.window else None
            self.proposal = self.harness.propose(request, expected_window_handle=expected)
            title = self.proposal.expected_window_title or "current desktop"
            self.status.set(f"Preview: {self.proposal.description} (target: {title}).")
        except Exception as exc:
            self.proposal = None
            self.status.set(str(exc))

    def _execute(self) -> None:
        if self.proposal is None:
            self.status.set("Preview the action first.")
            return
        target = self.proposal.expected_window_title or "current desktop"
        if not messagebox.askyesno(
            "Confirm desktop action", f"{self.proposal.description}\nTarget: {target}"
        ):
            return
        try:
            result = self.harness.execute(self.proposal, confirmed=True)
            self.status.set(f"Done: {result.detail}")
        except Exception as exc:
            self.status.set(f"Action failed: {exc}")
        finally:
            self.proposal = None

    def _rate(self, rating: int) -> None:
        if not self.last_suggestion:
            self.status.set("Request a suggestion before rating it.")
            return
        correction = self.correction.get() if rating < 0 else None
        try:
            if self.on_rate is not None:
                self.on_rate(rating, correction or None)
            self.feedback.rate(self.task.get(), self.last_suggestion, self.action.get(), rating)
        except (ValueError, RuntimeError) as exc:
            self.status.set(f"Could not save feedback: {exc}")
            return
        self.last_suggestion = ""
        self.status.set(
            "Feedback saved locally. A correction can train a validated adapter later."
            if rating < 0 and correction
            else "Feedback saved locally. A bare downvote is not a training label."
            if rating < 0
            else "Feedback saved locally as an approved label."
        )

    def run(self) -> None:
        self.root.mainloop()
