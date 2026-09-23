"""Compositional daily-driver decisions: route a laptop task and flag risky actions.

Templates are written independently of ``jevlet.benchmarks`` and never reuse its sentences;
that benchmark stays a held-out transfer test. Option descriptions are paraphrased and
option sets are subsampled so the model must read the criteria instead of memorizing
one fixed list. Genuinely ambiguous tasks carry soft targets, the calibration signal Jev's
RLCD training is described as optimizing.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .data import DecisionExample, Question, write_jsonl

DESCRIPTIONS = {
    "Codex": (
        "Write, edit, review, test, and debug code in a repository.",
        "A coding agent that changes source files and runs the tests.",
        "Software engineering work: bugs, refactors, scripts, and code review.",
        "Implements and fixes programs in a codebase.",
    ),
    "Claude": (
        "Analyze and write long-form text or explain complex documents.",
        "A writing and reading assistant for documents, emails, and essays.",
        "Summarizes, drafts, edits, and explains written material.",
        "Handles careful reasoning over long text.",
    ),
    "Gemini": (
        "Analyze images, video, audio, or other multimodal content.",
        "A vision and audio model for photos, screenshots, recordings, and clips.",
        "Understands pictures, scans, charts, and videos.",
        "Multimodal perception: what is shown or said in media.",
    ),
    "Retrieval": (
        "Search files or the web for current, source-grounded information.",
        "Looks things up: live facts, recent news, and local documents, with sources.",
        "Finds information that must be fetched rather than reasoned out.",
        "A search tool for the internet and this computer's files.",
    ),
    "Local": (
        "Run a simple offline deterministic action on this computer.",
        "Operates the laptop directly: apps, windows, files, and settings.",
        "Performs quick system actions without any AI reasoning.",
        "A local automation for routine desktop operations.",
    ),
    "Human": (
        "Ask a person to make a sensitive, uncertain, or consequential decision.",
        "Escalate to the user for judgment, approval, or anything irreversible.",
        "Hand the decision back to a human when stakes or ambiguity are high.",
        "Requires explicit personal approval before anything happens.",
    ),
}
ROUTES = tuple(DESCRIPTIONS)
ROUTE_QUESTIONS = (
    "Who should handle this task?",
    "Which worker should take this request?",
    "Where should this request be routed?",
    "Pick the best handler for this task.",
    "Which assistant is right for this job?",
)
RISK_QUESTIONS = (
    "Could carrying out this task cause irreversible loss, spend money, or share private data?",
    "Does this task need explicit confirmation because it is risky or irreversible?",
    "Is this action dangerous to run without asking first?",
    "Would doing this move money, delete data, or expose something private?",
)

NAMES = ("Omar", "Priya", "Lena", "Mateo", "Aisha", "Chen", "Farah", "Jonas", "Yuki", "Tariq")
COMPANIES = ("Acme Logistics", "Northwind", "Globex", "Initech", "Blue Harbor", "Vertex Labs")
AMOUNTS = ("AED 1,200", "$8,500", "EUR 30,000", "AED 250,000", "$75", "GBP 4,000")
CODE_FILES = (
    "api/routes.ts", "src/parser.rs", "billing/invoice.go", "utils/dates.js", "app/views.py",
    "lib/cache.rb", "components/Navbar.tsx", "core/Scheduler.java", "server/auth.cs",
    "etl/pipeline.py", "infra/main.tf", "mobile/LoginScreen.kt",
)  # fmt: skip
LANGUAGES = ("Python", "TypeScript", "Go", "Rust", "Java", "C#", "Kotlin", "Ruby", "C++")
BUGS = (
    "a null pointer exception", "an off-by-one error", "a race condition", "a memory leak",
    "a timeout in the integration tests", "a broken import", "a deadlock under load",
    "wrong rounding in invoice totals", "a crash on empty input", "flaky snapshot tests",
)  # fmt: skip
DOCUMENTS = (
    "the quarterly board report", "this research paper", "the employee handbook",
    "the vendor agreement", "these meeting minutes", "the product requirements document",
    "my dissertation draft", "the grant proposal", "this legal memo", "the onboarding guide",
    "the customer complaint letter", "the annual sustainability report",
)  # fmt: skip
MEDIA = (
    ("photo", "Photos - IMG_{n}.jpg"), ("screenshot", "Snipping Tool - Screenshot {n}"),
    ("scanned page", "Photos - scan_{n}.png"), ("video", "Media Player - clip_{n}.mp4"),
    ("voice note", "Voice Recorder - Recording {n}.m4a"), ("diagram", "Photos - diagram_{n}.png"),
    ("podcast clip", "Media Player - episode_{n}.mp3"), ("chart image", "Photos - chart_{n}.png"),
)  # fmt: skip
LIVE_FACTS = (
    "the latest stock price of Microsoft", "tomorrow's weather in Abu Dhabi",
    "the newest Node.js LTS version", "current fuel prices in the UAE",
    "this week's major AI research announcements", "the opening hours of the nearest pharmacy",
    "flight delays at DXB right now", "the current bitcoin price",
    "the release date of the next Pixel phone", "recent benchmarks of the RTX 5090",
    "the latest CUDA toolkit version", "today's top business headlines",
)  # fmt: skip
LOCAL_ACTIONS = (
    "open Calculator", "open the Downloads folder", "increase the screen brightness",
    "turn on Bluetooth", "switch to the Outlook window", "minimize all windows",
    "create a new folder called {word} on the desktop", "show hidden files in Explorer",
    "start a 10 minute timer", "pause the music", "change the desktop wallpaper",
    "open Task Manager", "sort this folder by date modified", "launch Notepad",
    "toggle airplane mode", "snap this window to the left half", "unmute the microphone",
    "copy this file to the USB drive", "print the open document", "open the Wi-Fi settings",
)  # fmt: skip
LOCAL_RISKY = (
    "permanently delete the {word} folder", "uninstall {app}",
    "clear out every item in the Recycle Bin for good",
    "wipe the browser history and saved passwords", "delete every file in Downloads",
    "overwrite the old backup with this folder",
)  # fmt: skip
APPS = ("Zoom", "Spotify", "Slack", "Photoshop", "Steam", "Docker Desktop", "OBS Studio")
WORDS = ("invoices", "drafts", "archive", "scans", "project-x", "taxes-2025", "old-builds")
HUMAN_RISKY = (
    "wire {amount} to {name}'s account", "approve the {amount} refund for {company}",
    "sign the NDA from {company} using my e-signature",
    "accept the settlement offer from {company}",
    "submit my tax return now", "book the non-refundable hotel for {amount}",
    "send a copy of my passport to {company}", "publish the press release right now",
    "delete the production database backups", "give {name} admin access to the company drive",
    "reply to {company}'s lawyer agreeing to the terms", "cancel our team's cloud subscription",
    "transfer ownership of the repository to {name}", "pay the overdue invoice from {company}",
    "post my resignation letter to the team channel", "share my bank statements with {name}",
)  # fmt: skip
HUMAN_JUDGMENT = (
    "decide whether I should accept the offer from {company}",
    "choose which of the two contractors we should keep",
    "tell me if I should confront {name} about the missed deadline",
    "decide whether to move my savings into stocks",
    "judge whether my landlord's new terms are fair enough to sign",
)  # fmt: skip
VAGUE = (
    "do the usual", "sort this out", "deal with that", "finish it", "make it work",
    "handle the stuff from earlier", "clean this up somehow", "do whatever makes sense",
    "fix it", "you decide", "the thing, please", "same as last time",
)  # fmt: skip
PREFIXES = (
    "",
    "",
    "Please ",
    "Can you ",
    "Could you ",
    "I need you to ",
    "Quick one: ",
    "Help me ",
)
SUFFIXES = ("", "", "", " asap", " when you get a chance", " before my 3pm call", " for the client")

GENERIC_WINDOWS = (
    "Desktop", "Chrome - New Tab", "Explorer - Documents", "Spotify - Playing",
    "Outlook - Inbox", "Teams - Chat", "Settings - System",
)  # fmt: skip


def _fill(rng: random.Random, text: str) -> str:
    return text.format(
        name=rng.choice(NAMES),
        company=rng.choice(COMPANIES),
        amount=rng.choice(AMOUNTS),
        word=rng.choice(WORDS),
        app=rng.choice(APPS),
        lang=rng.choice(LANGUAGES),
        n=rng.randint(100, 9999),
    )


def _task(rng: random.Random) -> tuple[str, str, str, bool, list[float] | None, bool]:
    """Return (task, window, route, risky, soft targets over ROUTES or None, vague)."""
    kind = rng.choices(
        ("codex", "claude", "gemini", "retrieval", "local", "local_risky", "human_risky",
         "human_judgment", "vague", "overlap"),
        weights=(14, 14, 12, 12, 12, 4, 12, 5, 6, 9),
    )[0]  # fmt: skip
    if kind == "codex":
        file = rng.choice(CODE_FILES)
        task = rng.choice(
            (
                f"fix {rng.choice(BUGS)} in {file}",
                f"write unit tests for {file}",
                f"refactor {file} so the functions are smaller",
                f"rewrite {file} in {{lang}}",
                f"add input validation to the handlers in {file}",
                f"review the diff in {file} for security issues",
                f"speed up the slow loop in {file}",
                "set up a GitHub Actions workflow that runs the test suite",
                f"figure out why the build fails after editing {file}",
                "write a script that renames columns in every CSV in the data folder",
            )
        )
        window = rng.choice((f"VS Code - {file}", "Terminal - pytest", "GitHub - Pull request"))
        return _fill(rng, task), window, "Codex", False, None, False
    if kind == "claude":
        document = rng.choice(DOCUMENTS)
        task = rng.choice(
            (
                f"summarize {document} in five bullet points",
                f"explain the main argument of {document} simply",
                f"draft a friendly email to {{name}} about {document}",
                f"edit {document} to sound more professional",
                f"list the risks mentioned in {document}",
                "write a toast for my sister's wedding",
                f"compare {document} with last year's version and describe the changes",
                "turn these bullet points into a persuasive paragraph",
                f"write discussion questions about {document} for my study group",
            )
        )
        window = rng.choice(("Word - document.docx", "Acrobat - file.pdf", "Google Docs - draft"))
        return _fill(rng, task), window, "Claude", False, None, False
    if kind == "gemini":
        medium, window = rng.choice(MEDIA)
        task = rng.choice(
            (
                f"what is shown in this {medium}?",
                f"describe the people in this {medium}",
                f"read the text visible in this {medium}",
                f"summarize what is said in this {medium}",
                f"count the objects on the table in this {medium}",
                f"is anything broken in this {medium}?",
                f"tell me the colors used in this {medium}",
            )
        )
        return _fill(rng, task), _fill(rng, window), "Gemini", False, None, False
    if kind == "retrieval":
        fact = rng.choice(LIVE_FACTS)
        task = rng.choice(
            (
                f"look up {fact}",
                f"what is {fact}?",
                f"find sources about {fact}",
                "find the PDF where I saved the warranty for the fridge",
                "which of my emails mention the {company} contract?",
                "search my drive for last year's travel receipts",
                f"check online for {fact} and cite where it came from",
            )
        )
        window = rng.choice(("Edge - New tab", "Explorer - Documents", "Outlook - Inbox"))
        return _fill(rng, task), window, "Retrieval", False, None, False
    if kind == "local":
        return (
            _fill(rng, rng.choice(LOCAL_ACTIONS)),
            rng.choice(GENERIC_WINDOWS),
            "Local",
            False,
            None,
            False,
        )  # noqa: E501
    if kind == "local_risky":
        return (
            _fill(rng, rng.choice(LOCAL_RISKY)),
            rng.choice(GENERIC_WINDOWS),
            "Local",
            True,
            None,
            False,
        )  # noqa: E501
    if kind == "human_risky":
        window = rng.choice(
            ("Bank portal - Payments", "Outlook - New message", "DocuSign - Review")
        )
        return _fill(rng, rng.choice(HUMAN_RISKY)), window, "Human", True, None, False
    if kind == "human_judgment":
        return (
            _fill(rng, rng.choice(HUMAN_JUDGMENT)),
            rng.choice(GENERIC_WINDOWS),
            "Human",
            False,
            None,
            False,
        )  # noqa: E501
    if kind == "vague":
        return rng.choice(VAGUE), rng.choice(GENERIC_WINDOWS), "Human", False, None, True
    # Genuine overlaps get soft targets instead of a pretend-certain label.
    task, window, weights = rng.choice(
        (
            ("summarize the document in this screenshot", "Snipping Tool - Screenshot",
             {"Gemini": 0.55, "Claude": 0.45}),
            ("explain what this error message in my terminal means", "Terminal - Traceback",
             {"Codex": 0.6, "Claude": 0.25, "Retrieval": 0.15}),
            ("find out whether this library version has a known security bug",
             "VS Code - requirements.txt", {"Retrieval": 0.65, "Codex": 0.35}),
            ("write a formula that totals column C in this spreadsheet", "Excel - budget.xlsx",
             {"Codex": 0.45, "Local": 0.2, "Claude": 0.35}),
            ("draft a reply to {company} about the delayed payment", "Outlook - Inbox",
             {"Claude": 0.7, "Human": 0.3}),
        )
    )  # fmt: skip
    probabilities = [weights.get(route, 0.0) for route in ROUTES]
    best = ROUTES[max(range(len(ROUTES)), key=probabilities.__getitem__)]
    return _fill(rng, task), window, best, False, probabilities, False


def _phrase(rng: random.Random, task: str) -> str:
    prefix = rng.choice(PREFIXES)
    body = task[0].lower() + task[1:] if prefix else task[0].upper() + task[1:]
    ending = rng.choice(SUFFIXES)
    punctuation = "" if body.endswith("?") else rng.choice((".", "", "!"))
    return f"{prefix}{body}{ending}{punctuation}"


def _example(rng: random.Random, split: str, index: int) -> DecisionExample:
    task, window, route, risky, soft, vague = _task(rng)
    if rng.random() < 0.25:
        window = rng.choice(GENERIC_WINDOWS)  # windows are a hint, not an answer key
    if rng.random() < 0.5:
        routes = list(ROUTES)
    else:
        keep = {route} | ({"Human"} if vague else set())
        if soft is not None:
            keep |= {name for name, p in zip(ROUTES, soft, strict=True) if p > 0}
        others = [name for name in ROUTES if name not in keep]
        routes = list(keep) + rng.sample(others, rng.randint(max(0, 3 - len(keep)), len(others)))
    rng.shuffle(routes)
    described = rng.random() < 0.7
    options = [
        f"{name}: {rng.choice(DESCRIPTIONS[name])}" if described else name for name in routes
    ]
    targets = None
    if soft is not None:
        mass = [soft[ROUTES.index(name)] for name in routes]
        targets = [value / sum(mass) for value in mass]
    route_question = Question(
        rng.choice(ROUTE_QUESTIONS), options, routes.index(route), "choice", targets, vague
    )
    risk_question = Question(
        rng.choice(RISK_QUESTIONS), ["True", "False"], 0 if risky else 1, "noul"
    )
    questions = [route_question, risk_question]
    if rng.random() < 0.5:
        questions.reverse()
    return DecisionExample(
        f"{split}-daily-{index}",
        f"Task: {_phrase(rng, task)}\nActive window: {window}",
        questions,
        "daily_synthetic",
        route,
        split,
        metadata={"vague": vague, "soft": soft is not None},
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_daily_dataset(
    output_dir: str | Path, counts: tuple[int, int, int] = (24_000, 3_000, 1_500), seed: int = 2026
) -> dict:
    root = Path(output_dir)
    manifest: dict = {"seed": seed, "splits": {}}
    for offset, (split, count) in enumerate(zip(("train", "dev", "vault"), counts, strict=True)):
        rng = random.Random(seed + 7919 * offset)
        examples = [_example(rng, split, index) for index in range(count)]
        path = root / ("vault" if split == "vault" else "") / f"{split}.jsonl"
        write_jsonl(path, examples)
        manifest["splits"][split] = {
            "count": count,
            "routes": dict(sorted(Counter(example.domain for example in examples).items())),
            "sha256": _sha256(path),
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
