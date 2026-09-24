"""Generated assistant commands: which skill, which arguments, how risky.

Every example is built with ``skills.slot_options`` on a sampled ``Environment``, the same
code the live planner uses, so training options are the options the model will see. Gold
arguments missed by the lexical shortlist are inserted (and counted) so a weak shortlist is
visible in the manifest instead of silently teaching "not applicable".
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from jevlet.benchmarks import RISK_QUESTION, daily_state
from jevlet.data import DecisionExample, Question, write_jsonl

from .skills import (
    MEDIA_ACTIONS,
    SERVICES,
    SKILL_BY_KEY,
    SKILL_QUESTION,
    SKILLS,
    SLOTS,
    THEMES,
    VOLUME_ACTIONS,
    Environment,
    slot_options,
    window_label,
)

APP_POOL = (
    "Spotify", "Visual Studio Code", "Microsoft Edge", "Google Chrome", "Firefox", "Brave",
    "Outlook", "Microsoft Teams", "Slack", "Discord", "WhatsApp", "Telegram", "Signal", "Zoom",
    "Notepad", "Notepad++", "Calculator", "Settings", "File Explorer", "Word", "Excel",
    "PowerPoint", "OneNote", "Photos", "Paint", "Snipping Tool", "Task Manager", "Terminal",
    "Windows PowerShell", "Command Prompt", "Steam", "Epic Games Launcher", "OBS Studio",
    "VLC media player", "Media Player", "Clock", "Calendar", "Microsoft Store", "Obsidian",
    "Notion", "Figma", "Postman", "Docker Desktop", "PyCharm", "Android Studio", "Blender",
    "GIMP", "Audacity", "Adobe Acrobat", "Adobe Photoshop", "DaVinci Resolve", "Sticky Notes",
    "Camera", "Maps", "Weather", "Xbox", "Remote Desktop Connection", "Control Panel",
    "Git Bash", "GitHub Desktop", "Cursor", "ChatGPT", "Claude", "Thunderbird", "Opera",
    "PowerToys", "qBittorrent", "Anki", "Kindle", "Zotero", "LibreOffice Writer", "Evernote",
    "Todoist", "Trello", "1Password", "Bitwarden", "Dropbox", "OneDrive", "Google Drive",
    "Skype", "Webex", "Cisco AnyConnect", "WinRAR", "7-Zip File Manager", "Everything",
)  # fmt: skip
APP_ALIASES = {
    "Spotify": ("my music app", "spotify", "the music player"),
    "Visual Studio Code": ("vs code", "vscode", "code editor", "visual studio code"),
    "Microsoft Edge": ("edge", "microsoft edge"),
    "Google Chrome": ("chrome", "google chrome"),
    "Outlook": ("outlook", "my email", "the mail app"),
    "Microsoft Teams": ("teams", "ms teams"),
    "File Explorer": ("file explorer", "explorer", "my files"),
    "Calculator": ("the calculator", "calc", "calculator"),
    "Terminal": ("the terminal", "terminal", "a shell"),
    "Task Manager": ("task manager",),
    "Snipping Tool": ("snipping tool", "the snip tool"),
    "Word": ("word", "microsoft word"),
    "Excel": ("excel", "a spreadsheet app"),
    "PowerPoint": ("powerpoint", "slides app"),
    "OneNote": ("onenote", "my notes app"),
}
WINDOW_POOL = (
    ("chrome", "{topic} - Google Chrome", ("chrome", "the chrome window", "my browser")),
    ("msedge", "{topic} - Microsoft Edge", ("edge", "the edge window", "the browser")),
    ("Code", "{file} - Jevlet - Visual Studio Code", ("vs code", "my editor", "the code window")),
    ("OUTLOOK", "Inbox - omar@northwind.ae - Outlook", ("outlook", "my inbox", "email")),
    ("Spotify", "Spotify Premium", ("spotify", "the music")),
    ("ms-teams", "Chat | Microsoft Teams", ("teams", "the teams chat")),
    ("WINWORD", "{doc}.docx - Word", ("word", "the word document", "my document")),
    ("EXCEL", "{book}.xlsx - Excel", ("excel", "the spreadsheet")),
    ("explorer", "Downloads", ("explorer", "the downloads folder window")),
    ("WindowsTerminal", "PowerShell", ("the terminal", "powershell")),
    ("Discord", "#general | Study Group - Discord", ("discord",)),
    ("slack", "general (Channel) - Northwind - Slack", ("slack",)),
    ("notepad", "{doc}.txt - Notepad", ("notepad", "my notes")),
    ("Photos", "IMG_2041.jpg - Photos", ("photos", "the photo")),
    ("obs64", "OBS 30.2 - Profile: Streaming", ("obs",)),
    ("ChatGPT", "ChatGPT", ("chatgpt",)),
    ("Zoom", "Zoom Meeting", ("zoom", "the meeting")),
)  # fmt: skip
TOPICS = ("Flights to London", "RTX A2000 drivers", "YouTube", "Pull request #88", "Gmail")
FILES = ("training.py", "README.md", "planner.py", "app.tsx")
DOCS = ("thesis_draft", "budget", "notes", "cover_letter", "Q3 report")

SETTINGS_PHRASES = {
    "Display and brightness": ("display settings", "change the screen brightness", "screen resolution settings"),
    "Sound and volume devices": ("sound settings", "change my audio output device", "switch speakers to headphones"),
    "Bluetooth and devices": ("bluetooth settings", "pair my headphones", "connect a bluetooth mouse"),
    "Wi-Fi networks": ("wifi settings", "change the wifi network", "connect to wifi"),
    "VPN": ("vpn settings", "set up a vpn"),
    "Wallpaper and background": ("change my wallpaper", "background settings"),
    "Colors and dark mode": ("color settings", "accent color settings"),
    "Windows Update": ("check for updates", "windows update"),
    "Installed apps": ("uninstall programs settings", "see my installed apps"),
    "Default apps": ("change my default browser", "default apps settings"),
    "Camera privacy": ("which apps can use my camera", "camera permissions"),
    "Microphone privacy": ("microphone permissions", "which apps use my mic"),
    "Power and battery": ("battery settings", "power and sleep settings", "change when the screen sleeps"),
    "Storage": ("storage settings", "free up disk space"),
    "Keyboard and typing": ("keyboard settings", "typing settings"),
    "Date, time, and time zone": ("change the time zone", "date and time settings"),
    "Notifications": ("notification settings", "turn off notifications for apps"),
    "Mouse": ("mouse settings", "change mouse speed"),
    "Printers and scanners": ("add a printer", "printer settings"),
    "Accounts and sign-in": ("sign-in options", "change my pin"),
    "Language and region": ("language settings", "add a keyboard language"),
    "Accessibility text size": ("make text bigger", "text size settings"),
    "Multiple displays": ("set up my second monitor", "multiple displays settings"),
    "Focus and do not disturb": ("do not disturb settings", "focus settings"),
}  # fmt: skip
WEBSITE_PHRASES = {
    "YouTube": ("youtube",), "Gmail": ("gmail", "my gmail"), "Google Drive": ("google drive",),
    "Google Calendar": ("google calendar",), "GitHub": ("github",), "ChatGPT": ("chatgpt website",),
    "Claude": ("claude.ai",), "Gemini": ("gemini website",), "Hugging Face": ("hugging face", "huggingface"),
    "Google Colab": ("colab", "google colab"), "LinkedIn": ("linkedin",), "Reddit": ("reddit",),
    "X (Twitter)": ("twitter", "x.com"), "WhatsApp Web": ("whatsapp web",), "Netflix": ("netflix",),
    "Amazon": ("amazon",), "Wikipedia": ("wikipedia",), "Google Maps": ("google maps",),
    "Outlook on the web": ("outlook web", "outlook online"), "Stack Overflow": ("stack overflow",),
}  # fmt: skip
FOLDER_PHRASES = {
    "Downloads": ("my downloads", "the downloads folder"), "Documents": ("my documents", "documents"),
    "Desktop": ("the desktop folder",), "Pictures": ("my pictures", "the pictures folder"),
    "Music": ("my music folder",), "Videos": ("my videos",), "Screenshots": ("my screenshots folder",),
    "Recycle Bin": ("the recycle bin",), "This PC": ("this pc", "my drives"), "Home folder": ("my home folder",),
}  # fmt: skip
SHORTCUT_PHRASES = {
    "Copy": ("copy that", "copy the selection"), "Paste": ("paste", "paste it here"),
    "Cut": ("cut this",), "Undo": ("undo", "undo that", "take that back"), "Redo": ("redo",),
    "Save": ("save this", "save the file"), "Select all": ("select everything", "select all"),
    "Find on page": ("find on this page", "search this page"), "New tab": ("open a new tab", "new tab"),
    "Close tab": ("close this tab",), "Reopen closed tab": ("reopen the tab I closed", "bring back the closed tab"),
    "Refresh": ("refresh", "reload the page"), "Zoom in": ("zoom in", "make it bigger"),
    "Zoom out": ("zoom out", "make it smaller"), "Print": ("print this",),
    "Show desktop": ("show the desktop", "hide all windows"), "Task view": ("show all my windows", "task view"),
    "Snip a screenshot": ("snip part of the screen", "clip an area of the screen"),
    "Emoji panel": ("open the emoji picker",), "Clipboard history": ("show my clipboard history",),
    "Next virtual desktop": ("go to the next desktop",), "Previous virtual desktop": ("go to the previous desktop",),
}  # fmt: skip
SEARCH_QUERIES = (
    "cheap flights to london", "rtx a2000 driver download", "how to center a div", "best pizza near me",
    "python asyncio tutorial", "weather in dubai tomorrow", "latest cuda toolkit", "how tall is burj khalifa",
    "vegan lasagna recipe", "usd to aed exchange rate", "pytorch sdpa custom mask", "news about typesafe jev",
)  # fmt: skip
TYPE_TEXTS = (
    "see you at 5", "thanks, sounds good", "I'll be 10 minutes late", "Happy birthday!",
    "meeting moved to 3pm", "on my way", "can you send the file?", "omar@northwind.ae",
)  # fmt: skip
AI_REQUESTS = (
    "summarize this article", "write a poem about rain", "explain quantum tunneling simply",
    "fix the bug in my regex", "draft a reply to my landlord", "describe what's in this image",
    "translate this paragraph to french", "outline a blog post about remote work",
)  # fmt: skip
CLARIFY = (
    ("do the thing", 0.1), ("handle it", 0.1), ("you know what to do", 0.1), ("fix everything", 0.2),
    ("delete all my photos", 0.95), ("format the external drive", 0.97), ("send 500 dollars to omar", 0.97),
    ("buy the thing in my cart", 0.95), ("email my boss that I quit", 0.95), ("uninstall chrome", 0.8),
    ("should I take the job offer", 0.1), ("make me a sandwich", 0.05), ("wipe my laptop", 0.98),
    ("post this on my linkedin", 0.9), ("empty the recycle bin forever", 0.9), ("turn off the firewall", 0.95),
)  # fmt: skip
PREFIXES = ("", "", "", "please ", "can you ", "hey jevlet, ", "could you ", "quickly ", "I want to ")


def _typo(rng: random.Random, text: str) -> str:
    words = text.split()
    candidates = [index for index, word in enumerate(words) if len(word) > 4]
    if not candidates or rng.random() > 0.08:
        return text
    index = rng.choice(candidates)
    word = words[index]
    cut = rng.randrange(1, len(word) - 1)
    words[index] = word[:cut] + word[cut + 1 :]
    return " ".join(words)


def _environment(rng: random.Random, local_apps: tuple[str, ...]) -> tuple[Environment, dict]:
    pool = list(dict.fromkeys(APP_POOL + local_apps))
    apps = sorted(rng.sample(pool, min(len(pool), rng.randint(40, 90))), key=str.casefold)
    windows, labels = [], {}
    for process, title, phrases in rng.sample(WINDOW_POOL, rng.randint(2, 9)):
        filled = title.format(topic=rng.choice(TOPICS), file=rng.choice(FILES), doc=rng.choice(DOCS), book=rng.choice(DOCS))
        label = window_label(process, filled)
        windows.append(label)
        labels[label] = phrases
    return Environment(apps, windows, windows[0]), labels


def _command(rng: random.Random, skill: str, env: Environment, windows: dict) -> tuple[str, dict[str, str], float]:
    """Return (command, gold arguments, P(risky))."""
    if skill == "open_app":
        app = rng.choice([a for a in env.apps if a in APP_ALIASES] or env.apps) if rng.random() < 0.6 else rng.choice(env.apps)
        name = rng.choice(APP_ALIASES.get(app, (app.lower(), app)))
        verb = rng.choice(("open", "launch", "start", "fire up", "run", "bring up", "open up"))
        return f"{verb} {name}", {"app": app}, 0.02
    if skill in {"switch_window", "close_window", "minimize_window", "maximize_window"}:
        use_current = rng.random() < 0.35
        target = env.current_window if use_current else rng.choice(env.windows)
        name = rng.choice(("this", "this window", "it", "the current window")) if use_current else rng.choice(windows[target])
        verb = {
            "switch_window": ("switch to", "go to", "bring up", "show me", "jump to"),
            "close_window": ("close", "quit", "exit", "shut"),
            "minimize_window": ("minimize", "hide", "shrink"),
            "maximize_window": ("maximize", "make fullscreen", "enlarge"),
        }[skill]
        gold = f"Current window ({env.current_window})" if use_current or target == env.current_window else target
        return f"{rng.choice(verb)} {name}", {"window": gold}, 0.2 if skill == "close_window" else 0.02
    if skill == "media":
        action = rng.choice(MEDIA_ACTIONS)
        phrases = {
            "Play or pause": ("pause the music", "play", "resume the song", "pause", "stop the music for a sec"),
            "Next track": ("next song", "skip this track", "skip", "play the next one"),
            "Previous track": ("previous song", "go back a track", "play the last song again"),
        }[action]
        return rng.choice(phrases), {"media": action}, 0.01
    if skill == "volume":
        action = rng.choice(VOLUME_ACTIONS)
        phrases = {
            "Volume up": ("turn it up", "louder", "volume up", "increase the volume", "I can't hear it"),
            "Volume down": ("turn it down", "quieter", "lower the volume", "too loud"),
            "Mute or unmute": ("mute", "unmute", "mute the sound", "silence the laptop"),
        }[action]
        return rng.choice(phrases), {"volume": action}, 0.01
    if skill == "settings":
        page = rng.choice(list(SETTINGS_PHRASES))
        phrase = rng.choice(SETTINGS_PHRASES[page])
        return rng.choice((phrase, f"open {phrase}", f"take me to {phrase}")), {"setting": page}, 0.03
    if skill == "theme":
        mode = rng.choice(THEMES)
        word = "dark" if mode == "Dark mode" else "light"
        phrase = rng.choice((f"turn on {word} mode", f"switch to {word} theme", f"make everything {word}", f"{word} mode"))
        return phrase, {"theme": mode}, 0.02
    if skill == "search":
        query = rng.choice(SEARCH_QUERIES)
        template = rng.choice(("search for {q}", "google {q}", "look up {q}", "search the web for {q}", "{q}", "find {q} online"))
        return template.format(q=query), {"text": query}, 0.01
    if skill == "website":
        site = rng.choice(list(WEBSITE_PHRASES))
        phrase = rng.choice(WEBSITE_PHRASES[site])
        return f"{rng.choice(('open', 'go to', 'take me to', 'pull up'))} {phrase}", {"website": site}, 0.02
    if skill == "folder":
        folder = rng.choice(list(FOLDER_PHRASES))
        phrase = rng.choice(FOLDER_PHRASES[folder])
        return f"{rng.choice(('open', 'show me', 'go to'))} {phrase}", {"folder": folder}, 0.03
    if skill == "type":
        text = rng.choice(TYPE_TEXTS)
        template = rng.choice(('type "{t}"', "type {t}", 'write "{t}" here', "enter {t}", 'type out "{t}"'))
        return template.format(t=text), {"text": text}, 0.2
    if skill == "shortcut":
        name = rng.choice(list(SHORTCUT_PHRASES))
        return rng.choice(SHORTCUT_PHRASES[name]), {"shortcut": name}, 0.05 if name != "Close tab" else 0.15
    if skill == "click":
        target = rng.choice(("reply", "send", "play", "settings", "the save", "new chat", "the downloads tab", "submit"))
        risk = 0.8 if target in {"send", "submit"} else 0.08
        return f"{rng.choice(('click', 'press', 'hit', 'tap'))} {target} {rng.choice(('button', '', 'on screen'))}".strip(), {}, risk
    if skill == "timer":
        amount = rng.choice(("25 minutes", "an hour", "10 min", "90 seconds", "5 minutes", "half an hour", "2 hours"))
        return rng.choice((f"set a timer for {amount}", f"remind me in {amount}", f"countdown {amount}", f"timer {amount}")), {}, 0.01
    if skill == "screenshot":
        return rng.choice(("take a screenshot", "screenshot this", "capture my screen", "save a picture of my screen")), {}, 0.05
    if skill == "lock":
        return rng.choice(("lock my computer", "lock the screen", "lock the laptop", "I'm stepping away, lock it")), {}, 0.05
    if skill == "ask_ai":
        service = rng.choice(SERVICES)
        request = rng.choice(AI_REQUESTS)
        template = rng.choice(("ask {s} to {r}", "have {s} {r}", "use {s} to {r}", "{s}, {r}", "get {s} to {r}"))
        return template.format(s=service.lower() if rng.random() < 0.5 else service, r=request), {"service": service, "text": request}, 0.05
    phrase, risk = rng.choice(CLARIFY)
    return phrase, {}, risk


def _example(rng: random.Random, index: int, split: str, local_apps: tuple[str, ...], stats: Counter) -> DecisionExample:
    env, windows = _environment(rng, local_apps)
    skill_key = rng.choice([skill.key for skill in SKILLS])
    command, gold, risk = _command(rng, skill_key, env, windows)
    noisy = _typo(rng, rng.choice(PREFIXES) + command)
    # A typo inside text the user wants typed or searched would change the argument itself.
    command = noisy if gold.get("text", "") in noisy else rng.choice(PREFIXES) + command
    skills = list(SKILLS)
    if rng.random() < 0.5:
        rng.shuffle(skills)
    names = [skill.name for skill in skills]
    questions = [
        Question(SKILL_QUESTION, [f"{s.name}: {s.description}" for s in skills], names.index(SKILL_BY_KEY[skill_key].name)),
        Question(RISK_QUESTION, ["True", "False"], 0 if risk >= 0.5 else 1, "noul", [risk, 1 - risk]),
    ]
    for slot in SKILL_BY_KEY[skill_key].slots:
        options = slot_options(slot, command, env)
        answer = gold[slot]
        if answer not in options:
            stats[f"inserted_{slot}"] += 1
            options.insert(rng.randrange(len(options)), answer)
        questions.append(Question(SLOTS[slot].question, options, options.index(answer)))
    if rng.random() < 0.3:  # a question that does not apply to this command
        other = rng.choice([s for s in SLOTS if s not in SKILL_BY_KEY[skill_key].slots and s != "text"])
        options = slot_options(other, command, env)
        questions.append(Question(SLOTS[other].question, options, len(options) - 1))
    stats[skill_key] += 1
    return DecisionExample(
        f"{split}-command-{index}",
        daily_state(command, env.current_window),
        questions,
        "assistant",
        skill_key,
        split,
        metadata={"command": command},
    )


def generate_command_dataset(
    output_dir: str | Path,
    counts: tuple[int, int] = (40_000, 3_000),
    seed: int = 777,
    local_apps: tuple[str, ...] = (),
) -> dict:
    root = Path(output_dir)
    manifest: dict = {"seed": seed, "local_apps": len(local_apps), "splits": {}}
    for offset, (split, count) in enumerate(zip(("train", "dev"), counts, strict=True)):
        rng = random.Random(seed + 7919 * offset)
        stats: Counter = Counter()
        examples = [_example(rng, index, split, local_apps, stats) for index in range(count)]
        path = root / f"{split}.jsonl"
        write_jsonl(path, examples)
        manifest["splits"][split] = {
            "count": count,
            "stats": dict(sorted(stats.items())),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
