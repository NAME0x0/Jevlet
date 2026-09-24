"""Generated screen-grounding decisions over realistic Windows control inventories.

Each app lists the controls UI Automation typically exposes and, for the controls a task can
target, several phrasings of that task. Training rows show a random visible subset of the
app's controls plus window chrome; ``held_out`` apps never appear in training so the dev
split measures transfer to unseen applications. Tasks that belong to another app are
labeled ``NONE_OF_THESE`` so the model learns to say the screen cannot do it.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .benchmarks import RISK_QUESTION, daily_state
from .data import DecisionExample, Question, write_jsonl
from .grounding import GROUNDING_QUESTION, NONE_OF_THESE, control_text

CHROME = (("Button", "Minimize"), ("Button", "Maximize"), ("Button", "Close"))
NAMES = ("Omar", "Priya", "Lena", "Mateo", "Aisha", "Chen", "Farah", "Jonas")
TOPICS = ("invoices", "the offsite", "Q3 budget", "flight tickets", "the contract", "payroll")
WORDS = ("draft", "final", "notes", "backup", "report", "scan")


@dataclass(frozen=True, slots=True)
class App:
    name: str
    titles: tuple[str, ...]
    controls: tuple[tuple[str, str], ...]
    tasks: dict[str, tuple[str, ...]]  # control name -> task phrasings
    risky: frozenset[str] = frozenset()


APPS = (
    App(
        "outlook",
        ("Inbox - Outlook", "Sent Items - Outlook", "RE: {topic} - Message"),
        (
            ("Button", "New mail"), ("Button", "Reply"), ("Button", "Reply all"),
            ("Button", "Forward"), ("Button", "Delete"), ("Button", "Archive"),
            ("Button", "Flag"), ("Button", "Mark as read"), ("Button", "Move to"),
            ("Edit", "Search"), ("Button", "Calendar"), ("Button", "People"),
            ("Button", "Attach file"), ("Button", "Send"), ("Button", "Discard"),
            ("TabItem", "Home"), ("TabItem", "View"), ("TreeItem", "Inbox"),
            ("TreeItem", "Drafts"), ("TreeItem", "Sent Items"), ("TreeItem", "Junk Email"),
        ),
        {
            "New mail": ("start a new email", "compose a message to {name}", "write a fresh email"),
            "Reply": ("reply to this email", "answer {name}'s message", "respond to the sender"),
            "Reply all": ("reply to everyone on this thread", "answer the whole group"),
            "Forward": ("forward this to {name}", "pass this email on to {name}"),
            "Delete": ("delete this message", "trash this email"),
            "Archive": ("archive this email", "get this out of my inbox but keep it"),
            "Flag": ("flag this for follow-up", "mark this so I remember to act on it"),
            "Mark as read": ("mark this as read", "clear the unread marker"),
            "Search": ("search my inbox for {topic}", "find emails about {topic}"),
            "Calendar": ("open my calendar", "show my meetings this week"),
            "Attach file": ("attach the {word} file", "add an attachment"),
            "Send": ("send this email now", "send the message"),
            "Drafts": ("show my unsent drafts", "open the drafts folder"),
            "Junk Email": ("check the spam folder", "look in junk mail"),
        },
        frozenset({"Delete", "Send"}),
    ),
    App(
        "edge",
        (
            "New tab - Microsoft Edge",
            "{topic} - Search - Microsoft Edge",
            "GitHub - Microsoft Edge",
        ),
        (
            ("Button", "Back"), ("Button", "Forward"), ("Button", "Refresh"), ("Button", "Home"),
            ("Button", "New tab"), ("Button", "Close tab"), ("Button", "Downloads"),
            ("Button", "Favorites"), ("Button", "Settings and more"), ("Button", "Extensions"),
            ("Edit", "Address and search bar"), ("Button", "Add this page to favorites"),
            ("Button", "Read aloud"), ("Button", "Split screen"),
        ),
        {
            "Back": ("go back to the previous page", "return to the last page"),
            "Refresh": ("reload this page", "refresh the site"),
            "New tab": ("open a new tab", "give me a blank tab"),
            "Close tab": ("close this tab", "get rid of the current tab"),
            "Downloads": ("show my downloads", "where did my download go"),
            "Address and search bar": ("go to github.com", "search the web for {topic}"),
            "Add this page to favorites": ("bookmark this page", "save this site to favorites"),
            "Read aloud": ("read this article out loud", "have the page read to me"),
            "Extensions": ("manage my browser extensions", "install an add-on"),
            "Split screen": ("show two pages side by side",),
        },
    ),
    App(
        "vscode",
        ("{word}.py - Jevlet - Visual Studio Code", "README.md - Visual Studio Code"),
        (
            ("TabItem", "Explorer"), ("TabItem", "Search"), ("TabItem", "Source Control"),
            ("TabItem", "Run and Debug"), ("TabItem", "Extensions"),
            ("Button", "Split Editor Right"), ("Button", "Close"), ("Button", "Commit"),
            ("Button", "Run Python File"),
            ("Button", "Toggle Panel"), ("MenuItem", "File"), ("MenuItem", "Edit"),
            ("MenuItem", "Selection"), ("MenuItem", "View"), ("MenuItem", "Terminal"),
            ("Button", "Manage"), ("Button", "Accounts"),
        ),
        {
            "Source Control": ("see which files I changed", "open the git changes view"),
            "Run and Debug": ("start debugging", "set up a debug session"),
            "Extensions": ("install a new extension", "find a linter extension"),
            "Search": ("search the whole project for {word}", "find every use of {word}"),
            "Commit": ("commit my changes", "record these edits in git"),
            "Run Python File": ("run this script", "execute the current file"),
            "Split Editor Right": ("put this file side by side", "open a split view"),
            "Terminal": ("open a terminal", "give me a shell"),
            "Explorer": ("show the project files", "browse the folder tree"),
            "Toggle Panel": ("hide the bottom panel", "show the output panel"),
        },
    ),
    App(
        "explorer",
        ("Documents - File Explorer", "Downloads - File Explorer", "This PC - File Explorer"),
        (
            ("Button", "New folder"), ("Button", "Cut"), ("Button", "Copy"), ("Button", "Paste"),
            ("Button", "Rename"), ("Button", "Share"), ("Button", "Delete"), ("Button", "Sort"),
            ("Button", "View"), ("Button", "Back"), ("Button", "Up"), ("Button", "Refresh"),
            ("Edit", "Search Documents"), ("TreeItem", "Desktop"), ("TreeItem", "Downloads"),
            ("TreeItem", "Documents"), ("TreeItem", "Pictures"), ("TreeItem", "This PC"),
            ("Button", "Details pane"),
        ),
        {
            "New folder": ("make a new folder here", "create a folder for the {word} files"),
            "Rename": ("rename this file", "change the name of the selected file"),
            "Delete": ("delete the selected file", "remove this file"),
            "Copy": ("copy the selected file", "duplicate this to the clipboard"),
            "Paste": ("paste it here", "drop the copied file into this folder"),
            "Search Documents": ("find files named {word}", "search this folder for {word}"),
            "Downloads": ("go to my downloads", "open the downloads folder"),
            "Pictures": ("show my photos folder", "open pictures"),
            "Sort": ("sort these by date", "order the files by size"),
            "Share": ("share this file with {name}", "send this file to {name}"),
            "Up": ("go to the parent folder", "move up one level"),
            "Details pane": ("show file details", "see the properties of this file"),
        },
        frozenset({"Delete", "Share"}),
    ),
    App(
        "settings",
        ("Settings", "Settings - System", "Settings - Personalization"),
        (
            ("ListItem", "System"), ("ListItem", "Bluetooth & devices"),
            ("ListItem", "Network & internet"), ("ListItem", "Personalization"),
            ("ListItem", "Apps"), ("ListItem", "Accounts"), ("ListItem", "Time & language"),
            ("ListItem", "Gaming"), ("ListItem", "Accessibility"),
            ("ListItem", "Privacy & security"), ("ListItem", "Windows Update"),
            ("Edit", "Find a setting"), ("Button", "Back"),
        ),
        {
            "Bluetooth & devices": ("pair my headphones", "connect a bluetooth mouse"),
            "Network & internet": ("change the wifi network", "set up a VPN"),
            "Personalization": ("change my wallpaper", "switch to dark mode"),
            "Windows Update": ("check for updates", "see if Windows needs patching"),
            "Apps": ("uninstall a program", "see which apps start at login"),
            "Time & language": ("change the time zone", "add a keyboard language"),
            "Accessibility": ("make the text bigger", "turn on high contrast"),
            "Privacy & security": ("check which apps use my camera", "review location access"),
            "Find a setting": ("search settings for {word}", "look up the power settings"),
            "Accounts": ("add another user", "change my sign-in options"),
        },
    ),
    App(
        "word",
        ("{word}.docx - Word", "Document1 - Word"),
        (
            ("Button", "Save"), ("Button", "Undo"), ("Button", "Redo"), ("Button", "Bold"),
            ("Button", "Italic"), ("Button", "Underline"), ("Button", "Bullets"),
            ("Button", "Numbering"), ("Button", "Find"), ("Button", "Replace"),
            ("Button", "Share"), ("Button", "Comments"), ("ComboBox", "Font"),
            ("ComboBox", "Font Size"), ("TabItem", "Insert"), ("TabItem", "Layout"),
            ("TabItem", "Review"), ("Button", "Dictate"), ("Button", "Editor"),
        ),
        {
            "Bold": ("make the selected text bold", "put this in bold"),
            "Italic": ("italicize this", "make the selection italic"),
            "Bullets": ("turn this into a bulleted list", "add bullet points"),
            "Numbering": ("make this a numbered list",),
            "Find": ("find the word {word} in this document", "locate {word}"),
            "Replace": ("replace every {word} with draft", "swap one word for another everywhere"),
            "Save": ("save the document", "save my changes"),
            "Share": ("share this document with {name}", "send this doc to {name}"),
            "Comments": ("add a comment", "leave a note on this paragraph"),
            "Font Size": ("make the text bigger", "change the font size"),
            "Undo": ("undo that", "take back the last change"),
            "Dictate": ("let me type by speaking", "start dictation"),
            "Editor": ("check my grammar", "run the spelling check"),
            "Review": ("track changes", "open the review tools"),
        },
        frozenset({"Share"}),
    ),
    App(
        "excel",
        ("{word}.xlsx - Excel", "Book1 - Excel"),
        (
            ("Button", "AutoSum"), ("Button", "Sort & Filter"), ("Button", "Merge & Center"),
            ("Button", "Wrap Text"), ("Button", "Format as Table"), ("Button", "Save"),
            ("Button", "Undo"), ("Edit", "Name Box"), ("Edit", "Formula Bar"),
            ("TabItem", "Insert"), ("TabItem", "Data"), ("TabItem", "Formulas"),
            ("Button", "Recommended Charts"), ("Button", "Conditional Formatting"),
            ("Button", "Freeze Panes"),
        ),
        {
            "AutoSum": ("total this column", "add up these numbers"),
            "Sort & Filter": ("sort the rows", "filter this table"),
            "Formula Bar": ("edit the formula", "change what this cell calculates"),
            "Merge & Center": ("merge these cells", "center the title across columns"),
            "Wrap Text": ("wrap the text in this cell",),
            "Recommended Charts": ("make a chart from this data", "graph these values"),
            "Conditional Formatting": ("highlight values over 100", "color cells by value"),
            "Freeze Panes": ("keep the header row visible", "freeze the top row"),
            "Name Box": ("jump to cell B12", "go to a specific cell"),
            "Format as Table": ("turn this range into a table",),
        },
    ),
    App(
        "teams",
        ("Chat | Microsoft Teams", "Meeting with {name} | Microsoft Teams"),
        (
            ("Button", "Chat"), ("Button", "Teams"), ("Button", "Calendar"), ("Button", "Calls"),
            ("Button", "Activity"), ("Button", "New chat"), ("Button", "Meet now"),
            ("Button", "Join"), ("Button", "Leave"), ("Button", "Mute"), ("Button", "Camera"),
            ("Button", "Share"), ("Button", "Raise"), ("Edit", "Type a message"),
            ("Edit", "Search"), ("Button", "More actions"),
        ),
        {
            "Mute": ("mute my mic", "silence my microphone"),
            "Camera": ("turn my camera off", "stop my video"),
            "Share": ("share my screen", "present my desktop"),
            "Leave": ("leave the meeting", "hang up the call"),
            "Type a message": ("send a message to the team", "reply in this chat"),
            "New chat": ("start a chat with {name}", "message {name} privately"),
            "Raise": ("raise my hand", "signal that I want to speak"),
            "Calendar": ("open my meetings", "see today's schedule"),
            "Join": ("join the meeting", "hop into the call"),
            "Search": ("search chats for {topic}", "find the message about {topic}"),
        },
        frozenset({"Share"}),
    ),
    App(
        "spotify",
        ("Spotify Premium", "{name} Radio - Spotify"),
        (
            ("Button", "Play"), ("Button", "Pause"), ("Button", "Next"), ("Button", "Previous"),
            ("Button", "Enable shuffle"), ("Button", "Enable repeat"),
            ("Button", "Save to Your Library"), ("Slider", "Change volume"), ("Button", "Home"),
            ("Button", "Search"), ("Button", "Your Library"), ("Edit", "What do you want to play?"),
            ("Button", "Lyrics"), ("Button", "Queue"),
        ),
        {
            "Next": ("skip this song", "play the next track"),
            "Previous": ("play the last track again", "go back a song"),
            "Enable shuffle": ("shuffle the playlist", "mix up the song order"),
            "Save to Your Library": ("save this song", "like this track"),
            "What do you want to play?": ("find songs by {name}", "search for a jazz playlist"),
            "Change volume": ("turn the music down", "make it louder"),
            "Pause": ("pause the music", "stop playback for a moment"),
            "Lyrics": ("show the lyrics", "what are the words to this song"),
            "Queue": ("see what's playing next", "open the queue"),
        },
    ),
    App(
        "calculator",
        ("Calculator",),
        (
            ("Button", "Clear"), ("Button", "Backspace"), ("Button", "Square root"),
            ("Button", "Percent"), ("Button", "Memory store"), ("Button", "Equals"),
            ("Button", "Plus"), ("Button", "Minus"), ("Button", "Multiply by"),
            ("Button", "Divide by"), ("Button", "Open Navigation"), ("Button", "History"),
            ("Button", "Negate"), ("Button", "Reciprocal"),
        ),
        {
            "Clear": ("clear the calculator", "start the calculation over"),
            "Square root": ("take the square root",),
            "Percent": ("work out a percentage",),
            "Memory store": ("store this number", "remember this result"),
            "Open Navigation": ("switch to scientific mode", "change the calculator type"),
            "Equals": ("get the result", "finish the calculation"),
            "History": ("show previous calculations",),
            "Backspace": ("delete the last digit",),
        },
    ),
    App(
        "zoom",
        ("Zoom Meeting", "Zoom Workplace"),
        (
            ("Button", "Mute audio"), ("Button", "Start video"), ("Button", "Participants"),
            ("Button", "Chat"), ("Button", "Share Screen"), ("Button", "Record"),
            ("Button", "Reactions"), ("Button", "Leave"), ("Button", "End meeting for all"),
            ("Button", "Security"), ("Button", "Apps"),
        ),
        {
            "Mute audio": ("mute myself", "turn off my microphone"),
            "Start video": ("turn my camera on", "start my video"),
            "Participants": ("see who is in the call", "show the attendee list"),
            "Chat": ("open the meeting chat", "message everyone in the meeting"),
            "Share Screen": ("show my screen to everyone", "present my slides"),
            "Record": ("record this meeting", "start recording"),
            "Leave": ("leave this call", "drop off the meeting"),
            "End meeting for all": ("end the meeting for everyone", "close the call for all"),
            "Reactions": ("give a thumbs up", "clap for the speaker"),
        },
        frozenset({"Share Screen", "Record", "End meeting for all"}),
    ),
    App(
        "notepad",
        ("Untitled - Notepad", "{word}.txt - Notepad"),
        (
            ("MenuItem", "File"), ("MenuItem", "Edit"), ("MenuItem", "View"),
            ("Button", "Settings"), ("Edit", "Text editor"), ("TabItem", "Untitled"),
            ("Button", "Add New Tab"), ("MenuItem", "Find"), ("MenuItem", "Replace"),
            ("MenuItem", "Word wrap"), ("MenuItem", "Zoom in"),
        ),
        {
            "Text editor": ("type a note", "write something in the document"),
            "Add New Tab": ("open another note", "start a new tab"),
            "Find": ("find a word in this note",),
            "Replace": ("replace text in this note",),
            "Word wrap": ("wrap long lines", "stop lines running off the screen"),
            "Zoom in": ("make the text larger on screen",),
            "File": ("save this note", "open a text file"),
        },
    ),
)  # fmt: skip
HELD_OUT = frozenset({"teams", "spotify"})


def _fill(rng: random.Random, text: str) -> str:
    return text.format(name=rng.choice(NAMES), topic=rng.choice(TOPICS), word=rng.choice(WORDS))


def _example(rng: random.Random, apps: tuple[App, ...], split: str, index: int) -> DecisionExample:
    app = rng.choice(apps)
    foreign = rng.random() < 0.15
    if foreign:
        # A task that belongs to a different app: nothing on this screen does it.
        other = rng.choice([candidate for candidate in APPS if candidate.name != app.name])
        target_name = rng.choice(sorted(other.tasks))
        task = _fill(rng, rng.choice(other.tasks[target_name]))
        risky = target_name in other.risky
    else:
        target_name = rng.choice(sorted(app.tasks))
        task = _fill(rng, rng.choice(app.tasks[target_name]))
        risky = target_name in app.risky
    target = next((c for c in app.controls if c[1] == target_name), None) if not foreign else None
    visible = [c for c in app.controls if c != target and rng.random() < 0.75]
    visible += list(CHROME)
    if target is not None:
        visible.append(target)
    rng.shuffle(visible)
    options = list(dict.fromkeys(control_text(kind, name) for kind, name in visible))
    options.append(NONE_OF_THESE)
    if foreign and any(task_name == target_name for _, task_name in app.controls):
        # Same control name exists here (e.g. "Search"); the task still targets the other app.
        options = [option for option in options if not option.endswith(f": {target_name}")]
    label = options.index(control_text(*target)) if target is not None else len(options) - 1
    window = _fill(rng, rng.choice(app.titles))
    phrasing = rng.choice(("", "Please ", "Can you ", "I want to "))
    body = task if phrasing else task[0].upper() + task[1:]
    questions = [
        Question(GROUNDING_QUESTION, options, label, "choice", None, foreign),
        Question(RISK_QUESTION, ["True", "False"], 0 if risky else 1, "noul"),
    ]
    return DecisionExample(
        f"{split}-ground-{index}",
        daily_state(f"{phrasing}{body}", window),
        questions,
        "grounding",
        app.name,
        split,
        is_ood=app.name in HELD_OUT,
        metadata={"foreign": foreign, "target": target_name},
    )


def generate_grounding_dataset(
    output_dir: str | Path, counts: tuple[int, int] = (30_000, 3_000), seed: int = 4242
) -> dict:
    """Train on seen apps; dev on held-out apps (Teams, Spotify) to measure transfer."""
    root = Path(output_dir)
    seen = tuple(app for app in APPS if app.name not in HELD_OUT)
    unseen = tuple(app for app in APPS if app.name in HELD_OUT)
    manifest: dict = {"seed": seed, "held_out_apps": sorted(HELD_OUT), "splits": {}}
    for offset, (split, count, apps) in enumerate(
        (("train", counts[0], seen), ("dev", counts[1], unseen))
    ):
        rng = random.Random(seed + 104729 * offset)
        examples = [_example(rng, apps, split, index) for index in range(count)]
        path = root / f"{split}.jsonl"
        write_jsonl(path, examples)
        manifest["splits"][split] = {
            "count": count,
            "apps": dict(sorted(Counter(example.domain for example in examples).items())),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
