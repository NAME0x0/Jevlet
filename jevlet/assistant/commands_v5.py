"""v5 command phrasings: calendar, alarms, reminders, to-dos, notes, utilities, and system.

Written independently of ``benchmark_v2``; a test rejects any benchmark command within 0.8
token overlap of 20k generated ones. Times and dates are part of the command text only: the
app parses them deterministically, so the model learns the skill and the title span.
"""

from __future__ import annotations

import random

TIMES = (
    "at 6pm", "at 7:30", "tomorrow at 9", "tomorrow morning at 8", "on friday at 4pm", "next monday at 10am",
    "tonight at 8", "at noon", "in 2 hours", "in 45 minutes", "on the 15th at 11", "this evening at 7",
    "on wednesday at 2:30", "next week on thursday at 3", "at 5:15 pm", "tomorrow afternoon at 3",
)  # fmt: skip
DAYS = ("today", "tomorrow", "on friday", "this weekend", "next week", "on monday", "tonight", "")
CLOCK_TIMES = ("6:30", "7am", "5:45", "8 am", "noon", "9:15", "6", "10pm", "11:30 tonight", "7 tomorrow")
ALARM_LABELS = ("gym", "wake up", "meds", "school run", "standup", "laundry", "pick up the kids")
REMINDER_TASKS = (
    "take out the trash", "call the bank", "send the invoice", "buy flowers for mum", "check the oven",
    "stretch", "reply to priya", "pick up the dry cleaning", "renew the domain", "drink water",
)  # fmt: skip
EVENT_TITLES = (
    "coffee with lena", "project review", "doctor's appointment", "yoga class", "call with the bank",
    "parent teacher meeting", "flight to cairo", "interview with northwind", "birthday dinner", "car service",
    "team retro", "haircut", "piano lesson",
)  # fmt: skip
TODO_ITEMS = (
    "buy batteries", "fix the leaking tap", "update my cv", "return the parcel", "clean the fridge",
    "back up my laptop", "cancel the old gym membership", "book a vet visit", "order new glasses",
)  # fmt: skip
NOTES = (
    "the spare key is under the blue pot", "locker code is 4417", "train leaves from platform 9",
    "omar prefers tea without sugar", "the router admin page is 192.168.1.1", "invoice number is INV-2291",
)  # fmt: skip
SONGS = (
    "some jazz", "taylor swift", "hotel california", "the weeknd", "chill piano music", "arabic pop",
    "my workout playlist", "coldplay yellow", "some classical music", "lo-fi hip hop", "daft punk",
)  # fmt: skip
EXPRESSIONS = (
    "15% of 80", "12 times 37", "2 to the power of 10", "the square root of 144", "250 divided by 7",
    "5 miles in km", "100 fahrenheit in celsius", "3.5 kg in pounds", "20 usd in aed", "1024 / 16",
)  # fmt: skip
CITIES = ("london", "dubai", "sydney", "los angeles", "paris", "singapore", "toronto", "mumbai", "cairo")
EMAIL_TOPICS = (
    "the quarterly numbers", "tomorrow's delivery", "the rent increase", "my vacation days",
    "the broken printer", "the project deadline",
)  # fmt: skip
FILE_POOL = (
    "quarterly_report_q2.docx", "passport_scan.pdf", "wedding_photos.zip", "lease_agreement.pdf",
    "budget_2026.xlsx", "presentation_final.pptx", "notes_meeting.txt", "cv_2026.pdf", "invoice_0913.pdf",
    "family_trip.mp4", "recipe_lasagna.docx", "insurance_claim.pdf",
)  # fmt: skip
EXTRA_OPEN = ("get {a} up", "{a} please", "I need {a}", "load up {a}", "boot {a}", "pop open {a}", "can I have {a}")
SYMPTOM_SETTINGS = {
    "Bluetooth and devices": ("my earbuds won't pair", "the mouse isn't connecting", "connect my speaker"),
    "Wi-Fi networks": ("the internet isn't working", "join a different network", "show available networks"),
    "Display and brightness": ("the text looks blurry", "change the screen resolution", "rotate the display"),
    "Storage": ("I'm running out of space", "what's using my disk"),
    "Default apps": ("links keep opening in edge", "change which app opens pdfs"),
    "Windows Update": ("is my pc up to date", "install the latest updates"),
    "Notifications": ("too many popups", "stop apps from bothering me"),
    "Night light": ("warmer screen at night", "turn on the blue light filter", "night light settings"),
    "Sound and volume devices": ("sound comes out of the wrong speaker", "my mic isn't picked up"),
    "Power and battery": ("stop the screen turning off so fast", "battery saver settings"),
    "Printers and scanners": ("add my printer", "the printer is missing"),
}  # fmt: skip


def _label(filename: str) -> str:
    return filename.rsplit(".", 1)[0].replace("_", " ")


def fill_stores(rng: random.Random) -> dict[str, list[str]]:
    """Sampled calendar, alarms, to-dos, and file search results for one training example."""
    days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    events = [
        f"{title.capitalize()} · {rng.choice(days)} {rng.randint(7, 21):02d}:{rng.choice(('00', '30'))}"
        for title in rng.sample(EVENT_TITLES, rng.randint(2, 6))
    ]
    alarms = [
        f"{rng.randint(5, 22):02d}:{rng.choice(('00', '15', '30', '45'))} · {label.capitalize()}"
        for label in rng.sample(ALARM_LABELS, rng.randint(1, 4))
    ]
    todos = rng.sample(TODO_ITEMS, rng.randint(2, 6))
    files = rng.sample(FILE_POOL, rng.randint(3, 7))
    return {"events": events, "alarms": alarms, "todos": todos, "files": files}


def command(rng: random.Random, skill: str, env) -> tuple[str, dict[str, str], float] | None:
    """(command, gold arguments, P(risky)) for v5 skills; None for skills handled elsewhere."""
    pick = rng.choice
    if skill == "play_music":
        song = pick(SONGS)
        return pick(("play {s}", "put on {s}", "I want to hear {s}", "queue up {s}", "throw on {s}", "blast {s}")).format(s=song), {"text": song}, 0.02
    if skill == "brightness":
        brighter = rng.random() < 0.5
        phrases = (
            ("brighter screen", "turn up the brightness", "the screen is too dark", "increase brightness", "max brightness")
            if brighter else
            ("dim the screen", "lower the brightness", "the screen is too bright", "turn the brightness down", "dimmer please")
        )  # fmt: skip
        return pick(phrases), {"brightness": "Brighter" if brighter else "Dimmer"}, 0.02
    if skill == "radio":
        action = pick(("Wi-Fi on", "Wi-Fi off", "Bluetooth on", "Bluetooth off"))
        phrases = {
            "Wi-Fi on": ("turn wifi on", "enable wi-fi", "reconnect to wifi", "switch the wifi back on"),
            "Wi-Fi off": ("turn off wifi", "disable wi-fi", "go offline", "kill the wifi"),
            "Bluetooth on": ("enable bluetooth", "switch bluetooth on", "turn on bluetooth"),
            "Bluetooth off": ("disable bluetooth", "bluetooth off", "turn bluetooth off"),
        }[action]
        return pick(phrases), {"radio": action}, 0.15
    if skill == "open_file":
        target = pick(env.files)
        label = _label(target)
        words = label.split()
        partial = " ".join(words[: max(1, len(words) - 1)]) if len(words) > 1 and rng.random() < 0.5 else label
        phrase = pick(("open {f}", "open my {f}", "find {f} and open it", "show me the {f} file", "where's my {f}, open it"))
        return phrase.format(f=partial), {"file": target}, 0.03
    if skill == "stopwatch":
        action = pick(("Start", "Stop", "Reset"))
        phrases = {
            "Start": ("start a stopwatch", "begin timing", "time how long this takes"),
            "Stop": ("stop the stopwatch", "stop timing now", "how long was that, stop"),
            "Reset": ("reset the stopwatch", "clear the stopwatch"),
        }[action]
        return pick(phrases), {"stopwatch": action}, 0.01
    if skill == "set_alarm":
        time = pick(CLOCK_TIMES)
        label = pick(ALARM_LABELS) if rng.random() < 0.35 else ""
        phrase = pick(("set an alarm for {t}", "wake me at {t}", "alarm {t}", "{t} alarm please", "I need an alarm at {t}"))
        text = phrase.format(t=time) + (f" for {label}" if label else "")
        return text, {"text": label} if label else {}, 0.02
    if skill == "list_alarms":
        return pick(("show my alarms", "which alarms are on", "list alarms", "do I have an alarm tomorrow")), {}, 0.01
    if skill == "cancel_alarm":
        target = pick(env.alarms)
        clock, label = target.split(" · ")
        phrase = pick(("cancel the {l} alarm", "delete my {c} alarm", "switch off the alarm for {c}", "no {l} alarm today"))
        return phrase.format(l=label.lower(), c=clock), {"alarm": target}, 0.12
    if skill == "set_reminder":
        task, time = pick(REMINDER_TASKS), pick(TIMES)
        phrase = pick(("remind me to {x} {t}", "{t}, remind me to {x}", "don't let me forget to {x} {t}", "set a reminder to {x} {t}"))
        return phrase.format(x=task, t=time), {"text": task}, 0.03
    if skill == "create_event":
        title, time = pick(EVENT_TITLES), pick(TIMES)
        phrase = pick(("add {x} {t} to my calendar", "schedule {x} {t}", "put {x} on my calendar {t}", "book {x} {t}", "I have {x} {t}, add it"))
        return phrase.format(x=title, t=time), {"text": title}, 0.05
    if skill == "show_agenda":
        day = pick(DAYS)
        phrase = pick(("what's on my calendar {d}", "what's my schedule {d}", "do I have anything {d}", "show my agenda {d}", "any meetings {d}"))
        return phrase.format(d=day).strip(), {}, 0.01
    if skill in {"move_event", "cancel_event"}:
        target = pick(env.events)
        title = target.split(" · ")[0].lower()
        if skill == "move_event":
            phrase = pick(("move {e} to {t}", "reschedule {e} to {t}", "push {e} back to {t}", "shift {e} to {t}"))
            return phrase.format(e=title, t=pick(TIMES)), {"event": target}, 0.2
        phrase = pick(("cancel {e}", "remove {e} from my calendar", "delete the {e} event", "I can't make {e}, cancel it"))
        return phrase.format(e=title), {"event": target}, 0.45
    if skill == "add_todo":
        item = pick(TODO_ITEMS)
        phrase = pick(("add {x} to my to-do list", "put {x} on my list", "todo: {x}", "I need to {x}, add it to my todos", "new task: {x}"))
        return phrase.format(x=item), {"text": item}, 0.02
    if skill == "show_todos":
        return pick(("what's on my to-do list", "show my tasks", "what do I still need to do", "open my todo list")), {}, 0.01
    if skill == "complete_todo":
        item = pick(env.todos)
        phrase = pick(("mark {x} as done", "I finished {x}", "tick off {x}", "done with {x}", "{x} is done"))
        return phrase.format(x=item), {"todo": item}, 0.03
    if skill == "take_note":
        note = pick(NOTES)
        phrase = pick(("note that {n}", "take a note: {n}", "jot down {n}", "write down that {n}", "remember that {n}"))
        return phrase.format(n=note), {"text": note}, 0.02
    if skill == "show_notes":
        return pick(("show my notes", "what notes do I have", "open my notes", "read my notes back")), {}, 0.01
    if skill == "calculate":
        expr = pick(EXPRESSIONS)
        phrase = pick(("what's {e}", "calculate {e}", "{e}", "how much is {e}", "work out {e}"))
        return phrase.format(e=expr), {"text": expr}, 0.01
    if skill == "world_time":
        city = pick(CITIES)
        phrase = pick(("what time is it in {c}", "time in {c}", "what's the time over in {c}", "is it late in {c} right now"))
        return phrase.format(c=city), {"text": city}, 0.01
    if skill == "today":
        return pick(("what's today's date", "what's the time", "is it friday today", "which day of the week is it", "date please")), {}, 0.01
    if skill == "battery":
        return pick(("battery level", "how's my battery", "am I plugged in", "how long will my battery last", "battery percentage")), {}, 0.01
    if skill == "power":
        action = pick(("Sleep", "Restart", "Shut down", "Sign out"))
        phrases = {
            "Sleep": ("put the computer to sleep", "sleep mode", "go to sleep now"),
            "Restart": ("restart the computer", "reboot", "restart windows"),
            "Shut down": ("shut down the laptop", "power off", "turn the computer off"),
            "Sign out": ("sign me out", "log out of windows", "log off"),
        }[action]
        return pick(phrases), {"power": action}, 0.7 if action == "Shut down" else 0.65
    if skill == "compose_email":
        topic = pick(EMAIL_TOPICS)
        phrase = pick(("draft an email about {t}", "write an email to priya about {t}", "start a new email about {t}", "new email about {t}"))
        return phrase.format(t=topic), {"text": topic}, 0.1
    return None


def extra_open_app(rng: random.Random, name: str) -> str:
    return rng.choice(EXTRA_OPEN).format(a=name)


def symptom_setting(rng: random.Random) -> tuple[str, str]:
    page = rng.choice(list(SYMPTOM_SETTINGS))
    return rng.choice(SYMPTOM_SETTINGS[page]), page
