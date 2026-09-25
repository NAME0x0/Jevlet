"""Command phrasings for calendar, alarms, reminders, to-dos, notes, utilities, and system skills.

Fillers come from ``lexicon`` (composed, thousands of values per slot). Templates were written
without the ``benchmark_v2`` phrasings in view of the generator; a decontamination filter also
drops any generated command within 0.8 token overlap of a benchmark case. Times and dates stay
in the command text: the app parses them deterministically; the model learns skill and span.
"""

from __future__ import annotations

import random

from . import lexicon as lx

DAYS = ("today", "tomorrow", "on friday", "this weekend", "next week", "on monday", "tonight", "", "this week", "on sunday", "tomorrow morning", "for the rest of the day")
EXTRA_OPEN = (
    "get {a} up", "{a} please", "I need {a}", "load up {a}", "boot {a}", "pop open {a}", "can I have {a}",
    "open {a} for me", "{a}", "start up {a}", "run {a}", "launch {a} now", "I wanna use {a}", "show me {a}",
    "get me into {a}", "switch on {a}", "{a} app", "open the {a} app",
)  # fmt: skip
SYMPTOM_SETTINGS = {
    "Bluetooth and devices": ("my earbuds won't pair", "the mouse isn't connecting", "connect my speaker", "my keyboard stopped connecting", "add a bluetooth device", "controller won't connect"),
    "Wi-Fi networks": ("the internet isn't working", "join a different network", "show available networks", "forget this wifi network", "my wifi keeps dropping"),
    "Display and brightness": ("the text looks blurry", "change the screen resolution", "rotate the display", "screen is stretched", "change the scaling"),
    "Storage": ("I'm running out of space", "what's using my disk", "my drive is full", "clean up temporary files"),
    "Default apps": ("links keep opening in edge", "change which app opens pdfs", "make chrome my default", "set the default mail app"),
    "Windows Update": ("is my pc up to date", "install the latest updates", "pause updates for a week", "update history"),
    "Notifications": ("too many popups", "stop apps from bothering me", "silence notifications from teams", "notifications are annoying"),
    "Night light": ("warmer screen at night", "turn on the blue light filter", "night light settings", "reduce blue light"),
    "Sound and volume devices": ("sound comes out of the wrong speaker", "my mic isn't picked up", "switch audio to my headset", "no sound from the speakers"),
    "Power and battery": ("stop the screen turning off so fast", "battery saver settings", "change the power mode", "make the battery last longer"),
    "Printers and scanners": ("add my printer", "the printer is missing", "set up a scanner", "change the default printer"),
    "Mouse": ("the cursor is too slow", "swap the mouse buttons", "scrolling is too fast"),
    "Accessibility text size": ("everything is too small to read", "increase the font size"),
    "Focus and do not disturb": ("I need to focus, no interruptions", "turn on do not disturb"),
}  # fmt: skip


def _label(filename: str) -> str:
    return filename.rsplit(".", 1)[0].replace("_", " ")


def fill_stores(rng: random.Random) -> dict[str, list[str]]:
    """Sampled calendar, alarms, reminders, to-dos, and file search results for one example."""
    days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "today", "tomorrow")
    events = list(dict.fromkeys(
        f"{lx.event_title(rng).capitalize()} · {rng.choice(days)} {rng.randint(7, 21):02d}:{rng.choice(('00', '30'))}"
        for _ in range(rng.randint(1, 12))
    ))  # fmt: skip
    alarms = list(dict.fromkeys(
        f"{rng.randint(5, 22):02d}:{rng.choice(('00', '15', '30', '45'))} · {label.capitalize()}"
        for label in rng.sample(lx.ALARM_LABELS, rng.randint(1, 7))
    ))  # fmt: skip
    reminders = list(dict.fromkeys(
        f"{lx.task(rng).capitalize()} · {rng.choice(days)} {rng.randint(7, 21):02d}:{rng.choice(('00', '30'))}"
        for _ in range(rng.randint(1, 8))
    ))  # fmt: skip
    todos = list(dict.fromkeys(lx.task(rng) for _ in range(rng.randint(1, 12))))
    files = list(dict.fromkeys(lx.filename(rng) for _ in range(rng.randint(1, 10))))
    return {"events": events, "alarms": alarms, "reminders": reminders, "todos": todos, "files": files}


def _title(entry: str) -> str:
    return entry.split(" · ")[0].lower()


def command(rng: random.Random, skill: str, env) -> tuple[str, dict[str, str], float] | None:
    """(command, gold arguments, P(risky)) for v5/v6 skills; None for skills handled elsewhere."""
    pick = rng.choice
    if skill == "play_music":
        song = lx.music(rng)
        phrase = pick((
            "play {s}", "put on {s}", "I want to hear {s}", "queue up {s}", "throw on {s}", "blast {s}",
            "play me {s}", "can we listen to {s}", "start playing {s}", "spotify {s}", "some {s} please",
            "I'm in the mood for {s}", "play {s} on spotify", "music: {s}", "let's hear {s}",
        ))  # fmt: skip
        return phrase.format(s=song), {"text": song}, 0.02
    if skill == "brightness":
        brighter = rng.random() < 0.5
        phrases = (
            ("brighter screen", "turn up the brightness", "the screen is too dark", "increase brightness", "max brightness",
             "brightness up", "make the screen brighter", "I can't see anything, it's so dim", "brighten the display", "screen brightness 100")
            if brighter else
            ("dim the screen", "lower the brightness", "the screen is too bright", "turn the brightness down", "dimmer please",
             "brightness down", "it's blinding me", "reduce screen brightness", "make the display darker", "dim it a bit")
        )  # fmt: skip
        return pick(phrases), {"brightness": "Brighter" if brighter else "Dimmer"}, 0.02
    if skill == "radio":
        action = pick(("Wi-Fi on", "Wi-Fi off", "Bluetooth on", "Bluetooth off"))
        phrases = {
            "Wi-Fi on": ("turn wifi on", "enable wi-fi", "reconnect to wifi", "switch the wifi back on", "wifi on", "get me back online"),
            "Wi-Fi off": ("turn off wifi", "disable wi-fi", "go offline", "kill the wifi", "wifi off", "cut the internet connection"),
            "Bluetooth on": ("enable bluetooth", "switch bluetooth on", "turn on bluetooth", "bluetooth on", "activate bluetooth"),
            "Bluetooth off": ("disable bluetooth", "bluetooth off", "turn bluetooth off", "switch off bluetooth to save battery"),
        }[action]  # fmt: skip
        return pick(phrases), {"radio": action}, 0.15
    if skill == "open_file":
        target = pick(env.files)
        label = _label(target)
        words = label.split()
        partial = " ".join(words[: max(1, len(words) - 1)]) if len(words) > 1 and rng.random() < 0.5 else label
        phrase = pick((
            "open {f}", "open my {f}", "find {f} and open it", "show me the {f} file", "where's my {f}, open it",
            "pull up {f}", "open the {f} document", "I need the {f} file", "find my {f}", "bring up the {f} pdf",
            "can you open {f}", "load {f}",
        ))  # fmt: skip
        return phrase.format(f=partial), {"file": target}, 0.03
    if skill == "stopwatch":
        action = pick(("Start", "Stop", "Reset"))
        phrases = {
            "Start": ("start a stopwatch", "begin timing", "time how long this takes", "stopwatch go", "start timing me", "start counting up"),
            "Stop": ("stop the stopwatch", "stop timing now", "how long was that, stop", "end the stopwatch", "pause the stopwatch"),
            "Reset": ("reset the stopwatch", "clear the stopwatch", "zero the stopwatch", "restart the stopwatch from zero"),
        }[action]  # fmt: skip
        return pick(phrases), {"stopwatch": action}, 0.01
    if skill == "timer_control":
        action = pick(("Pause", "Resume", "Cancel", "Time left", "Add time"))
        amount = pick(("5 minutes", "2 more minutes", "10 min", "30 seconds", "a minute", "15 minutes"))
        phrases = {
            "Pause": ("pause the timer", "pause my countdown", "stop the timer for a moment", "freeze the timer", "timer pause", "put the timer on hold"),
            "Resume": ("resume the timer", "continue the countdown", "unpause the timer", "start the timer again", "keep the timer going", "timer resume"),
            "Cancel": ("cancel the timer", "stop the timer", "delete my timer", "turn off the countdown", "never mind the timer", "get rid of the timer", "end the timer"),
            "Time left": ("how much time is left", "time remaining on the timer", "how long until the timer goes off", "check my timer", "how's the timer doing", "when does my timer end"),
            "Add time": (f"add {amount} to the timer", f"give me {amount} more on the timer", f"extend the timer by {amount}", f"timer plus {amount}", f"put {amount} more on the countdown"),
        }[action]  # fmt: skip
        return pick(phrases), {"timer_action": action}, 0.02
    if skill == "set_alarm":
        time = lx.clock_time(rng)
        label = pick(lx.ALARM_LABELS) if rng.random() < 0.35 else ""
        phrase = pick((
            "set an alarm for {t}", "wake me at {t}", "alarm {t}", "{t} alarm please", "I need an alarm at {t}",
            "alarm at {t}", "get me up at {t}", "set my alarm to {t}", "new alarm {t}", "make an alarm for {t}",
            "can you set an alarm at {t}", "I have to be up by {t}", "set a {t} alarm", "alarm clock {t}",
        ))  # fmt: skip
        text = phrase.format(t=time)
        if label:
            text = pick((f"{text} for {label}", f"{text} called {label}", f"{text} labelled {label}"))
        return text, {"text": label} if label else {}, 0.02
    if skill == "list_alarms":
        return pick((
            "show my alarms", "which alarms are on", "list alarms", "do I have an alarm tomorrow", "what alarms are set",
            "when is my next alarm", "is my alarm on for the morning", "alarms", "check my alarms", "what time is my alarm",
        )), {}, 0.01  # fmt: skip
    if skill == "cancel_alarm":
        target = pick(env.alarms)
        clock, label = target.split(" · ")
        phrase = pick((
            "cancel the {l} alarm", "delete my {c} alarm", "switch off the alarm for {c}", "no {l} alarm today",
            "remove the {c} alarm", "get rid of the {l} alarm", "I don't need the {c} alarm", "turn off my {l} alarm",
            "delete the alarm called {l}", "disable the {c} alarm",
        ))  # fmt: skip
        return phrase.format(l=label.lower(), c=clock), {"alarm": target}, 0.12
    if skill == "alarm_control":
        action = pick(("Snooze", "Stop ringing"))
        phrases = {
            "Snooze": ("snooze", "snooze the alarm", "let me sleep 10 more minutes", "snooze for 5", "not yet, snooze", "give me a few more minutes", "snooze button"),
            "Stop ringing": ("stop the alarm", "dismiss the alarm", "silence the alarm", "ok I'm awake", "I'm up, stop it", "alarm off", "stop ringing", "enough, I'm up"),
        }[action]  # fmt: skip
        return pick(phrases), {"alarm_action": action}, 0.01
    if skill == "set_reminder":
        item, time = lx.task(rng), lx.when(rng)
        phrase = pick((
            "remind me to {x} {t}", "{t}, remind me to {x}", "don't let me forget to {x} {t}", "set a reminder to {x} {t}",
            "reminder to {x} {t}", "{t} remind me to {x}", "can you remind me to {x} {t}", "remind me {t} to {x}",
            "make sure I {x} {t}", "nudge me to {x} {t}", "ping me {t} to {x}", "I need a reminder to {x} {t}",
            "remind me about {x} {t}", "set a reminder {t}: {x}",
        ))  # fmt: skip
        return phrase.format(x=item, t=time).strip(), {"text": item}, 0.03
    if skill == "list_reminders":
        day = pick(DAYS)
        return pick((
            "what reminders do I have", "show my reminders", f"list reminders {day}", "did I set any reminders",
            f"what did I ask you to remind me about {day}", f"reminders for {day}", "open my reminders",
            "any reminders coming up", "what am I supposed to remember", "read me my reminders",
        )).strip(), {}, 0.01  # fmt: skip
    if skill == "cancel_reminder":
        target = pick(env.reminders)
        item = _title(target)
        phrase = pick((
            "cancel the reminder to {x}", "delete my {x} reminder", "remove the reminder about {x}",
            "no need to remind me to {x}", "stop reminding me to {x}", "I've done it, cancel the {x} reminder",
            "clear the reminder for {x}", "get rid of the reminder to {x}", "delete reminder: {x}",
        ))  # fmt: skip
        return phrase.format(x=item), {"reminder": target}, 0.1
    if skill == "create_event":
        title, time = lx.event_title(rng), lx.when(rng)
        phrase = pick((
            "add {x} {t} to my calendar", "schedule {x} {t}", "put {x} on my calendar {t}", "book {x} {t}",
            "I have {x} {t}, add it", "create an event {x} {t}", "new event: {x} {t}", "calendar: {x} {t}",
            "block out {t} for {x}", "set up {x} {t}", "add an appointment {x} {t}", "pencil in {x} {t}",
            "{t} I've got {x}, put it in the calendar", "make a calendar entry for {x} {t}",
        ))  # fmt: skip
        return phrase.format(x=title, t=time).strip(), {"text": title}, 0.05
    if skill == "show_agenda":
        day = pick(DAYS)
        phrase = pick((
            "what's on my calendar {d}", "what's my schedule {d}", "do I have anything {d}", "show my agenda {d}",
            "any meetings {d}", "what have I got on {d}", "what's next on my calendar", "open my calendar",
            "am I booked {d}", "what's my day like {d}", "my day", "what's happening {d}", "show me the events {d}",
            "when is my next meeting",
        ))  # fmt: skip
        return " ".join(phrase.format(d=day).split()), {}, 0.01
    if skill in {"move_event", "cancel_event"}:
        target = pick(env.events)
        title = _title(target)
        if skill == "move_event":
            phrase = pick((
                "move {e} to {t}", "reschedule {e} to {t}", "push {e} back to {t}", "shift {e} to {t}",
                "change {e} to {t}", "can we do {e} {t} instead", "bump {e} to {t}", "put {e} {t} instead",
                "{e} is now {t}, update it", "delay {e} until {t}",
            ))  # fmt: skip
            return phrase.format(e=title, t=lx.when(rng)), {"event": target}, 0.2
        phrase = pick((
            "cancel {e}", "remove {e} from my calendar", "delete the {e} event", "I can't make {e}, cancel it",
            "take {e} off my calendar", "{e} is off, delete it", "scrap {e}", "clear {e} from my schedule",
            "call off {e}",
        ))  # fmt: skip
        return phrase.format(e=title), {"event": target}, 0.45
    if skill == "add_todo":
        item = lx.task(rng)
        phrase = pick((
            "add {x} to my to-do list", "put {x} on my list", "todo: {x}", "I need to {x}, add it to my todos",
            "new task: {x}", "add a task to {x}", "to do {x}", "add {x} to my tasks", "stick {x} on the list",
            "task: {x}", "put {x} on my to do", "note down a task, {x}",
        ))  # fmt: skip
        return phrase.format(x=item), {"text": item}, 0.02
    if skill == "show_todos":
        return pick((
            "what's on my to-do list", "show my tasks", "what do I still need to do", "open my todo list",
            "list my to-dos", "what tasks are left", "my todo list", "what's left on my list", "show open tasks",
            "anything left to do today",
        )), {}, 0.01  # fmt: skip
    if skill == "complete_todo":
        item = pick(env.todos)
        phrase = pick((
            "mark {x} as done", "I finished {x}", "tick off {x}", "done with {x}", "{x} is done",
            "check off {x}", "complete {x}", "cross {x} off my list", "finished: {x}", "{x}, done",
        ))  # fmt: skip
        return phrase.format(x=item), {"todo": item}, 0.03
    if skill == "take_note":
        text = lx.note(rng)
        phrase = pick((
            "note that {n}", "take a note: {n}", "jot down {n}", "write down that {n}", "remember that {n}",
            "make a note: {n}", "save a note that {n}", "note {n}", "new note: {n}", "keep a note that {n}",
            "add a note saying {n}",
        ))  # fmt: skip
        return phrase.format(n=text), {"text": text}, 0.02
    if skill == "show_notes":
        return pick((
            "show my notes", "what notes do I have", "open my notes", "read my notes back", "list my notes",
            "what did I write down", "notes", "show me what I noted",
        )), {}, 0.01  # fmt: skip
    if skill == "calculate":
        expr = lx.expression(rng)
        phrase = pick(("what's {e}", "calculate {e}", "{e}", "how much is {e}", "work out {e}", "what is {e}", "{e}?", "quick maths: {e}", "compute {e}", "{e} equals"))
        return phrase.format(e=expr), {"text": expr}, 0.01
    if skill == "world_time":
        place = lx.city(rng)
        phrase = pick((
            "what time is it in {c}", "time in {c}", "what's the time over in {c}", "is it late in {c} right now",
            "clock in {c}", "{c} time", "what's the local time in {c}", "is it morning in {c} yet", "time difference with {c}",
        ))  # fmt: skip
        return phrase.format(c=place), {"text": place}, 0.01
    if skill == "weather":
        use_city = rng.random() < 0.5
        place = lx.city(rng) if use_city else ""
        day = pick(("today", "tomorrow", "this weekend", "tonight", "on friday", "this afternoon", "later", "next week", ""))
        phrase = pick((
            "what's the weather {d}", "weather {d}", "will it be sunny {d}", "what's the forecast {d}",
            "should I take a jacket {d}", "chance of rain {d}", "how's the weather looking {d}", "temperature {d}",
            "is it windy {d}", "any storms {d}", "how warm is it {d}", "will it snow {d}", "weather forecast {d}",
        ))  # fmt: skip
        text = phrase.format(d=day).strip()
        if place:
            text = pick((f"{text} in {place}", f"{place} weather {day}".strip(), f"weather in {place} {day}".strip(), f"what's it like in {place} {day}".strip()))
        return text, {"text": place} if place else {}, 0.01
    if skill == "directions":
        target = lx.place(rng)
        phrase = pick((
            "directions to {p}", "navigate to {p}", "how far is {p}", "route to {p}", "how's the traffic to {p}",
            "what's the fastest way to {p}", "get me to {p}", "show me the way to {p}", "how many km to {p}",
            "eta to {p}", "is there traffic on the way to {p}", "map to {p}", "how long to walk to {p}",
            "drive time to {p}", "best route to {p}", "how do we get over to {p}",
        ))  # fmt: skip
        return phrase.format(p=target), {"text": target}, 0.02
    if skill == "today":
        return pick((
            "what's today's date", "what's the time", "is it friday today", "which day of the week is it", "date please",
            "what time is it", "today's date", "what's the date", "time check", "is today a weekday", "what month is it",
        )), {}, 0.01  # fmt: skip
    if skill == "battery":
        return pick((
            "battery level", "how's my battery", "am I plugged in", "how long will my battery last", "battery percentage",
            "is the laptop charging", "check battery", "how much charge do I have", "battery status", "do I need to charge",
        )), {}, 0.01  # fmt: skip
    if skill == "power":
        action = pick(("Sleep", "Restart", "Shut down", "Sign out"))
        phrases = {
            "Sleep": ("put the computer to sleep", "sleep mode", "go to sleep now", "sleep the pc", "hibernate"),
            "Restart": ("restart the computer", "reboot", "restart windows", "restart my laptop", "do a restart"),
            "Shut down": ("shut down the laptop", "power off", "turn the computer off", "shutdown", "switch off the pc"),
            "Sign out": ("sign me out", "log out of windows", "log off", "sign out of my account"),
        }[action]  # fmt: skip
        return pick(phrases), {"power": action}, 0.7 if action == "Shut down" else 0.65
    if skill == "compose_email":
        topic = lx.email_topic(rng)
        recipient = lx.person(rng)
        phrase = pick((
            "draft an email about {t}", "write an email to {r} about {t}", "start a new email about {t}", "new email about {t}",
            "email {r} about {t}", "compose an email regarding {t}", "draft a message to {r} about {t}", "open a new email about {t}",
            "write to {r} about {t}", "prepare an email about {t}",
        ))  # fmt: skip
        return phrase.format(t=topic, r=recipient), {"text": topic}, 0.1
    return None


def extra_open_app(rng: random.Random, name: str) -> str:
    return rng.choice(EXTRA_OPEN).format(a=name)


def symptom_setting(rng: random.Random) -> tuple[str, str]:
    page = rng.choice(list(SYMPTOM_SETTINGS))
    return rng.choice(SYMPTOM_SETTINGS[page]), page
