"""Compositional vocabularies for generated commands.

v5 drew fillers from lists of 6-13 items, so 600k rows would mostly repeat. Here each filler is
composed (verb x object, activity x person, random numbers and units), giving thousands of
distinct values per slot. Words that are argument values in ``benchmark_v2`` are removed from
every pool so the held-out cases keep unseen arguments.
"""

from __future__ import annotations

import random

BLOCKED = (
    "plants", "mum", "haircut", "omar", "milk", "car insurance", "internet bill", "landlord",
    "wifi password", "parking", "tokyo", "new york", "lofi", "bohemian", "offsite", "meeting notes",
    "thesis", "holiday budget", "dentist", "standup", "sara", "gym", "pills", "plumber", "timesheet",
    "chicken", "berlin", "abu dhabi", "airport", "passport", "lisbon", "ikea", "everest", "invoice",
)  # fmt: skip


def _clean(items) -> tuple[str, ...]:
    kept = [item for item in items if not any(word in item.casefold() for word in BLOCKED)]
    return tuple(dict.fromkeys(kept))


NAMES = _clean((
    "lena", "priya", "aisha", "ben", "carlos", "chloe", "daniel", "elena", "fatima", "george",
    "hannah", "ibrahim", "jack", "jin", "karim", "laila", "lucas", "maya", "mei", "nadia", "noah",
    "olivia", "pablo", "rania", "ravi", "rosa", "sam", "sofia", "tariq", "tom", "yusuf", "zara",
    "amir", "anna", "dev", "emma", "felix", "grace", "hugo", "isla", "james", "kofi", "leo", "marta",
    "nina", "oscar", "ruth", "sean", "tina", "victor", "wei", "yara", "ahmed", "bilal", "chen",
    "diego", "farah", "hassan", "ines", "jonas", "khalid", "lucy", "mateo", "nour", "omari", "rhea",
))  # fmt: skip
RELATIONS = _clean((
    "dad", "mom", "my sister", "my brother", "grandma", "grandpa", "my aunt", "my uncle", "my boss",
    "the electrician", "the school", "the vet", "the bank", "the insurance company",
    "the pharmacy", "my manager", "the recruiter", "the tutor", "the babysitter", "the mechanic",
    "my accountant", "the council", "the estate agent", "the doctor's office", "my cousin", "my wife",
    "my husband", "my partner", "the nursery", "the hotel", "the courier", "the builder",
))  # fmt: skip
COMPANIES = ("northwind", "contoso", "fabrikam", "acme", "globex", "initech", "tailspin", "wingtip", "litware", "proseware")
CITIES = _clean((
    "london", "dubai", "sydney", "los angeles", "paris", "singapore", "toronto", "mumbai", "cairo",
    "madrid", "rome", "amsterdam", "istanbul", "seoul", "beijing", "shanghai", "hong kong", "bangkok",
    "jakarta", "manila", "karachi", "lahore", "delhi", "bangalore", "riyadh", "doha", "muscat",
    "kuwait city", "amman", "beirut", "nairobi", "lagos", "cape town", "johannesburg", "casablanca",
    "chicago", "san francisco", "seattle", "miami", "boston", "vancouver", "montreal", "mexico city",
    "sao paulo", "buenos aires", "lima", "bogota", "auckland", "melbourne", "perth", "dublin",
    "edinburgh", "manchester", "munich", "zurich", "vienna", "prague", "warsaw", "stockholm", "oslo",
    "copenhagen", "helsinki", "athens", "brussels", "barcelona", "milan", "kyoto", "osaka", "taipei",
    "kuala lumpur", "hanoi", "dhaka", "colombo", "kathmandu", "tehran", "baghdad", "sharjah", "jeddah",
))  # fmt: skip
PLACES = _clean((
    "the office", "work", "home", "the mall", "the train station", "the stadium", "the beach",
    "the city centre", "downtown", "the hospital", "the university", "the nearest pharmacy",
    "a petrol station", "the nearest gas station", "the supermarket", "the gym near me", "my parents' house",
    "grandma's house", "the hotel", "the conference centre", "the harbour", "the old town",
    "the ferry terminal", "the bus station", "the museum", "the zoo", "the park", "the library",
    "the post office", "the embassy", "the clinic", "the school", "42 king street", "7 palm avenue",
    "the marina", "the golf club", "the cinema", "the car wash", "ikea", "the vet",
)) + CITIES[:40]  # fmt: skip
ITEMS = _clean((
    "bread", "eggs", "coffee beans", "printer ink", "batteries", "a birthday card", "dog food",
    "cat litter", "light bulbs", "toothpaste", "shampoo", "a phone charger", "bin bags", "rice",
    "olive oil", "detergent", "stamps", "flowers", "sunscreen", "school supplies", "a new kettle",
    "headphones", "an hdmi cable", "painkillers", "vitamins", "nappies", "a present for {name}",
    "wrapping paper", "tea bags", "a usb stick", "new trainers", "a desk lamp", "paper towels",
    "cleaning spray", "a water filter", "cat food", "a lunchbox", "notebooks", "a raincoat",
))  # fmt: skip
BILLS = _clean(("the electricity bill", "the water bill", "the phone bill", "rent", "the credit card", "council tax", "the gas bill", "school fees", "the car loan", "the tv licence"))
DOCUMENTS = _clean((
    "the report", "the contract", "the slides", "my cv", "the proposal", "the tax form",
    "the expense claim", "the permission slip", "the lease", "the budget sheet", "the minutes",
    "the grant application", "the visa form", "the quote", "the purchase order", "my essay",
    "the lab report", "the design brief", "the press release", "the onboarding pack",
))  # fmt: skip
APPOINTMENTS = _clean((
    "a doctor's appointment", "an eye test", "a physio session", "a vet visit", "a car service",
    "a blood test", "a massage", "a nail appointment", "a parent meeting", "a boiler check",
    "a tyre change", "a bank appointment", "a therapy session", "a vaccination", "a hair appointment",
))  # fmt: skip
CHORES = _clean((
    "walk the dog", "feed the cat", "take out the bins", "put the laundry on", "empty the dishwasher",
    "defrost the freezer", "charge my laptop", "back up my phone", "move the car", "lock the back door",
    "turn off the heater", "change the bedsheets", "clean the oven", "vacuum the stairs", "stretch",
    "meditate", "drink some water", "take a break", "go for a run", "check the post", "iron a shirt",
    "hang the washing", "wipe the counters", "restock the fridge", "check the tyre pressure",
    "take my vitamins", "do my physio exercises", "practise piano", "study for the exam",
))  # fmt: skip
TASK_FORMS = (
    ("call", "person"), ("text", "person"), ("email", "person"), ("reply to", "person"),
    ("ring", "person"), ("message", "person"), ("pay", "bill"), ("buy", "item"), ("order", "item"),
    ("pick up", "item"), ("return", "item"), ("book", "appointment"), ("send", "document"),
    ("print", "document"), ("sign", "document"), ("review", "document"), ("finish", "document"),
    ("submit", "document"), ("update", "document"), ("chore", ""), ("chore", ""), ("chore", ""),
    ("thank", "person"), ("check in with", "person"), ("wish {name} happy birthday", ""),
    ("cancel", "subscription"), ("renew", "subscription"),
)  # fmt: skip
SUBSCRIPTIONS = _clean(("my netflix subscription", "the magazine subscription", "the cloud storage plan", "the domain name", "my library card", "the parking permit", "my phone contract", "the antivirus licence", "the van hire"))
EVENT_ACTIVITIES = _clean(("coffee", "lunch", "dinner", "drinks", "brunch", "a walk", "a call", "a catch-up", "a video call", "tennis", "a run", "breakfast", "a study session", "a game night"))
EVENT_FIXED = _clean((
    "project review", "team retro", "sprint planning", "budget meeting", "board meeting",
    "design review", "onboarding session", "quarterly review", "all-hands", "product demo",
    "yoga class", "piano lesson", "spin class", "swimming lesson", "driving lesson", "arabic class",
    "pottery workshop", "book club", "movie night", "football practice", "wedding rehearsal",
    "parents' evening", "car service", "eye test", "physio", "vet appointment", "blood test",
    "birthday dinner", "code review", "client workshop", "training day", "school play", "housewarming",
))  # fmt: skip
PARTIES = ("birthday party", "leaving drinks", "graduation", "wedding", "engagement party", "baby shower")
ALARM_LABELS = _clean((
    "wake up", "meds", "school run", "laundry", "pick up the kids", "leave for work", "prayer",
    "fajr", "bedtime", "workout", "take the bread out", "call the office", "feed the baby",
    "move the car", "start cooking", "stand up and stretch", "check the oven", "study", "yoga",
    "catch the train", "water break", "turn off the iron", "nap over", "start the meeting prep",
))  # fmt: skip
NOTE_FORMS = (
    "the {place} door code is {n4}", "{name}'s number is 0{n3} {n3} {n4}", "my locker number is {n2}",
    "the gate code is {n4}", "{name} likes {food}", "the meeting room is {room}",
    "the spare charger is in the {spot}", "the {thing} is in the {spot}", "{name}'s birthday is on the {nth}",
    "the booking reference is {ref}", "the tyre pressure should be {n2} psi", "{name}'s flight lands at {hh}:{mm}",
    "the recipe needs {small} eggs", "the school closes early on {day}", "the bin day is {day}",
    "the guest network is called {word}{n2}", "the delivery code is {n4}", "{name} owes me {n2} dirhams",
    "the boiler reset button is on the {side}", "the account number ends in {n4}", "table booked for {small} at {hh}",
)  # fmt: skip
FOODS = ("sushi", "spicy food", "dark chocolate", "green tea", "pizza with olives", "falafel", "mango juice", "black coffee")
SPOTS = ("kitchen drawer", "blue box", "glovebox", "top shelf", "hallway cupboard", "desk drawer", "garage", "bedside table")
THINGS = ("spare key", "passport photo", "warranty card", "first aid kit", "torch", "tape measure", "hdmi adapter")
ARTISTS = _clean((
    "taylor swift", "the weeknd", "coldplay", "daft punk", "adele", "drake", "beyonce", "ed sheeran",
    "billie eilish", "fairuz", "amr diab", "ar rahman", "arctic monkeys", "radiohead", "bts", "dua lipa",
    "kendrick lamar", "hans zimmer", "ludovico einaudi", "miles davis", "nina simone", "queen",
    "the beatles", "fleetwood mac", "bad bunny", "rosalia", "tame impala", "sza", "frank ocean",
    "mozart", "chopin", "bach", "norah jones", "burna boy", "coke studio", "nusrat fateh ali khan",
))  # fmt: skip
GENRES = ("jazz", "classical music", "hip hop", "rock", "house music", "arabic pop", "k-pop", "afrobeats", "chill piano", "ambient music", "80s hits", "country", "r&b", "bollywood songs", "reggaeton", "lo-fi hip hop", "study music", "rain sounds", "white noise", "podcast music")
MOODS = ("some", "relaxing", "upbeat", "sad", "calm", "happy", "focus", "workout", "morning", "late night", "party")
SONG_TITLES = _clean(("hotel california", "yellow", "blinding lights", "shape of you", "someone like you", "get lucky", "clair de lune", "take five", "hey jude", "levitating", "as it was", "flowers", "believer", "halo", "wonderwall"))
EMAIL_NOUNS = _clean((
    "the quarterly numbers", "tomorrow's delivery", "the rent increase", "my vacation days", "the broken printer",
    "the project deadline", "the budget", "next week's meeting", "the contract renewal", "the delayed shipment",
    "my leave request", "the website launch", "the new hire", "the team dinner", "the server outage",
    "the client feedback", "the sales figures", "the training schedule", "the office move", "the expense report",
    "the product roadmap", "the security audit", "the holiday rota", "the q4 targets", "the survey results",
    "the refund", "my order", "the school trip", "the missing parcel", "the job offer", "the interview",
    "the lease renewal", "the bike repair", "the conference trip", "the booking change",
))  # fmt: skip
FILE_TOPICS = _clean((
    "quarterly_report", "passport_scan", "wedding_photos", "lease_agreement", "budget", "presentation",
    "notes_meeting", "cv", "receipt", "family_trip", "recipe_lasagna", "insurance_claim", "bank_statement",
    "payslip", "project_plan", "research_paper", "floor_plan", "boarding_pass", "car_registration",
    "medical_report", "warranty", "school_report", "tenancy_deposit", "logo_design", "podcast_episode",
    "vacation_itinerary", "timetable", "marketing_deck", "sales_pipeline", "grant_proposal", "invoice_template",
))  # fmt: skip
FILE_EXTS = ("pdf", "docx", "xlsx", "pptx", "txt", "jpg", "png", "mp4", "zip", "csv")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
UNITS = (
    ("miles", "km"), ("km", "miles"), ("kg", "pounds"), ("pounds", "kg"), ("fahrenheit", "celsius"),
    ("celsius", "fahrenheit"), ("feet", "meters"), ("inches", "cm"), ("cm", "inches"), ("litres", "gallons"),
    ("ounces", "grams"), ("usd", "aed"), ("euros", "dollars"), ("pounds sterling", "euros"), ("aed", "inr"),
    ("cups", "ml"), ("mph", "km/h"), ("acres", "square meters"), ("hours", "minutes"), ("gb", "mb"),
)  # fmt: skip


def person(rng: random.Random) -> str:
    return rng.choice(NAMES) if rng.random() < 0.6 else rng.choice(RELATIONS)


def task(rng: random.Random) -> str:
    """Something to be reminded of or to put on a to-do list: 'call priya', 'pay the gas bill'."""
    verb, kind = rng.choice(TASK_FORMS)
    if verb == "chore":
        return rng.choice(CHORES)
    if not kind:
        return verb.format(name=rng.choice(NAMES))
    target = {
        "person": lambda: person(rng),
        "bill": lambda: rng.choice(BILLS),
        "item": lambda: rng.choice(ITEMS).format(name=rng.choice(NAMES)),
        "appointment": lambda: rng.choice(APPOINTMENTS),
        "document": lambda: rng.choice(DOCUMENTS),
        "subscription": lambda: rng.choice(SUBSCRIPTIONS),
    }[kind]()
    text = f"{verb} {target}"
    if kind == "document" and verb == "send" and rng.random() < 0.5:
        text += f" to {rng.choice(NAMES)}"
    if kind == "person" and verb in {"call", "email", "message"} and rng.random() < 0.25:
        text += f" about {rng.choice(EMAIL_NOUNS)}"
    return text


def event_title(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.35:
        return f"{rng.choice(EVENT_ACTIVITIES)} with {rng.choice(NAMES)}"
    if roll < 0.45:
        return f"{rng.choice(NAMES)}'s {rng.choice(PARTIES)}"
    if roll < 0.55:
        return f"interview with {rng.choice(COMPANIES)}"
    if roll < 0.62:
        return f"flight to {rng.choice(CITIES)}"
    if roll < 0.68:
        return f"1:1 with {rng.choice(NAMES)}"
    if roll < 0.73:
        return f"call with {rng.choice(COMPANIES)}"
    return rng.choice(EVENT_FIXED)


def note(rng: random.Random) -> str:
    return rng.choice(NOTE_FORMS).format(
        place=rng.choice(("front", "office", "garage", "building", "storage room")),
        n2=rng.randint(10, 99), n3=rng.randint(100, 999), n4=rng.randint(1000, 9999),
        name=rng.choice(NAMES), food=rng.choice(FOODS), room=rng.choice(("b12", "the blue room", "3.04", "atlas")),
        spot=rng.choice(SPOTS), thing=rng.choice(THINGS), nth=f"{rng.randint(1, 28)}th",
        ref=f"{rng.choice('ABCDEFGHJK')}{rng.choice('LMNPQRSTUV')}{rng.randint(1000, 9999)}",
        hh=rng.randint(6, 22), mm=rng.choice(("00", "15", "30", "45")), small=rng.randint(2, 8),
        day=rng.choice(WEEKDAYS), word=rng.choice(("falcon", "maple", "orbit", "cedar", "nova")),
        side=rng.choice(("left side", "right side", "bottom", "back")),
    )  # fmt: skip


def music(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.3:
        return rng.choice(ARTISTS)
    if roll < 0.5:
        return f"{rng.choice(MOODS)} {rng.choice(GENRES)}"
    if roll < 0.65:
        return rng.choice(SONG_TITLES)
    if roll < 0.75:
        return f"{rng.choice(SONG_TITLES)} by {rng.choice(ARTISTS)}"
    if roll < 0.85:
        return f"my {rng.choice(('workout', 'chill', 'driving', 'focus', 'sleep', 'running', 'cooking'))} playlist"
    return f"{rng.choice(ARTISTS)}'s latest album" if rng.random() < 0.5 else rng.choice(GENRES)


def expression(rng: random.Random) -> str:
    a, b = rng.randint(2, 999), rng.randint(2, 99)
    roll = rng.random()
    if roll < 0.35:
        op = rng.choice(("times", "plus", "minus", "divided by", "*", "+", "-", "/", "x"))
        return f"{a} {op} {b}"
    if roll < 0.55:
        return f"{rng.randint(1, 99)}{rng.choice(('%', ' percent'))} of {rng.randint(10, 5000)}"
    if roll < 0.65:
        return rng.choice((f"the square root of {a}", f"{b} squared", f"{b} to the power of {rng.randint(2, 6)}", f"{b} cubed"))
    source, target = rng.choice(UNITS)
    amount = rng.choice((str(rng.randint(1, 500)), f"{rng.randint(1, 99)}.{rng.randint(1, 9)}"))
    return rng.choice((f"{amount} {source} in {target}", f"{amount} {source} to {target}", f"convert {amount} {source} to {target}"))


def city(rng: random.Random) -> str:
    return rng.choice(CITIES)


def place(rng: random.Random) -> str:
    return rng.choice(PLACES)


def email_topic(rng: random.Random) -> str:
    return rng.choice(EMAIL_NOUNS)


def filename(rng: random.Random) -> str:
    topic = rng.choice(FILE_TOPICS)
    roll = rng.random()
    if roll < 0.15:
        return f"IMG_{rng.randint(1000, 9999)}.jpg"
    if roll < 0.22:
        return f"Screenshot {rng.randint(2024, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}.png"
    suffix = rng.choice(("", f"_{rng.randint(2023, 2026)}", "_final", f"_v{rng.randint(2, 9)}", "_draft", f"_{rng.choice(NAMES)}"))
    ext = rng.choice(FILE_EXTS)
    return f"{topic}{suffix}.{ext}"


def clock_time(rng: random.Random) -> str:
    hour = rng.randint(1, 12)
    minute = rng.choice(("00", "05", "10", "15", "20", "30", "40", "45", "50"))
    return rng.choice((
        f"{hour}:{minute}", f"{hour}am", f"{hour} am", f"{hour}pm", f"{hour} pm", f"{hour}:{minute} am",
        f"{hour}:{minute}pm", f"{rng.randint(5, 23):02d}:{minute}", f"half past {hour}", f"quarter to {hour}",
        f"quarter past {hour}", f"{hour} in the morning", f"{hour} tonight", f"{hour} tomorrow", f"{hour}.{minute}",
        "noon", "midnight", f"{hour} o'clock",
    ))  # fmt: skip


HOW_TOS = _clean((
    "fix a leaking tap", "make sourdough", "tie a tie", "change a car tyre", "remove a wine stain", "speed up my laptop",
    "write a cover letter", "cook basmati rice", "screenshot on windows", "reset my router", "clean suede shoes",
    "grow tomatoes indoors", "learn python fast", "negotiate a salary", "unclog a drain", "fold a fitted sheet",
    "boil an egg", "descale a kettle", "jump start a car", "remove a stripped screw", "center a div",
    "use git rebase", "make cold brew", "stop a cat scratching furniture", "fall asleep faster", "cite a website in apa",
))  # fmt: skip
SEARCH_THINGS = _clean((
    "laptops under 1000", "running shoes", "coffee shops", "noise cancelling headphones", "budget phones",
    "sushi", "hiking trails", "brunch spots", "standing desks", "air purifiers", "electric cars", "gaming chairs",
    "kids activities", "dog friendly cafes", "co-working spaces", "vegan restaurants",
))  # fmt: skip
DISHES = ("vegan lasagna", "chicken biryani", "shakshuka", "banana bread", "pad thai", "mac and cheese", "hummus", "lentil soup", "tiramisu", "fish tacos", "butter chicken", "pancakes")
CONCEPTS = _clean((
    "a roth ira", "quantum computing", "inflation", "machine learning", "a vpn", "retinol", "compound interest",
    "the gdp of india", "a heat pump", "blockchain", "cognitive behavioural therapy", "a mortgage offset",
    "zone 2 training", "photosynthesis", "a capsule wardrobe", "the stoics",
))  # fmt: skip
PRODUCTS = ("iphone 17", "pixel 10", "galaxy s26", "macbook air", "surface laptop", "steam deck", "kindle", "airpods pro", "ps5", "rtx 5070")
TOPICS_NEWS = ("the election", "interest rates", "the champions league", "ai regulation", "the heatwave", "oil prices", "the transfer window", "spacex", "the housing market")


def search_query(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.18:
        return f"how to {rng.choice(HOW_TOS)}"
    if roll < 0.32:
        where = rng.choice(("", " near me", f" in {rng.choice(CITIES)}", " 2026", " for beginners"))
        return f"best {rng.choice(SEARCH_THINGS)}{where}"
    if roll < 0.42:
        return f"{rng.choice(DISHES)} recipe"
    if roll < 0.52:
        return f"{rng.choice(CITIES)} {rng.choice(('things to do', 'public holidays', 'metro map', 'best time to visit', 'population', 'visa requirements'))}"
    if roll < 0.62:
        return f"what is {rng.choice(CONCEPTS)}"
    if roll < 0.72:
        a, b = rng.sample(PRODUCTS, 2)
        return rng.choice((f"{a} review", f"{a} vs {b}", f"{a} price", f"{a} release date"))
    if roll < 0.8:
        return f"news about {rng.choice(TOPICS_NEWS)}"
    if roll < 0.88:
        return f"{rng.choice(('meaning of', 'synonym for', 'how to pronounce', 'define'))} {rng.choice(('ubiquitous', 'serendipity', 'quinoa', 'ephemeral', 'gnocchi', 'epitome', 'niche'))}"
    return rng.choice((
        "cheap flights to london", "rtx a2000 driver download", "best pizza near me", "python asyncio tutorial",
        "latest cuda toolkit", "usd to aed exchange rate", "pytorch sdpa custom mask", "news about typesafe jev",
        "opening times for the post office", "who won the match last night", "train times to manchester",
    ))  # fmt: skip


def type_text(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    return rng.choice((
        "see you at 5", "thanks, sounds good", "I'll be 10 minutes late", "on my way", "can you send the file?",
        f"running {rng.randint(5, 30)} minutes late", f"sounds good, see you at {rng.randint(1, 12)}", f"thanks {name}!",
        "ok", "call me when you're free", f"{name}@{rng.choice(COMPANIES)}.com", f"{rng.randint(1, 250)} {rng.choice(('king street', 'palm avenue', 'station road', 'high street'))}",
        f"I'll send it over by {rng.choice(WEEKDAYS)}", f"happy birthday {name}!", f"can we move it to {rng.choice(WEEKDAYS)}?",
        "noted, thank you", "let me check and get back to you", "yes please", "no worries", "meeting moved to 3pm",
        f"best regards, {name}", "lgtm, merging now", "brb", "will do", f"order number {rng.randint(100000, 999999)}",
    ))  # fmt: skip


def ai_request(rng: random.Random) -> str:
    topic = rng.choice(("remote work", "rain", "the ocean", "coffee", "my cat", "autumn", "startups", "climate change", "chess"))
    return rng.choice((
        f"summarize {rng.choice(('this article', 'this pdf', 'the email thread', 'this page', 'this report'))}",
        f"write a poem about {topic}", f"write a haiku about {topic}", f"write a toast for {rng.choice(NAMES)}'s wedding",
        f"write a cover letter for a {rng.choice(('data analyst', 'nurse', 'teacher', 'product manager'))} job",
        f"explain {rng.choice(CONCEPTS)} simply", "explain this error", f"explain how {rng.choice(('vaccines', 'jet engines', 'wifi', 'credit scores'))} work",
        f"fix {rng.choice(('the bug in my regex', 'this sql query', 'my python script', 'this excel formula'))}",
        f"draft a reply to {person(rng)}", f"draft a complaint to {rng.choice(('my airline', 'the council', 'the internet provider'))}",
        f"translate this {rng.choice(('paragraph', 'email', 'message'))} to {rng.choice(('french', 'arabic', 'spanish', 'german', 'urdu', 'hindi'))}",
        f"outline a blog post about {topic}", f"outline a presentation on {topic}", f"brainstorm names for a {rng.choice(('bakery', 'podcast', 'startup', 'cat'))}",
        f"plan a 3 day trip to {rng.choice(CITIES)}", "make a weekly meal plan", "review my essay", "review this contract",
        "rewrite this more politely", "make this paragraph shorter", "make me a workout plan", "make a study timetable",
        "describe what's in this image", "check my grammar", "suggest a title for this",
    ))  # fmt: skip


DANGEROUS_VERBS = ("delete", "wipe", "erase", "remove", "permanently delete")
DANGEROUS_OBJECTS = ("photos", "emails", "messages", "contacts", "downloads", "documents", "files on the desktop", "backups", "browser history", "saved passwords")


def clarify_request(rng: random.Random) -> tuple[str, float]:
    """A request no skill should run: vague, out of reach, or dangerous. Returns (text, P(risky))."""
    roll = rng.random()
    if roll < 0.3:
        return rng.choice((
            "do the thing", "handle it", "you know what to do", "fix everything", "sort it out", "make it work",
            "the thing from before", "same as yesterday", "just do it", "whatever you think", "fix this", "help",
            "hmm", "idk", "can you help me with something", "do something", "sort my life out", "and the other one",
        )), round(rng.uniform(0.05, 0.2), 2)  # fmt: skip
    if roll < 0.45:
        return rng.choice((
            "should I take the job offer", "make me a sandwich", "is this a good idea", "walk the dog for me",
            "drive me to work", "clean my room", "fix my car", "what should I do with my life", "am I making a mistake",
            "water the garden for me", "go to the shops for me", "cook dinner",
        )), round(rng.uniform(0.05, 0.3), 2)  # fmt: skip
    if roll < 0.62:
        scope = rng.choice(("all my", "every one of my", "all of my", "my", "the"))
        return f"{rng.choice(DANGEROUS_VERBS)} {scope} {rng.choice(DANGEROUS_OBJECTS)}", round(rng.uniform(0.9, 0.99), 2)
    if roll < 0.74:
        amount = rng.choice(("500 dollars", "200 pounds", "1000 aed", "50 euros", "my rent money"))
        return rng.choice((
            f"send {amount} to {rng.choice(NAMES)}", f"transfer {amount} to {person(rng)}", f"buy {rng.choice(ITEMS).format(name=rng.choice(NAMES))} on amazon",
            f"order {rng.choice(DISHES)} for delivery", f"book me a flight to {rng.choice(CITIES)}", "buy the thing in my cart",
            f"reserve a table at {rng.choice(('the italian place', 'nobu', 'the steakhouse'))}", "order me a taxi", "renew my subscription with my card",
        )), round(rng.uniform(0.85, 0.98), 2)  # fmt: skip
    if roll < 0.86:
        return rng.choice((
            f"post {rng.choice(('this', 'my screenshot', 'that photo'))} on {rng.choice(('instagram', 'linkedin', 'twitter', 'facebook'))}",
            f"email {person(rng)} that I quit", f"reply all saying {rng.choice(('I disagree', 'this is a joke', 'count me out'))}",
            f"share my {rng.choice(('location', 'bank details', 'pin', 'login'))} with {rng.choice(NAMES)}",
            f"tell {rng.choice(NAMES)} I'm not coming", f"text {person(rng)} that I'm sick",
        )), round(rng.uniform(0.85, 0.97), 2)  # fmt: skip
    return rng.choice((
        "turn off the firewall", "disable the antivirus", "turn off windows security", "switch off automatic updates forever",
        "format the external drive", "format the usb stick", "factory reset the laptop", "uninstall chrome", "uninstall every game",
        "empty the recycle bin forever", "delete my facebook account", "run this script as admin", "disable the login password",
        "accept all the terms and conditions", "wipe my laptop", "delete system32", "give everyone access to my files",
    )), round(rng.uniform(0.8, 0.98), 2)  # fmt: skip


def when(rng: random.Random) -> str:
    """A date or time phrase for reminders and events (the app parses it, not the model)."""
    day = rng.choice((
        "", "tomorrow", "today", "tonight", f"on {rng.choice(WEEKDAYS)}", f"next {rng.choice(WEEKDAYS)}",
        f"this {rng.choice(WEEKDAYS)}", f"on the {rng.randint(1, 28)}th", "this evening", "tomorrow morning",
        "tomorrow afternoon", "next week", "on monday morning", f"on {rng.choice(WEEKDAYS)} evening",
    ))  # fmt: skip
    time = rng.choice((f"at {clock_time(rng)}", f"around {rng.randint(1, 12)}", f"by {rng.randint(1, 12)}pm", ""))
    if not day and not time:
        amount = rng.choice(("10 minutes", "20 minutes", "an hour", "2 hours", "half an hour", "3 days", "a week", "45 mins"))
        return f"in {amount}"
    return " ".join(part for part in (day, time) if part)
