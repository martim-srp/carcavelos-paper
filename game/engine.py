"""The rules of the game.

This module decides everything that is true in a game: which clues exist,
which are locked, what each action costs and whether an accusation is right.
It never calls the AI and never touches the database. The AI layer and the
UI call these functions, and the rules here have the final word.

Security rule: nothing in this module returns the text of a clue the player
has not opened, and nothing returns the solution while the game is running.
"""

import random
import re
import secrets
from dataclasses import asdict, dataclass, field, fields

from game.case import load_case, load_cracks, load_hunches, load_keyfacts, load_lies, load_voice, option_ids, resolve_variant

COLUMNS = ("who", "how", "where", "why")
HINT_COST = 15  # minutes, used when the case file's clock has no hint_cost
# Nerves (Engine.nerves): where every suspect starts, what each different card shown
# and each lie caught adds, and the top, where they crack.
NERVES_START, NERVES_PER_CARD, NERVES_PER_LIE, NERVES_MAX = 10, 30, 20, 100
ACCUSE_NOW_MINUTES = 30  # with this little time left, the next step is to accuse
# Questions the guide suggests once every card is found, before the notebook. Each answers a
# "why" that no evidence card holds (in one variant the Teaching Assistant's motive is only told
# by the Professor). The same in every variant, so it never points at the culprit.
GUIDE_QUESTIONS = (
    {"suspect": "professor", "topic": "teaching_assistant",
     "why": "The rankings explain the students. The Companion has her own secrets. But what does the "
            "Teaching Assistant get out of any of this? The Professor is their boss. Ask him about the TA."},
)
# The detective's rank at the end of a solved case, best first: (lowest score, title).
RANKS = ((85, "Chief Inspector"), (70, "Inspector"), (55, "Detective Sergeant"), (0, "Detective Constable"))
COLD_RANK = "Case gone cold"
# How a hint names a topic: "Ask the Professor about the exam."
_HINT_TOPICS = {"alibi": "their alibi", "exam": "the exam", "report": "the report", "work": "their work",
                "teaching_assistant": "the Teaching Assistant"}

# Things that look like a password or an API key. Used for the Companion's password trap.
_KEY_PATTERN = re.compile(r"\b(sk|pk|rk|ghp|gho|xoxb|xoxp)[-_][A-Za-z0-9_-]{8,}", re.IGNORECASE)
# "my password hunter2": with no "is" or ":", the next word counts only if it has a digit or a
# symbol, so "the password written on the card" stays an ordinary question.
_PASSWORD_WORDS = re.compile(
    r"\b(password|passwd|pwd|passcode|senha|palavra-passe)\s+(?=\S*[\d!@#$%^&*+=/\\|~])\S{4,}", re.IGNORECASE
)
# "password is X", "my pin: X", "pw = X": once someone says what their password IS, the value
# counts however short or weak it looks (cat, 1234, hunter2). The separator is required here,
# so "I forgot my password" and "did you pass the exam" stay ordinary questions.
_PASSWORD_GIVEN = re.compile(
    r"\b(password|passwd|pwd|passcode|passphrase|pw|pin|senha|palavra-passe)\b"
    r"((?:\s*(?:is|was|=|:|é|era))+)\s*[\"'“‘]?([^\s\"'”’?]+)",
    re.IGNORECASE,
)
# Words that can follow "password is" without being the password ("the password is in the safe").
_NOT_A_VALUE = frozenset(
    "a an the in on at under not no never what which where who whom why how that this it its my your "
    "his her their our somewhere written stored kept still also too same secret wrong right correct "
    "weak strong long short safe private missing gone lost changed different new old there here".split()
)
# A whole word that is a card number (STU-2291, staff-0417, STU-2291's). Matched against the
# whole word, so a password with a card number glued on (Lisboa2026/STU-2291) is not exempt.
_CARD_NUMBER = re.compile(r"(stu|staff)-\d{4}(['’]s)?", re.IGNORECASE)
# A file name: a stem, then a known extension. The stem is still checked for passwords.
_FILE_NAME = re.compile(r"([\w.-]+)\.(pdf|docx?|xlsx?|csv|txt|png|jpe?g)", re.IGNORECASE)
_NAME_PARTS = re.compile(r"[._-]+")
_STRIP = ".,;:!?\"'()[]"
# Field names used in the case file's logs (job=88213, card=STU-2291, ...). Only these
# key=value words count as game data; token=..., pass=... and any other assignment do not.
_GAME_FIELDS = frozenset(
    {"admin", "card", "expires", "job", "owner", "pages", "queue", "staff", "station"}
)
_FIELD = re.compile(r"^([A-Za-z_]+)=")


def _words(text):
    """The words of a text, without surrounding punctuation."""
    return [w.strip(_STRIP) for w in text.split() if w.strip(_STRIP)]


def _looks_like_password(word, known_words):
    """True if one word looks like a password: 8+ characters mixing lowercase letters and digits.

    Game data is never a password: words from the case file and card numbers. Log fields
    (job=88213) and file names (INV_Final_v3.pdf) only wrap a value, so the value inside is
    checked instead: card=STU-2291 is fine, owner=Carcavelos2026 and secret2026.txt are not.
    A file name's stem is checked part by part (report_final2.docx is "report" and "final2").
    """
    if word.lower() in known_words or _CARD_NUMBER.fullmatch(word):
        return False
    field = _FIELD.match(word)
    if field and field.group(1).lower() in _GAME_FIELDS:
        return _looks_like_password(word[field.end():], known_words)
    file_name = _FILE_NAME.fullmatch(word)
    if file_name:
        parts = _NAME_PARTS.split(file_name.group(1))
        return any(_looks_like_password(part, known_words) for part in parts if part)
    has_lower = any(c.islower() for c in word)
    has_digit = any(c.isdigit() for c in word)
    return len(word) >= 8 and has_lower and has_digit


def looks_like_secret(text, known_words=frozenset()):
    """True if the text contains something that looks like a password or an API key.

    Explicit keys (sk-..., ghp_...) and phrases like "my password is ..." always count, even
    when the value is short ("my pin is 1234", "Password is: cat").
    Otherwise any word that looks like a password counts (see _looks_like_password).
    known_words is the vocabulary of the case file.
    """
    if _KEY_PATTERN.search(text) or _PASSWORD_WORDS.search(text):
        return True
    for given in _PASSWORD_GIVEN.finditer(text):
        value = given.group(3).strip(_STRIP).lower()
        if value and value not in _NOT_A_VALUE and any(c.isalnum() for c in value):
            return True
    return any(_looks_like_password(word, known_words) for word in _words(text))


def case_vocabulary(case):
    """Every word that appears in the case file's text, lowercased.

    Players copy these words from clues into their questions, so they are never secrets.
    """
    words = set()

    def walk(value):
        if isinstance(value, str):
            words.update(w.lower() for w in _words(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(case)
    return frozenset(words)


class GameError(Exception):
    """An action the rules do not allow. The message is safe to show to the player."""


@dataclass
class Game:
    """Everything that changes during one game. Plain data, so it can be saved."""

    seed: int
    variant: str
    variables: dict
    clock: int = 0
    opened: list = field(default_factory=list)
    unlocked: list = field(default_factory=list)
    presented: list = field(default_factory=list)
    strikes: int = 0
    status: str = "playing"  # playing, solved or cold
    history: list = field(default_factory=list)
    badges: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)  # [suspect, claim_id], in the order caught
    hints: list = field(default_factory=list)  # clue ids the player bought a hint for
    cracked: list = field(default_factory=list)  # suspects whose nerves broke, in order
    mode: str = "normal"  # "normal" or "hard": hard mode gives no tips (see Engine.set_mode)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Rebuild a game from to_dict(). Keys this version does not know are ignored,
        so a save from a newer or older version of the code still loads."""
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


class Engine:
    """Applies the rules of one case file to games."""

    def __init__(self, case=None, hunches=None, voice=None, cracks=None):
        self.case = case if case is not None else load_case()
        self.hunch_lines = hunches if hunches is not None else load_hunches()
        self.voice = voice if voice is not None else load_voice()
        self.crack_lines = cracks if cracks is not None else load_cracks()
        self.keyfacts = load_keyfacts()
        self.lie_lines = load_lies()
        self.clock_rules = self.case["clock"]
        self._resolved = {v: resolve_variant(self.case, v) for v in self.case["variants"]}
        # Players see evidence as 1 to 13, in case-file order; the ids (D1, L4) stay internal.
        self._numbers = {clue_id: n for n, clue_id in enumerate(self.case["clues"], start=1)}
        self._vocabulary = case_vocabulary(self.case)

    # ---- Starting a game -------------------------------------------------

    def new_game(self, seed=None, variant=None):
        """Start a game. The seed decides the culprit and every random detail."""
        if seed is None:
            seed = secrets.randbelow(2**31)
        rng = random.Random(seed)
        variants = sorted(self.case["variants"])
        chosen = rng.choice(variants)
        if variant is not None:
            if variant not in self.case["variants"]:
                raise GameError(f"Unknown variant: {variant}")
            chosen = variant
        return Game(seed=seed, variant=chosen, variables=self._random_variables(rng))

    def _random_variables(self, rng):
        r = self.case["random"]
        cards = rng.sample(range(1000, 10000), 3)
        alias = rng.choice(r["alias"])
        initials = "".join(part[0] + "." for part in alias.replace(".", "").split())
        variables = {
            "rep_card": str(cards[0]),
            "ta_card": str(cards[1]),
            "noise_card": str(cards[2]),
            "alias": alias,
            "alias_initials": initials,
            "pastry": rng.choice(r["pastry"]),
            "weather": rng.choice(r["weather"]),
        }
        for suspect, moods in r["moods"].items():
            variables[f"mood_{suspect}"] = rng.choice(moods)
        return variables

    # ---- Reading the state ----------------------------------------------

    def _clues(self, game):
        return self._resolved[game.variant]["clues"]

    def _suspects(self, game):
        return self._resolved[game.variant]["suspects"]

    def _fill(self, game, text):
        return text.format_map(game.variables) if text else text

    def clue_state(self, game, clue_id):
        """open, available or locked."""
        clues = self._clues(game)
        if clue_id not in clues:
            raise GameError(f"There is no clue called {clue_id}.")
        if clue_id in game.opened:
            return "open"
        unlock = clues[clue_id]["unlock"]
        if unlock.get("start"):
            return "available"
        if "requires" in unlock:
            if all(c in game.opened for c in unlock["requires"]):
                return "available"
            return "locked"
        return "available" if clue_id in game.unlocked else "locked"

    def number(self, clue_id):
        """The number a player sees for a clue: 1 to 13."""
        return self._numbers[clue_id]

    def label(self, clue_id):
        """How a clue is named to the player: "Evidence 7. The faculty visitors' book"."""
        return f"Evidence {self._numbers[clue_id]}. {self.case['clues'][clue_id]['title']}"

    def clue_list(self, game):
        """Every clue with its title and state. Locked clues show no content."""
        return [
            {
                "id": clue_id,
                "number": self._numbers[clue_id],
                "title": clue["title"],
                "kind": clue["kind"],
                "state": self.clue_state(game, clue_id),
            }
            for clue_id, clue in self._clues(game).items()
        ]

    def read_clue(self, game, clue_id):
        """The full text of a clue the player has already opened. Free."""
        if self.clue_state(game, clue_id) != "open":
            raise GameError("You have not opened that clue yet.")
        clue = self._clues(game)[clue_id]
        return {
            "id": clue_id,
            "title": clue["title"],
            "kind": clue["kind"],
            "text": self._fill(game, clue.get("text")),
            "log": self._fill(game, clue.get("log")),
            "hidden_text": self._fill(game, clue.get("hidden_text")),
            "chart": clue.get("chart"),
        }

    def time_left(self, game):
        return self.clock_rules["limit_minutes"] - game.clock

    def can_afford(self, game, minutes):
        """True while the game is on and the clock has `minutes` left before 09:00. The
        interface asks this to hide moves the engine would refuse anyway."""
        return game.status == "playing" and minutes <= self.time_left(game)

    def out_of_time(self, game):
        """True once not even a question fits before 09:00: only an accusation is left."""
        return not self.can_afford(game, self.clock_rules["question_cost"])

    def night_events(self, game, since=0):
        """What happens around the building as the night goes on: story only, the same in
        every variant, so it never points at anyone and changes no rule. Returns the events
        whose time is after `since` (minutes) and no later than the clock now, oldest first,
        each as {"at", "time", "text"}."""
        result = []
        for event in self.case.get("night_events", []):
            if since < event["at"] <= game.clock:
                hours, minutes = divmod(event["at"], 60)
                result.append({"at": event["at"], "time": f"{hours:02d}:{minutes:02d}", "text": event["text"]})
        return result

    def clock_label(self, game):
        """The in-game time, starting at 00:00."""
        hours, minutes = divmod(game.clock, 60)
        return f"{hours:02d}:{minutes:02d}"

    # ---- Actions ------------------------------------------------------------

    def _spend(self, game, minutes):
        if game.status != "playing":
            raise GameError("The case is closed.")
        if minutes > self.time_left(game):
            raise GameError("Not enough time left before 09:00. Make your accusation.")
        game.clock += minutes

    def _check_suspect(self, game, suspect):
        if suspect not in self._suspects(game):
            raise GameError(f"There is no suspect called {suspect}.")

    def _award(self, game, badge_id):
        if badge_id not in game.badges:
            game.badges.append(badge_id)

    def badge_list(self, game):
        """The badges earned in this game, with their titles."""
        return [{"id": b, **self.case["badges"][b]} for b in game.badges]

    def _unlock(self, game, clue_id):
        if clue_id in game.unlocked or clue_id in game.opened:
            return []
        game.unlocked.append(clue_id)
        return [clue_id]

    def open_clue(self, game, clue_id):
        """Open an available clue. Costs clue_cost minutes."""
        state = self.clue_state(game, clue_id)
        if state == "open":
            raise GameError("That clue is already open.")
        if state == "locked":
            raise GameError("That clue is still locked.")
        self._spend(game, self.clock_rules["clue_cost"])
        game.opened.append(clue_id)
        game.history.append({"action": "open", "clue": clue_id, "clock": game.clock})
        return self.read_clue(game, clue_id)

    def ask(self, game, suspect, topic):
        """Record a question to a suspect, tagged with a topic.

        The AI layer writes the answer. It must pass the topic it detected;
        the engine checks the tag against the allowed list and decides what
        it unlocks. Returns the clues it unlocked and the scripted line for
        that topic, if there is one.
        """
        self._check_suspect(game, suspect)
        if topic not in self.case["topics"]:
            raise GameError(f"Unknown topic: {topic}")
        self._spend(game, self.clock_rules["question_cost"])
        s = self._suspects(game)[suspect]
        unlocked = []
        clue_id = s.get("topic_unlocks", {}).get(topic)
        if clue_id:
            unlocked = self._unlock(game, clue_id)
        game.history.append({"action": "ask", "suspect": suspect, "topic": topic, "clock": game.clock})
        return {"unlocked": unlocked, "line": self._fill(game, s.get("topic_lines", {}).get(topic))}

    def present(self, game, suspect, clue_id):
        """Show an opened clue to a suspect. Costs one question.

        Returns the clues it unlocked and the suspect's scripted reaction. If the
        clue breaks claims of that suspect not caught before, they are recorded
        and returned under "contradictions"; the key is left out when there are
        none, so use result.get("contradictions", []).
        """
        self._check_suspect(game, suspect)
        if self.clue_state(game, clue_id) != "open":
            raise GameError("You can only show a suspect a clue you have opened.")
        self._spend(game, self.clock_rules["question_cost"])
        if [suspect, clue_id] not in game.presented:
            game.presented.append([suspect, clue_id])
        unlocked = []
        for other_id, other in self._clues(game).items():
            rule = other["unlock"].get("present")
            if rule and rule["suspect"] == suspect and rule["clue"] == clue_id:
                unlocked += self._unlock(game, other_id)
        reveal = self._suspects(game)[suspect].get("reveals", {}).get(clue_id)
        if suspect == "professor" and clue_id == "D8":
            self._award(game, "honey_trap")
        caught = self._catch_contradictions(game, suspect, clue_id)
        event = {"action": "present", "suspect": suspect, "clue": clue_id, "clock": game.clock}
        if caught:
            event["contradictions"] = [item["claim"] for item in caught]
        cracked = self._crack(game, suspect)
        if cracked:
            event["cracked"] = True
        game.history.append(event)
        result = {"unlocked": unlocked, "reveal": self._fill(game, reveal)}
        if caught:
            result["contradictions"] = caught
        if cracked:
            result["cracked"] = cracked
        return result

    # ---- Progress -------------------------------------------------------------

    def topics_for(self, suspect):
        """The questions a player can put to a suspect. Nobody asks the TA about the TA."""
        self._check_suspect_id(suspect)
        return [t for t in self.case["topics"] if not (t == "teaching_assistant" and suspect == "ta")]

    def _check_suspect_id(self, suspect):
        if suspect not in self.case["suspects"]:
            raise GameError(f"There is no suspect called {suspect}.")

    def suspect_progress(self, game, suspect):
        """How much of a suspect you have covered, so a player knows when they are done.

        "asked" of "questions": the different questions asked. "shown" of "cards": the opened
        evidence cards already shown to them. "done" once both are complete: they have nothing
        new to say until you find another card. It comes only from what the player did,
        never from who is guilty or which card breaks a story.
        """
        self._check_suspect(game, suspect)
        topics = self.topics_for(suspect)
        asked = {e["topic"] for e in game.history if e.get("action") == "ask" and e.get("suspect") == suspect}
        opened = set(game.opened)
        shown = {clue for s, clue in game.presented if s == suspect and clue in opened}
        result = {
            "asked": len(asked & set(topics)), "questions": len(topics),
            "shown": len(shown), "cards": len(opened),
        }
        result["done"] = result["asked"] == result["questions"] and result["shown"] == result["cards"]
        return result

    # ---- Nerves ---------------------------------------------------------------

    def nerves(self, game, suspect):
        """How rattled a suspect is, 0 to 100: up with each different card shown to them,
        more with each of their lies caught. Showing the same card twice adds nothing.

        It comes only from what the player has done, never from who is guilty.
        """
        self._check_suspect(game, suspect)
        shown = sum(1 for presented_to, _clue in game.presented if presented_to == suspect)
        caught = sum(1 for s, _claim in game.contradictions if s == suspect)
        return min(NERVES_MAX, NERVES_START + NERVES_PER_CARD * shown + NERVES_PER_LIE * caught)

    def _crack(self, game, suspect):
        """At full nerves a suspect breaks, once. Returns what they blurt out, or None.

        Every suspect who cracks says the same kind of thing: they swear they did not do
        it and say where the paper really left from. The culprit cracks exactly like an
        innocent and nobody is ruled out, so cracking never shows who is guilty (showing
        cards at random would otherwise find the culprit). It gives one part of the answer.
        """
        if suspect in game.cracked or self.nerves(game, suspect) < NERVES_MAX:
            return None
        game.cracked.append(suspect)
        return {"suspect": suspect, "line": self._crack_line(game, suspect)}

    def _crack_line(self, game, suspect):
        """What a suspect blurted out when they cracked, or None if they have not cracked."""
        if suspect not in game.cracked:
            return None
        return self.crack_lines["where"][self._resolved[game.variant]["solution"]["where"]]

    def _catch_contradictions(self, game, suspect, clue_id):
        """Record each claim of the suspect that this clue breaks. Each claim counts once."""
        caught = []
        for claim in self._suspects(game)[suspect].get("claims", []):
            if clue_id in claim["broken_by"] and [suspect, claim["id"]] not in game.contradictions:
                game.contradictions.append([suspect, claim["id"]])
                caught.append(self._claim_view(game, suspect, claim, clue_id))
        return caught

    def _claim_view(self, game, suspect, claim, clue_id):
        return {
            "suspect": suspect,
            "claim": claim["id"],
            "text": self._fill(game, claim["text"]),
            "clue": clue_id,
        }

    def contradiction_list(self, game):
        """The contradictions caught so far, oldest first, with each claim's text.

        Only lists claims the player has broken, so it never gives away which
        clue breaks a claim they have not caught yet.
        """
        sources = {
            (event["suspect"], claim_id): event["clue"]
            for event in game.history
            for claim_id in event.get("contradictions", [])
        }
        result = []
        for suspect, claim_id in game.contradictions:
            claims = {c["id"]: c for c in self._suspects(game)[suspect].get("claims", [])}
            if claim_id not in claims:
                continue
            claim = claims[claim_id]
            clue_id = sources.get((suspect, claim_id))
            result.append(self._claim_view(game, suspect, claim, clue_id))
        return result

    # ---- Hints ----------------------------------------------------------------

    # ---- Mode ----------------------------------------------------------------

    MODES = ("normal", "hard")

    def hard(self, game):
        """Hard mode: the same case with no tips. No next step, no leads, no hints, no
        highlighted details, and evidence crosses nothing off the notebook for you
        (a wrong accusation still does)."""
        return getattr(game, "mode", "normal") == "hard"

    def set_mode(self, game, mode):
        """Choose normal or hard mode. Only before the first move, so it cannot be switched
        mid-case to buy a tip."""
        if mode not in self.MODES:
            raise GameError(f"Unknown mode: {mode}")
        if game.history:
            raise GameError("The mode is chosen before the first move.")
        game.mode = mode

    def hint_cost(self):
        return self.clock_rules.get("hint_cost", HINT_COST)

    def _hint_text(self, game, clue_id):
        """How to unlock a clue, in plain words, built from its unlock rule."""
        clues = self._clues(game)
        suspects = self._suspects(game)
        unlock = clues[clue_id]["unlock"]
        if "requires" in unlock:
            titles = " and ".join(self.label(c) for c in unlock["requires"])
            return f"Open {titles} first."
        if "topic" in unlock:
            return unlock["topic"].get("why") or self._ask_text(game, unlock["topic"])
        if "present" in unlock:
            rule = unlock["present"]
            name = self._lower_the(suspects[rule["suspect"]]["name"])
            return f"Show {self.label(rule['clue'])} to {name}."
        return "Keep investigating."

    def _ask_text(self, game, rule):
        """The plain instruction for a question rule: "Ask the Professor about the exam." """
        topic = _HINT_TOPICS.get(rule["topic"], rule["topic"].replace("_", " "))
        return f"Ask {self._lower_the(self._suspects(game)[rule['suspect']]['name'])} about {topic}."

    @staticmethod
    def _lower_the(name):
        return "the" + name[3:] if name.startswith("The ") else name

    def hint(self, game, clue_id):
        """Say how to unlock a locked clue. Costs hint_cost minutes the first time; free to read again."""
        if self.hard(game):
            raise GameError("There are no hints in hard mode.")
        if self.clue_state(game, clue_id) != "locked":
            raise GameError("That clue is not locked.")
        if clue_id not in game.hints:
            self._spend(game, self.hint_cost())
            game.hints.append(clue_id)
            game.history.append({"action": "hint", "clue": clue_id, "clock": game.clock})
        return self._hint_text(game, clue_id)

    def leads(self, game):
        """Free leads: what to do next with the clues the player has opened.

        For each locked clue that unlocks by showing an opened clue to a suspect,
        a lead says so. Built from the unlock rules, which are the same in every
        variant, so a lead never gives away the culprit. Returns a list of
        {"clue": locked clue id, "from": opened clue id, "text": ...}, in case order.
        None in hard mode.
        """
        if self.hard(game):
            return []
        result = []
        for clue_id, clue in self._clues(game).items():
            rule = clue["unlock"].get("present")
            if rule and rule["clue"] in game.opened and self.clue_state(game, clue_id) == "locked":
                result.append({"clue": clue_id, "from": rule["clue"], "text": self._hint_text(game, clue_id)})
        return result

    def hunches(self, game, clue_id, include_caught=False):
        """The detective's inner voice on an opened clue: a short line and the detail to highlight.

        One per suspect story this clue breaks in this variant (from the claims'
        broken_by lists), until that lie is caught. The line hints; it never names
        the liar. Returns a list of {"line", "mark"}, with the case's variables filled in.
        None in hard mode: the player has to spot the detail alone.
        """
        if self.hard(game) or self.clue_state(game, clue_id) != "open":
            return []
        result = []
        for suspect, data in self._suspects(game).items():
            for claim in data.get("claims", []):
                caught = [suspect, claim["id"]] in game.contradictions
                if clue_id not in claim["broken_by"] or (caught and not include_caught):
                    continue
                hunch = self.hunch_lines.get(f"{clue_id}|{suspect}.{claim['id']}")
                if hunch:
                    result.append({"line": self._fill(game, hunch["line"]), "mark": self._fill(game, hunch.get("mark", ""))})
        return result

    def musing(self, game, clue_id):
        """The detective's thought on an opened clue that breaks no story right now.

        Shown instead of a hunch. The same in every variant, so it never points at anyone.
        Returns a line, or None if the clue is not open or a hunch is showing.
        """
        if self.hard(game) or self.clue_state(game, clue_id) != "open" or self.hunches(game, clue_id):
            return None
        return self.voice.get("clues", {}).get(clue_id)

    def reaction(self, game, topic=None, caught=False, unlocked=False, shown=False, reacted=False):
        """A short thought after a suspect answers. Free, and no AI call.

        Picked by what just happened (a lie caught, a clue unlocked, a card the suspect
        answered or shrugged at, the question's topic, or a typed question), never by
        who is talking, so it gives
        nothing away. Rotates through the lines so it does not repeat every time.
        """
        replies = self.voice.get("replies", {})
        if caught:
            kind = "caught"
        elif unlocked:
            kind = "unlocked"
        elif shown:
            kind = "reacted" if reacted and replies.get("reacted") else "shown"
        else:
            kind = topic if topic in replies else "typed"
        lines = replies.get(kind) or []
        return lines[len(game.history) % len(lines)] if lines else None

    def thought(self, game, step):
        """The next step as the detective's thought, for the guide bar. Built from the step
        (itself built from the unlock rules), so it is the same in every variant."""
        if not step:
            return None
        lines = self.voice.get("next", {})
        if step["do"] == "search":
            return self.voice.get("search", {}).get(step["clue"]) or step["text"]
        if step.get("why"):
            return step["why"]  # the reason to ask, which already says what to ask
        if step["do"] in ("show", "ask"):
            clue = self._clues(game).get(step.get("clue"), {})
            topic = _HINT_TOPICS.get(step.get("topic"), str(step.get("topic", "")).replace("_", " "))
            name = self._suspects(game)[step["suspect"]]["name"]
            title = clue.get("title", "")
            title_l = title[4:] if title.startswith("The ") else title[:1].lower() + title[1:]
            return lines[step["do"]].format(name=name, name_l=self._lower_the(name), clue=step.get("clue", ""),
                                            title=title, title_l=title_l, topic=topic)
        if step["do"] == "accuse":
            if self.out_of_time(game) and lines.get("accuse_now"):
                return lines["accuse_now"]
            retry = any(event["action"] == "accuse" for event in game.history)
            return lines["accuse_retry" if retry and self.time_left(game) > ACCUSE_NOW_MINUTES else "accuse_time"]
        return lines.get(step["do"], step["text"])

    def key_fact(self, game, clue_id):
        """The card's one-line summary, for an open card: what it shows, never what it means.
        None in hard mode (no tips) or for a card that is not open yet."""
        if self.hard(game) or clue_id not in game.opened:
            return None
        return self.keyfacts.get(game.variant, {}).get(clue_id) or self.keyfacts["base"].get(clue_id)

    def lie_meaning(self, game, claim_id):
        """What a caught lie tells you: what is now known, and the question it opens. Never who
        is guilty, since a liar is not always the culprit. None in hard mode (no tips)."""
        if self.hard(game):
            return None
        return self.lie_lines.get(claim_id)

    def next_step(self, game):
        """The single most useful thing to do now, like a quest objective. Free.

        Built only from the unlock rules, which are the same in every variant, so it
        never points at the culprit. Returns {"do": search|ask|show|sheet|accuse,
        "text": ..., and the ids it needs: "clue", "suspect", "topic"}, or None once
        the case is closed. Order: search what is open to you, then show a card that
        opens a lead, then ask the question that opens a locked clue, then accuse.
        None in hard mode, where the player finds their own way.
        """
        if game.status != "playing" or self.hard(game):
            return None
        clues = self._clues(game)
        left = self.time_left(game)
        if self.out_of_time(game):
            return {"do": "accuse", "text": "Time is up. Make your accusation."}
        if left <= ACCUSE_NOW_MINUTES:
            return {"do": "accuse", "text": "Time is nearly up. Make your accusation."}
        if left >= self.clock_rules["clue_cost"]:
            for clue_id in clues:
                if self.clue_state(game, clue_id) == "available":
                    return {"do": "search", "clue": clue_id, "text": f"Search {self.label(clue_id)}."}
        for lead in self.leads(game):
            rule = clues[lead["clue"]]["unlock"]["present"]
            return {"do": "show", "clue": rule["clue"], "suspect": rule["suspect"], "text": lead["text"]}
        for clue_id, clue in clues.items():
            rule = clue["unlock"].get("topic")
            if rule and self.clue_state(game, clue_id) == "locked":
                step = {"do": "ask", "suspect": rule["suspect"], "topic": rule["topic"], "text": self._ask_text(game, rule)}
                if rule.get("why"):
                    step["why"] = rule["why"]
                return step
        if any(event["action"] == "accuse" for event in game.history):
            return {"do": "accuse", "text": "Change the parts that were wrong and accuse again."}
        asked = {(e.get("suspect"), e.get("topic")) for e in game.history if e["action"] == "ask"}
        for rule in GUIDE_QUESTIONS:
            if (rule["suspect"], rule["topic"]) not in asked and rule["topic"] in self.topics_for(rule["suspect"]):
                return {"do": "ask", "suspect": rule["suspect"], "topic": rule["topic"],
                        "text": self._ask_text(game, rule), "why": rule["why"]}
        return {"do": "sheet", "text": "You have every card. Cross off what can't be, then accuse."}

    def hint_list(self, game):
        """The hints bought so far for clues that are still locked, as {clue_id: text}."""
        if self.hard(game):
            return {}
        return {
            clue_id: self._hint_text(game, clue_id)
            for clue_id in game.hints
            if self.clue_state(game, clue_id) == "locked"
        }

    def accuse(self, game, who, how, where, why, proofs=None):
        """Make an accusation. Free, but three wrong ones and the case goes cold.

        proofs (optional): {column: clue id}, an opened evidence card offered as proof of
        each answer. A right proof on a solved case adds to the rank (see proven()).

        Returns how many of the four choices are right; right_parts() says which.
        """
        if game.status != "playing":
            raise GameError("The case is closed.")
        choice = {"who": who, "how": how, "where": where, "why": why}
        for column in COLUMNS:
            if choice[column] not in option_ids(self.case, column):
                raise GameError(f"That is not a valid option for {column}.")
        proofs = {column: clue for column, clue in (proofs or {}).items() if clue}
        for column, clue in proofs.items():
            if column not in COLUMNS:
                raise GameError(f"There is no {column} to prove.")
            if clue not in game.opened:
                raise GameError("You can only prove an answer with an evidence card you have opened.")
        solution = self._resolved[game.variant]["solution"]
        right = [c for c in COLUMNS if choice[c] == solution[c]]
        correct = len(right)
        if correct == len(COLUMNS):
            game.status = "solved"
            if game.clock < self.case["hedge_fund_minutes"]:
                self._award(game, "hedge_fund")
        else:
            game.strikes += 1
            if game.strikes >= self.case["max_strikes"]:
                game.status = "cold"
        event = {"action": "accuse", "correct": correct, "right": right, "choice": choice, "clock": game.clock}
        if proofs:
            event["proofs"] = proofs
        game.history.append(event)
        return {
            "correct": correct,
            "status": game.status,
            "strikes_left": self.case["max_strikes"] - game.strikes,
        }

    def right_parts(self, game):
        """Which of the four parts the latest accusation got right, in column order.

        Shown after a wrong accusation, so the player knows what to change. Empty
        if there has been no accusation yet.
        """
        for event in reversed(game.history):
            if event["action"] == "accuse":
                return list(event.get("right", []))
        return []

    def eliminated(self, game):
        """Options the evidence has already ruled out, for the case sheet and the accusation.

        Two sources, both proven: an opened clue's "rules_out" list in the case file
        ({column: [option ids]}, per variant), and the parts of a wrong accusation that
        were wrong. A crack rules nobody out (see _crack). The solution is never listed.
        Returns a list of {"column", "option", "by"}, where "by" is a clue id or
        "accusation", oldest first.
        """
        solution = self._resolved[game.variant]["solution"]
        clues = self._clues(game)
        result, seen = [], set()

        def add(column, option, by):
            if column in COLUMNS and option != solution[column] and (column, option) not in seen:
                seen.add((column, option))
                result.append({"column": column, "option": option, "by": by})

        for event in game.history:
            if event["action"] == "open" and not self.hard(game):
                for column, options in clues[event["clue"]].get("rules_out", {}).items():
                    for option in options:
                        add(column, option, event["clue"])
            elif event["action"] == "accuse" and "choice" in event:
                for column in COLUMNS:
                    if column not in event.get("right", []):
                        add(column, event["choice"][column], "accusation")
        return result

    def proven(self, game):
        """For a solved case: {column: True/False} for each answer the winning accusation
        offered a proof for (True when that evidence card proves it in this variant).
        Empty when no proof was given. Only once solved, so it never helps a guess."""
        if game.status != "solved":
            return {}
        winning = next(e for e in reversed(game.history) if e["action"] == "accuse")
        proofs = self.case["variants"][game.variant].get("proofs", {})
        return {column: clue in proofs.get(column, []) for column, clue in winning.get("proofs", {}).items()}

    def rank(self, game):
        """The detective's rank once the case is closed: {"title", "score", "parts"}.

        Out of 100. Solving gives 50; time left before 09:00 gives up to 25; each lie
        caught gives 5 (up to 20); each hint costs 3, each wrong accusation 10; each answer
        proved with the right evidence card gives 5 (up to 20, only when proofs were given).
        The score stays between 0 and 100. A cold case scores 0 and has its own title. Built
        only from what the player did, never from the variant, so it gives nothing away.
        """
        if game.status == "playing":
            raise GameError("The rank is only given when the case is closed.")
        if game.status != "solved":
            return {"title": COLD_RANK, "score": 0, "parts": {}}
        limit = self.clock_rules["limit_minutes"]
        parts = {
            "solved": 50,
            "time": round(25 * self.time_left(game) / limit),
            "lies": min(20, 5 * len(game.contradictions)),
            "hints": -3 * len(game.hints),
            "wrong": -10 * game.strikes,
        }
        proven = self.proven(game)
        if proven:
            parts["proof"] = 5 * sum(proven.values())
        score = max(0, min(100, sum(parts.values())))
        title = next(name for lowest, name in RANKS if score >= lowest)
        return {"title": title, "score": score, "parts": parts}

    def solution(self, game):
        """The answer. Only available once the game is over."""
        if game.status == "playing":
            raise GameError("The solution is only shown when the case is closed.")
        return dict(self._resolved[game.variant]["solution"])

    # ---- For the AI layer ---------------------------------------------------

    def check_secret(self, game, suspect, text):
        """Run before any player message is sent to the AI.

        Returns True if the message looks like a password or key. The AI layer must
        then not send the message anywhere. Typing one to the Companion, who asks
        for passwords, earns the badge.
        """
        self._check_suspect(game, suspect)
        if not looks_like_secret(text, self._vocabulary):
            return False
        if self._suspects(game)[suspect].get("asks_password"):
            self._award(game, "spy_password")
        return True

    def character_brief(self, game, suspect):
        """What the AI needs to play one suspect, and nothing more.

        Contains the suspect's persona, mood, cover story, topic lines and only
        the reveals the player has earned by presenting clues to them. Once they
        have cracked, "broke_down" holds what they blurted out (one part of the
        answer the player has already heard), so the AI does not deny it later;
        otherwise it is None. It never contains the rest of the solution or the
        text of any clue.
        """
        self._check_suspect(game, suspect)
        s = self._suspects(game)[suspect]
        earned = {
            clue_id: self._fill(game, s["reveals"][clue_id])
            for presented_to, clue_id in game.presented
            if presented_to == suspect and clue_id in s.get("reveals", {})
        }
        return {
            "name": s["name"],
            "persona": s["persona"],
            "mood": game.variables.get(f"mood_{suspect}"),
            "cover": self._fill(game, s["cover"]),
            "topic_lines": {k: self._fill(game, v) for k, v in s.get("topic_lines", {}).items()},
            "earned_reveals": earned,
            "broke_down": self._crack_line(game, suspect),
            "asks_password": bool(s.get("asks_password")),
        }
