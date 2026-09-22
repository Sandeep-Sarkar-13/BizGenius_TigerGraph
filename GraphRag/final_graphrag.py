import os
import re
import json
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
import pyTigerGraph
from pyTigerGraph import TigerGraphConnection
from google import genai
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = Path(
    os.getenv("CORPUS_PATH", BASE_DIR / "data" / "corpus.jsonl")
)

# This is the TigerGraph Cloud instance used by the notebook.
TG_HOST = os.getenv(
    "TG_HOST",
    "https://tg-c65a7a6e-8a3b-4f80-9678-04eb00845868.tg-2635877100.i.tgcloud.io"
)
TG_GRAPH = os.getenv(
    "TG_GRAPH",
    "Olympic_GraphRAG"
)
TG_SECRET = os.getenv("TG_SECRET")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

TG_SECRET = os.getenv("TG_SECRET", TG_SECRET)
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    GEMINI_API_KEY
)

if not TG_SECRET:
    TG_SECRET = input(
        "Enter your TigerGraph Database Secret: "
    ).strip()

if not GEMINI_API_KEY:
    GEMINI_API_KEY = input(
        "Enter your Gemini API key: "
    ).strip()


# ============================================================
# CONNECT TO EXISTING TIGERGRAPH CLOUD GRAPH
# ============================================================

print("=" * 80)
print("CONNECTING TO TIGERGRAPH CLOUD")
print("=" * 80)

conn = TigerGraphConnection(
    host=TG_HOST,
    graphname=TG_GRAPH,
    gsqlSecret=TG_SECRET
)

conn.getToken(TG_SECRET)

print("Connected to:", TG_GRAPH)
print("Host:", TG_HOST)


# ============================================================
# VERIFY EXISTING GRAPH
# ============================================================

schema = conn.getSchema(force=True)

print("\nGraph:", schema.get("GraphName"))
print(
    "Vertex types:",
    len(schema.get("VertexTypes", []))
)
print(
    "Edge types:",
    len(schema.get("EdgeTypes", []))
)

required_vertices = {
    "Document",
    "Olympic_Games",
    "Event",
    "Location"
}

required_edges = {
    "DOCUMENT_MENTIONS_GAMES",
    "DOCUMENT_MENTIONS_EVENT",
    "DOCUMENT_MENTIONS_LOCATION",
    "PART_OF",
    "HELD_AT"
}

actual_vertices = {
    v["Name"]
    for v in schema.get("VertexTypes", [])
}

actual_edges = {
    e["Name"]
    for e in schema.get("EdgeTypes", [])
}

missing_vertices = required_vertices - actual_vertices
missing_edges = required_edges - actual_edges

if missing_vertices or missing_edges:
    raise RuntimeError(
        f"Graph schema is missing vertices={missing_vertices}, "
        f"edges={missing_edges}"
    )

print("Required GraphRAG schema verified.")


# ============================================================
# LOAD CORPUS
# ============================================================

print("\nLoading corpus:", CORPUS_PATH)

if not CORPUS_PATH.exists():
    raise FileNotFoundError(
        f"Corpus not found: {CORPUS_PATH}"
    )

documents = []

with open(
    CORPUS_PATH,
    "r",
    encoding="utf-8"
) as f:
    for line in f:
        line = line.strip()
        if line:
            documents.append(json.loads(line))

corpus_df = pd.DataFrame(documents)

print(
    "Documents:",
    len(corpus_df)
)


# ============================================================
# CORPUS LOOKUP
# ============================================================

doc_lookup = corpus_df.set_index(
    corpus_df["doc_id"].astype(str)
)


# ============================================================
# INFOBOX PARSER
# ============================================================

def parse_infobox(text):
    fields = {}

    if not isinstance(text, str):
        return fields

    match = re.search(
        r"\[Infobox Olympic event\](.*?)(?=\n\n|\Z)",
        text,
        flags=re.DOTALL
    )

    if not match:
        return fields

    for line in match.group(1).splitlines():
        line = line.strip()

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip()
        value = value.strip()

        if key and value:
            fields[key] = value

    return fields


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# EVENT INDEX
# Final date-aware index used by the notebook
# ============================================================

event_rows = []

for _, row in corpus_df.iterrows():

    text = str(row["text"])
    title = str(row["title"])

    fields = parse_infobox(text)

    event_name = fields.get(
        "event",
        ""
    ).strip()

    if not event_name:
        continue

    games_value = fields.get(
        "games",
        ""
    ).strip()

    game_match = re.search(
        r"(\d{4})\s+(Summer|Winter)",
        games_value,
        flags=re.IGNORECASE
    )

    if game_match:
        year = int(
            game_match.group(1)
        )
        season = game_match.group(2)
    else:
        year = None
        season = ""

    sport_match = re.match(
        r"(.+?)\s+at the\s+\d{4}\s+"
        r"(?:Summer|Winter)\s+Olympics",
        title,
        flags=re.IGNORECASE
    )

    sport = (
        sport_match.group(1).strip()
        if sport_match
        else ""
    )

    event_rows.append({
        "event_id":
            f"event_{row['doc_id']}",
        "doc_id":
            str(row["doc_id"]),
        "event_name":
            event_name,
        "sport":
            sport,
        "games":
            games_value,
        "year":
            year,
        "season":
            season,
        "venue":
            fields.get("venue", "").strip(),
        "date":
            fields.get("date", "").strip(),
        "title":
            title
    })

event_index_df = pd.DataFrame(
    event_rows
)

print(
    "Events indexed:",
    len(event_index_df)
)


# ============================================================
# QUESTION HELPERS
# ============================================================

def extract_olympic_context(question):
    q = question.lower()

    match = re.search(
        r"(\d{4})\s+(summer|winter)\s+olympics",
        q
    )

    if match:
        return {
            "year": int(match.group(1)),
            "season": match.group(2)
        }

    match = re.search(
        r"\b(19|20)\d{2}\b",
        q
    )

    if match:
        return {
            "year": int(match.group(0)),
            "season": None
        }

    return {
        "year": None,
        "season": None
    }


def extract_location_clues(question):
    q = question.lower()

    locations = []

    patterns = [
        r"held at\s+(.+?)(?:\s+on\s+|\s+from\s+|\s+between\s+|$)",
        r"at\s+(.+?)(?:\s+on\s+|\s+from\s+|\s+between\s+|$)"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            q,
            flags=re.IGNORECASE
        )

        for match in matches:

            match = match.strip(
                " .,?"
            )

            if match and len(match) > 3:
                locations.append(match)

    stop_phrases = {
        "the event",
        "the olympics",
        "summer olympics",
        "winter olympics"
    }

    locations = [
        x for x in locations
        if x not in stop_phrases
    ]

    return list(
        dict.fromkeys(locations)
    )


# ============================================================
# DATE PARSER V8
# ============================================================

def extract_date_range_v8(question):

    q = question.lower()

    months = (
        "january|february|march|april|may|june|"
        "july|august|september|october|november|december"
    )

    # 15 to 22 August 2004
    # 15-22 August 2004
    # 15–22 August 2004

    pattern_range = rf"""
        (\d{{1,2}})
        \s*(?:to|-|–)\s*
        (\d{{1,2}})
        \s+({months})
        (?:\s+(\d{{4}}))?
    """

    match = re.search(
        pattern_range,
        q,
        flags=re.IGNORECASE | re.VERBOSE
    )

    if match:

        return {
            "start_day":
                int(match.group(1)),
            "end_day":
                int(match.group(2)),
            "month":
                match.group(3).lower(),
            "year":
                int(match.group(4))
                if match.group(4)
                else None,
            "end_year":
                int(match.group(4))
                if match.group(4)
                else None
        }

    # August 12, 2008

    pattern_month_first = rf"""
        ({months})
        \s+
        (\d{{1,2}})
        \s*,\s*
        (\d{{4}})
    """

    match = re.search(
        pattern_month_first,
        q,
        flags=re.IGNORECASE | re.VERBOSE
    )

    if match:

        return {
            "start_day":
                int(match.group(2)),
            "end_day":
                int(match.group(2)),
            "month":
                match.group(1).lower(),
            "year":
                int(match.group(3)),
            "end_year":
                int(match.group(3))
        }

    # 14 February 2010

    pattern_day_first = rf"""
        (\d{{1,2}})
        \s+
        ({months})
        \s+
        (\d{{4}})
    """

    match = re.search(
        pattern_day_first,
        q,
        flags=re.IGNORECASE | re.VERBOSE
    )

    if match:

        return {
            "start_day":
                int(match.group(1)),
            "end_day":
                int(match.group(1)),
            "month":
                match.group(2).lower(),
            "year":
                int(match.group(3)),
            "end_year":
                int(match.group(3))
        }

    return None


def enrich_question_date(question):

    question_date = extract_date_range_v8(
        question
    )

    if question_date is None:
        return None

    olympic_context = (
        extract_olympic_context(
            question
        )
    )

    olympic_year = (
        olympic_context.get("year")
    )

    if (
        question_date.get("year")
        is None
    ):
        question_date["year"] = (
            olympic_year
        )

    if (
        question_date.get("end_year")
        is None
    ):
        question_date["end_year"] = (
            olympic_year
        )

    return question_date


# ============================================================
# CALENDAR-AWARE DATE MATCHING
# ============================================================

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12
}


def date_to_ordinal(
    day,
    month,
    year
):
    import datetime

    return datetime.date(
        year,
        MONTHS[month.lower()],
        day
    ).toordinal()


def date_match_score_v7(
    question_date,
    event_date,
    event_year=None
):

    if not question_date:
        return 1.0

    if not event_date:
        return 0.0

    event_info = (
        extract_date_range_v8(
            event_date
        )
    )

    if event_info is None:
        return 0.0

    q_year = question_date.get(
        "year"
    )

    e_year = event_info.get(
        "year"
    )

    if e_year is None:
        e_year = event_year

    if q_year is not None and e_year is not None:
        if q_year != e_year:
            return 0.0

    q_end_year = (
        question_date.get(
            "end_year"
        )
        or q_year
    )

    e_end_year = (
        event_info.get(
            "end_year"
        )
        or e_year
    )

    if (
        q_year is None
        or q_end_year is None
        or e_year is None
        or e_end_year is None
    ):
        return 0.0

    try:

        q_start = date_to_ordinal(
            question_date["start_day"],
            question_date["month"],
            q_year
        )

        q_end = date_to_ordinal(
            question_date["end_day"],
            question_date["month"],
            q_end_year
        )

        e_start = date_to_ordinal(
            event_info["start_day"],
            event_info["month"],
            e_year
        )

        e_end = date_to_ordinal(
            event_info["end_day"],
            event_info["month"],
            e_end_year
        )

    except Exception:
        return 0.0

    # Exact range
    if (
        q_start == e_start
        and q_end == e_end
    ):
        return 1.0

    # Event contains requested period
    if (
        e_start <= q_start
        and e_end >= q_end
    ):
        return 0.95

    # Requested period contains event period
    if (
        q_start <= e_start
        and q_end >= e_end
    ):
        return 0.90

    # Same start or same end
    if q_start == e_start:
        return 0.75

    if q_end == e_end:
        return 0.75

    # Overlap
    overlap_start = max(
        q_start,
        e_start
    )

    overlap_end = min(
        q_end,
        e_end
    )

    if overlap_start <= overlap_end:

        overlap_days = (
            overlap_end
            - overlap_start
            + 1
        )

        q_days = (
            q_end
            - q_start
            + 1
        )

        return min(
            0.85,
            overlap_days / q_days
        )

    return 0.0


# ============================================================
# FINAL V8 EVENT RESOLVER
# ============================================================

def resolve_multihop_event_v8(
    question,
    top_k=10
):

    context = extract_olympic_context(
        question
    )

    year = context.get(
        "year"
    )

    season = context.get(
        "season"
    )

    location_clues = (
        extract_location_clues(
            question
        )
    )

    question_date = (
        enrich_question_date(
            question
        )
    )

    candidates = []

    for _, row in event_index_df.iterrows():

        # -----------------------------
        # Olympic year
        # -----------------------------

        if (
            year is not None
            and row["year"] != year
        ):
            continue

        # -----------------------------
        # Olympic season
        # -----------------------------

        if season:

            if (
                str(row["season"]).lower()
                != str(season).lower()
            ):
                continue

        # -----------------------------
        # Location
        # -----------------------------

        venue_norm = normalize_text(
            str(row["venue"])
        )

        location_score = 0.0

        for clue in location_clues:

            clue_norm = normalize_text(
                clue
            )

            if (
                "olympic" in clue_norm
                and "olympics" in clue_norm
            ):
                continue

            if clue_norm == venue_norm:

                location_score = max(
                    location_score,
                    1.0
                )

            elif (
                clue_norm in venue_norm
                or venue_norm in clue_norm
            ):

                location_score = max(
                    location_score,
                    0.9
                )

            else:

                fuzzy = (
                    fuzz.token_set_ratio(
                        clue_norm,
                        venue_norm
                    ) / 100.0
                )

                location_score = max(
                    location_score,
                    fuzzy
                )

        if location_clues and location_score == 0:
            continue

        # -----------------------------
        # Date
        # -----------------------------

        date_score = date_match_score_v7(
            question_date,
            str(row["date"]),
            event_year=row["year"]
        )

        if (
            question_date is not None
            and date_score == 0
        ):
            continue

        # -----------------------------
        # Event-name / sport similarity
        # -----------------------------

        q_norm = normalize_text(
            question
        )

        event_norm = normalize_text(
            str(row["event_name"])
        )

        sport_norm = normalize_text(
            str(row["sport"])
        )

        event_score = (
            fuzz.token_set_ratio(
                q_norm,
                event_norm
            ) / 100.0
        )

        sport_score = (
            fuzz.token_set_ratio(
                q_norm,
                sport_norm
            ) / 100.0
            if sport_norm
            else 0.0
        )

        # The final resolver prioritizes
        # explicit location/date evidence.
        final_score = (
            location_score * 100
            + date_score * 100
            + event_score * 10
            + sport_score * 5
        )

        candidates.append({

            "event_id":
                row["event_id"],

            "doc_id":
                row["doc_id"],

            "event_name":
                row["event_name"],

            "sport":
                row["sport"],

            "games":
                row["games"],

            "year":
                row["year"],

            "season":
                row["season"],

            "venue":
                row["venue"],

            "date":
                row["date"],

            "title":
                row["title"],

            "location_score":
                round(location_score, 4),

            "date_score":
                round(date_score, 4),

            "event_score":
                round(event_score, 4),

            "sport_score":
                round(sport_score, 4),

            "score":
                round(final_score, 4)
        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# TIGERGRAPH TRAVERSAL
# ============================================================

def get_event_graph_paths(
    event_id
):

    games = conn.getEdges(
        "Event",
        event_id,
        "PART_OF"
    )

    locations = conn.getEdges(
        "Event",
        event_id,
        "HELD_AT"
    )

    return {
        "event_id":
            event_id,

        "part_of":
            games,

        "held_at":
            locations
    }


# ============================================================
# SOURCE DOCUMENT
# ============================================================

def get_source_document_from_event(
    event_id
):

    doc_id = event_id.replace(
        "event_",
        "",
        1
    )

    if doc_id not in doc_lookup.index:
        return None

    return doc_lookup.loc[
        doc_id
    ]


# ============================================================
# BUILD GRAPHRAG CONTEXT
# ============================================================

def build_graph_context(
    question,
    event
):

    event_id = event[
        "event_id"
    ]

    source = (
        get_source_document_from_event(
            event_id
        )
    )

    if source is None:
        return None

    graph = get_event_graph_paths(
        event_id
    )

    fields = parse_infobox(
        source["text"]
    )

    parts = []

    parts.append(
        f"""QUESTION:
{question}"""
    )

    parts.append(
        f"""RESOLVED EVENT:
Event ID: {event_id}
Event: {event["event_name"]}
Sport: {event["sport"]}
Games: {event["games"]}
Venue: {event["venue"]}
Date: {event["date"]}"""
    )

    graph_lines = [
        "GRAPH RELATIONSHIPS:"
    ]

    for edge in graph["part_of"]:

        graph_lines.append(
            f'{edge["from_type"]}:{edge["from_id"]} '
            f'--{edge["e_type"]}--> '
            f'{edge["to_type"]}:{edge["to_id"]}'
        )

    for edge in graph["held_at"]:

        graph_lines.append(
            f'{edge["from_type"]}:{edge["from_id"]} '
            f'--{edge["e_type"]}--> '
            f'{edge["to_type"]}:{edge["to_id"]}'
        )

    parts.append(
        "\n".join(graph_lines)
    )

    parts.append(
        f"""SOURCE DOCUMENT:
Document ID: {source["doc_id"]}
Title: {source["title"]}

EVENT INFOBOX:
Event: {fields.get("event", "")}
Games: {fields.get("games", "")}
Venue: {fields.get("venue", "")}
Date: {fields.get("date", "")}
Competitors: {fields.get("competitors", "")}
Nations: {fields.get("nations", "")}
Gold: {fields.get("gold", "")}
Gold NOC: {fields.get("goldNOC", "")}"""
    )

    # Include a concise narrative source excerpt
    # around the gold-medal statement.
    text = str(source["text"])

    gold_lines = []

    for line in text.splitlines():

        if re.search(
            r"gold medal|won the gold|won gold",
            line,
            flags=re.IGNORECASE
        ):

            gold_lines.append(
                line.strip()
            )

    if gold_lines:

        parts.append(
            "SOURCE GOLD-MEDAL EVIDENCE:\n"
            + "\n".join(
                gold_lines[:5]
            )
        )

    return "\n\n".join(
        parts
    )


# ============================================================
# GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


def generate_graph_rag_answer(
    question,
    graph_context
):

    prompt = f"""
You are an evidence-grounded GraphRAG question answering system.

Answer the question using ONLY the supplied graph relationships
and source-document evidence.

RULES:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Use graph relationships to understand entity connections.
4. Use the source document to verify the actual factual answer.
5. If evidence is insufficient, clearly say so.
6. Keep the answer concise and direct.
7. Cite supporting evidence using [E1], [E2], etc.
8. Directly answer the question.
9. Do not return JSON.

QUESTION:
{question}

GRAPH + SOURCE EVIDENCE:
{graph_context}

FINAL ANSWER:
"""

    response = (
        gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
    )

    return response.text


# ============================================================
# END-TO-END GRAPHRAG
# ============================================================

def graph_rag(
    question,
    generate_answer=True
):

    # 1. Resolve event
    candidates = (
        resolve_multihop_event_v8(
            question,
            top_k=10
        )
    )

    if not candidates:

        return {
            "status":
                "not_found",

            "question":
                question,

            "answer":
                "I could not identify a matching Olympic event.",

            "candidates":
                []
        }

    best = candidates[0]

    # Avoid presenting an arbitrary event as
    # confidently resolved when evidence is tied.
    if len(candidates) > 1:

        gap = (
            candidates[0]["score"]
            - candidates[1]["score"]
        )

        if gap < 0.01:

            return {
                "status":
                    "ambiguous",

                "question":
                    question,

                "answer":
                    "Multiple Olympic events match the supplied "
                    "location/date evidence.",

                "candidates":
                    candidates
            }

    # 2. Build graph + source evidence
    context = build_graph_context(
        question,
        best
    )

    if context is None:

        return {
            "status":
                "no_source_document",

            "question":
                question,

            "event":
                best
        }

    # 3. Gemini answer
    answer = None

    if generate_answer:

        answer = (
            generate_graph_rag_answer(
                question,
                context
            )
        )

    # 4. Return complete trace
    return {

        "status":
            "success",

        "question":
            question,

        "event":
            best,

        "event_id":
            best["event_id"],

        "graph":
            get_event_graph_paths(
                best["event_id"]
            ),

        "source_document": {
            "doc_id":
                best["doc_id"],

            "title":
                best["title"],

            "url":
                str(
                    doc_lookup.loc[
                        best["doc_id"]
                    ]["url"]
                )
        },

        "context":
            context,

        "candidates":
            candidates,

        "answer":
            answer
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("TIGERGRAPH CLOUD + GEMINI GRAPHRAG READY")
    print("=" * 80)

    while True:

        question = input(
            "\nQuestion (or 'exit'): "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:
            break

        if not question:
            continue

        result = graph_rag(
            question,
            generate_answer=True
        )

        print("\nSTATUS:")
        print(
            result.get(
                "status"
            )
        )

        print("\nANSWER:")
        print(
            result.get(
                "answer"
            )
        )

        if result.get("event"):

            print("\nRESOLVED EVENT:")
            print(
                json.dumps(
                    result["event"],
                    indent=2,
                    ensure_ascii=False
                )
            )

        if result.get("graph"):

            print("\nGRAPH PATHS:")

            for edge in (
                result["graph"]
                .get("part_of", [])
            ):

                print(
                    f'{edge["from_type"]}:{edge["from_id"]} '
                    f'--{edge["e_type"]}--> '
                    f'{edge["to_type"]}:{edge["to_id"]}'
                )

            for edge in (
                result["graph"]
                .get("held_at", [])
            ):

                print(
                    f'{edge["from_type"]}:{edge["from_id"]} '
                    f'--{edge["e_type"]}--> '
                    f'{edge["to_type"]}:{edge["to_id"]}'
                )
