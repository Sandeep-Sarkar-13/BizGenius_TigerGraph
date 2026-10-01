# -*- coding: utf-8 -*-

import os
import re
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from google import genai
from dotenv import load_dotenv


# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

GEMINI_MODEL = "gemini-3.6-flash"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

HYBRID_TOP_K = 30
RERANK_TOP_K = 20
CONTEXT_NEIGHBOURS = 1

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_CACHE = ARTIFACT_DIR / "rag_chunks.pkl"
FAISS_CACHE = ARTIFACT_DIR / "rag_faiss.index"
BM25_CACHE = ARTIFACT_DIR / "rag_bm25.pkl"

embedding_model = None
reranker = None
faiss_index = None
bm25 = None
_rag_ready = False


# ENVIRONMENT

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is not set."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# LOAD JSONL

def load_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                records.append(
                    json.loads(line)
                )

    return records

# LOAD CORPUS

documents = load_jsonl(
    CORPUS_PATH
)

print(
    "Documents:",
    len(documents)
)

# NORMALIZATION

def normalize_text(text):

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# CHUNKING

def chunk_document(
    document,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = normalize_text(
        document["text"]
    )

    words = text.split()

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk_text = " ".join(
            words[start:end]
        )

        chunks.append({

            "chunk_id":
                f'{document["doc_id"]}_{chunk_index}',

            "doc_id":
                document["doc_id"],

            "title":
                document["title"],

            "url":
                document["url"],

            "wikidata_qid":
                document["wikidata_qid"],

            "wikipedia_pageid":
                document["wikipedia_pageid"],

            "chunk_index":
                chunk_index,

            "text":
                chunk_text
        })

        chunk_index += 1

        if end >= len(words):

            break

        start = end - overlap

    return chunks


# RAG ARTIFACT INITIALIZATION

def _build_chunks():

    all_chunks_local = []

    for document in documents:
        all_chunks_local.extend(
            chunk_document(document)
        )

    return all_chunks_local


def initialize_rag(force_rebuild=False):
    """Initialize heavy RAG resources exactly once.

    First run: build chunks, embeddings, FAISS and BM25 and persist them.
    Later runs: load the persisted artifacts instead of recomputing them.
    """

    global embedding_model
    global reranker
    global faiss_index
    global bm25
    global all_chunks
    global chunk_texts
    global chunk_lookup
    global _rag_ready

    if _rag_ready:
        return

    print("Initializing RAG resources...")

    # Load/build chunks
    if CHUNKS_CACHE.exists() and not force_rebuild:
        print("Loading cached chunks...")
        with open(CHUNKS_CACHE, "rb") as f:
            all_chunks = pickle.load(f)
    else:
        print("Building chunks (one-time operation)...")
        all_chunks = _build_chunks()
        with open(CHUNKS_CACHE, "wb") as f:
            pickle.dump(all_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Chunks: {len(all_chunks):,}")

    # Load/build FAISS index
    if FAISS_CACHE.exists() and not force_rebuild:
        print("Loading cached FAISS index...")
        faiss_index = faiss.read_index(str(FAISS_CACHE))
    else:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print("Generating embeddings (one-time operation)...")
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = embedding_model.encode(
            chunk_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, str(FAISS_CACHE))

    # The embedding model is needed for query-time vector encoding.
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"FAISS vectors: {faiss_index.ntotal:,}")

    # Load/build BM25
    if BM25_CACHE.exists() and not force_rebuild:
        print("Loading cached BM25 index...")
        with open(BM25_CACHE, "rb") as f:
            bm25 = pickle.load(f)
    else:
        print("Building BM25 index...")
        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in all_chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(BM25_CACHE, "wb") as f:
            pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Reranker is also loaded only once.
    print("Loading reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    chunk_lookup = {
        (chunk["doc_id"], chunk["chunk_index"]): chunk
        for chunk in all_chunks
    }

    _rag_ready = True
    print("RAG ready.")


# Keep these globals available for type/lookup compatibility.
all_chunks = []
chunk_texts = []
chunk_lookup = {}


# TOKENIZATION

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )

# VECTOR SEARCH

def vector_search(
    query,
    top_k=30
):

    initialize_rag()

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = (
        faiss_index.search(
            query_embedding,
            top_k
        )
    )

    results = []

    for idx, score in zip(
        indices[0],
        scores[0]
    ):

        if idx < 0:
            continue

        results.append({

            "index": int(idx),

            "score":
                float(score),

            **all_chunks[idx]
        })

    return results


# BM25 SEARCH

def bm25_search(
    query,
    top_k=30
):

    initialize_rag()

    tokens = tokenize(
        query
    )

    scores = bm25.get_scores(
        tokens
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append({

            "index":
                int(idx),

            "score":
                float(scores[idx]),

            **all_chunks[idx]
        })

    return results


# RRF

def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    rrf_k=60
):

    scores = {}
    metadata = {}

    for rank, result in enumerate(
        vector_results,
        start=1
    ):

        idx = result["index"]

        scores[idx] = (
            scores.get(idx, 0)
            +
            1 / (rrf_k + rank)
        )

        metadata[idx] = result


    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        idx = result["index"]

        scores[idx] = (
            scores.get(idx, 0)
            +
            1 / (rrf_k + rank)
        )

        metadata[idx] = result


    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    results = []

    for idx, score in ranked:

        result = metadata[
            idx
        ].copy()

        result[
            "rrf_score"
        ] = score

        results.append(
            result
        )


    return results


# HYBRID SEARCH

def hybrid_search(
    query,
    vector_k=30,
    bm25_k=30,
    top_k=30
):

    vector_results = vector_search(
        query,
        vector_k
    )

    bm25_results = bm25_search(
        query,
        bm25_k
    )

    fused_results = (
        reciprocal_rank_fusion(
            vector_results,
            bm25_results
        )
    )

    return fused_results[
        :top_k
    ]


# CROSS ENCODER

def rerank_results(
    query,
    candidates,
    top_k=RERANK_TOP_K
):

    initialize_rag()

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
        batch_size=32
    )

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for candidate, score in ranked[:top_k]:
        candidate = candidate.copy()
        candidate["rerank_score"] = float(score)
        results.append(candidate)

    return results


# CONTEXT LOOKUP

def expand_context(
    retrieved_chunks,
    neighbours=1
):

    initialize_rag()

    expanded = {}

    for chunk in retrieved_chunks:

        doc_id = chunk["doc_id"]
        chunk_index = chunk["chunk_index"]

        expanded[(doc_id, chunk_index)] = chunk

        for offset in range(-neighbours, neighbours + 1):

            if offset == 0:
                continue

            key = (doc_id, chunk_index + offset)

            if key in chunk_lookup:
                expanded[key] = chunk_lookup[key]

    return list(expanded.values())


# QUESTION ANALYZER

TEMPORAL_TERMS = [
    "before",
    "after",
    "previous",
    "next",
    "earlier",
    "later",
    "immediately before",
    "immediately after",
    "preceding",
    "following"
]


SUPERLATIVE_TERMS = [
    "highest",
    "lowest",
    "largest",
    "smallest",
    "most",
    "least",
    "maximum",
    "minimum"
]


MULTI_HOP_TERMS = [
    "who won",
    "which event",
    "held at",
    "associated with",
    "related to",
    "from which"
]


AGGREGATION_TERMS = [
    "number of events",
    "how many events",
    "count of events",
    "total events",
    "average number of events",
    "sum of events"
]


def analyze_question(
    question
):

    q = question.lower()


    # Specificity-first classification

    if any(
        term in q
        for term in SUPERLATIVE_TERMS
    ):

        qtype = "superlative"


    elif any(
        term in q
        for term in TEMPORAL_TERMS
    ):

        qtype = "temporal"


    elif any(
        term in q
        for term in MULTI_HOP_TERMS
    ):

        qtype = "multi_hop"


    elif any(
        term in q
        for term in AGGREGATION_TERMS
    ):

        qtype = "aggregation"


    else:

        qtype = "lookup"


    return {
        "question_type": qtype
    }


# TEMPORAL QUERY EXPANSION

def temporal_query_expansion(
    question
):

    queries = [
        question
    ]

    years = re.findall(
        r"\b(?:19|20)\d{2}\b",
        question
    )

    if not years:

        return queries

    year = int(
        years[0]
    )

    q_lower = question.lower()


    if (
        "immediately before"
        in q_lower
        or
        "previous"
        in q_lower
    ):

        queries.append(
            question.replace(
                str(year),
                str(year - 4)
            )
        )


    if (
        "immediately after"
        in q_lower
        or
        "next"
        in q_lower
    ):

        queries.append(
            question.replace(
                str(year),
                str(year + 4)
            )
        )


    return queries


# RAG-V2 RETRIEVAL

def rag_v2_retrieve(
    question,
    hybrid_top_k=HYBRID_TOP_K,
    rerank_top_k=RERANK_TOP_K,
    neighbours=CONTEXT_NEIGHBOURS
):

    analysis = analyze_question(
        question
    )

    queries = [
        question
    ]


    if analysis[
        "question_type"
    ] == "temporal":

        queries = temporal_query_expansion(
            question
        )


    candidate_map = {}


    for query in queries:

        results = hybrid_search(
            query,
            vector_k=30,
            bm25_k=30,
            top_k=hybrid_top_k
        )


        for result in results:

            idx = result[
                "index"
            ]

            if idx not in candidate_map:

                candidate_map[
                    idx
                ] = result


    candidates = list(
        candidate_map.values()
    )


    reranked = rerank_results(
        question,
        candidates,
        top_k=rerank_top_k
    )


    expanded = expand_context(
        reranked,
        neighbours
    )


    return {
        "question":
            question,

        "analysis":
            analysis,

        "queries":
            queries,

        "reranked":
            reranked,

        "expanded_context":
            expanded
    }


# FORMAT CONTEXT

def format_context(
    chunks
):

    parts = []


    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        parts.append(
            f"""
EVIDENCE {i}

Title:
{chunk["title"]}

Document ID:
{chunk["doc_id"]}

Wikidata QID:
{chunk["wikidata_qid"]}

URL:
{chunk["url"]}

Chunk:
{chunk["chunk_index"]}

Text:
{chunk["text"]}
"""
        )


    return "\n".join(
        parts
    )


# GEMINI ANSWER GENERATION

def generate_answer(
    question,
    context
):

    prompt = f"""
You are an evidence-grounded question
answering system.

Answer the question using ONLY the
provided evidence.

Rules:

1. Give ONLY the direct final answer.
2. Do NOT explain your reasoning.
3. Do NOT summarize the evidence.
4. Do NOT list evidence items.
5. Do NOT use [E1], [E2], [E3], etc.
6. Do NOT repeat the question.
7. Do NOT mention the retrieval process.
8. Do NOT mention the RAG system.
9. Keep the answer concise.
10. If the answer is a person, return the
    person's name.
11. If the answer is a number, return the
    number with a short description.
12. If the answer is an event, return the
    event name.
13. If the evidence is insufficient, say:
    "Insufficient evidence."

QUESTION:
{question}

EVIDENCE:
{context}

FINAL ANSWER:
"""


    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )


    return response.text


# DOCUMENT-LEVEL DATA

documents_by_id = {

    doc["doc_id"]:
        doc

    for doc in documents
}


# STRUCTURED EVENT DATA

document_records = []


for doc in documents:

    title = doc[
        "title"
    ]


    olympic_match = re.search(
        r"(\d{4})\s+(Summer|Winter)\s+Olympics",
        title,
        re.IGNORECASE
    )


    year = None
    season = None


    if olympic_match:

        year = int(
            olympic_match.group(1)
        )

        season = (
            olympic_match
            .group(2)
            .lower()
        )


    sport = None


    if " at the " in title.lower():

        sport = title.split(
            " at the ",
            1
        )[0].strip()


    document_records.append({

        "doc_id":
            doc["doc_id"],

        "title":
            title,

        "url":
            doc["url"],

        "wikidata_qid":
            doc["wikidata_qid"],

        "year":
            year,

        "season":
            season,

        "sport":
            sport,

        "text":
            doc["text"]
    })


document_df = pd.DataFrame(
    document_records
)

# COMPETITOR EXTRACTION

def extract_competitors(text):

    # The corpus stores competitor count inside the Olympic event infobox. Always prioritize this explicit structured field.

    patterns = [
        r"\bcompetitors\s*:\s*(\d+)\b",
        r"\bcompetitors\s*=\s*(\d+)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return int(match.group(1))

    # Fallback for natural-language text.
    match = re.search(
        r"\b(\d+)\s+competitors\b",
        text,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


document_df[
    "competitors"
] = (
    document_df[
        "text"
    ]
    .apply(
        extract_competitors
    )
)

# EVENT CANDIDATE RETRIEVAL

def retrieve_event_candidates(
    sport,
    year,
    season
):

    candidates = document_df[

        (
            document_df[
                "sport"
            ]
            .fillna("")
            .str.lower()
            .str.strip()
            ==
            sport.lower().strip()
        )

        &

        (
            document_df[
                "year"
            ]
            == year
        )

        &

        (
            document_df[
                "season"
            ]
            == season.lower()
        )

    ].copy()


    return candidates

# AGGREGATION

def parse_aggregation_question(
    question
):

    pattern = re.compile(
        r"how many\s+(.+?)\s+events?\s+at the\s+"
        r"(\d{4})\s+(Summer|Winter)\s+Olympics"
        r".*?"
        r"(more than|less than|at least|at most|equal to)\s+"
        r"(\d+)\s+competitors",
        re.IGNORECASE
    )


    match = pattern.search(
        question
    )


    if not match:

        return None


    operator_map = {

        "more than": ">",
        "less than": "<",
        "at least": ">=",
        "at most": "<=",
        "equal to": "=="
    }


    return {

        "sport":
            match.group(1).strip(),

        "year":
            int(match.group(2)),

        "season":
            match.group(3).lower(),

        "operator":
            operator_map[
                match.group(4).lower()
            ],

        "threshold":
            int(match.group(5))
    }


def apply_operator(
    value,
    operator,
    threshold
):

    if value is None:

        return False


    if operator == ">":

        return value > threshold

    if operator == "<":

        return value < threshold

    if operator == ">=":

        return value >= threshold

    if operator == "<=":

        return value <= threshold

    if operator == "==":

        return value == threshold


    return False


def solve_aggregation(question):

    parsed = parse_aggregation_question(
        question
    )

    if parsed is None:
        return None

    candidates = retrieve_event_candidates(
        parsed["sport"],
        parsed["year"],
        parsed["season"]
    ).copy()

    if candidates.empty:
        return None

    # Make sure competitor values are numeric.
    candidates["competitors"] = pd.to_numeric(
        candidates["competitors"],
        errors="coerce"
    )

    candidates = candidates[
        candidates["competitors"].notna()
    ].copy()

    if candidates.empty:
        return None

    candidates["matches_condition"] = (
        candidates["competitors"].apply(
            lambda value: apply_operator(
                value,
                parsed["operator"],
                parsed["threshold"]
            )
        )
    )

    matching = candidates[
        candidates["matches_condition"]
    ].copy()

    return {
        "parsed": parsed,
        "candidate_documents": candidates,
        "matching_documents": matching,
        "answer": len(matching)
    }

# SUPERLATIVE

def parse_superlative_question(
    question
):

    pattern = re.compile(
        r"which\s+(.+?)\s+events?\s+at the\s+"
        r"(\d{4})\s+(Summer|Winter)\s+Olympics"
        r".*?"
        r"(highest|lowest|largest|smallest|most|least)"
        r".*?number of competitors",
        re.IGNORECASE
    )


    match = pattern.search(
        question
    )


    if not match:

        return None


    operation_text = (
        match.group(4)
        .lower()
    )


    operation = (
        "max"
        if operation_text in {
            "highest",
            "largest",
            "most"
        }
        else
        "min"
    )


    return {

        "sport":
            match.group(1).strip(),

        "year":
            int(match.group(2)),

        "season":
            match.group(3).lower(),

        "operation":
            operation
    }


def solve_superlative(
    question
):

    parsed = parse_superlative_question(
        question
    )


    if parsed is None:

        return None


    candidates = retrieve_event_candidates(
        parsed["sport"],
        parsed["year"],
        parsed["season"]
    )


    candidates = candidates[
        candidates[
            "competitors"
        ].notna()
    ].copy()


    if candidates.empty:

        return None


    if parsed[
        "operation"
    ] == "max":

        best_value = candidates[
            "competitors"
        ].max()

    else:

        best_value = candidates[
            "competitors"
        ].min()


    winners = candidates[
        candidates[
            "competitors"
        ] == best_value
    ]


    return {

        "parsed":
            parsed,

        "candidate_documents":
            candidates,

        "best_value":
            best_value,

        "winners":
            winners
    }

# FINAL RAG-V2 ROUTER

def rag_v2_route(
    question
):

    analysis = analyze_question(
        question
    )

    qtype = analysis[
        "question_type"
    ]


    # Aggregation
   
    if qtype == "aggregation":

        result = solve_aggregation(question)

        print(
            "\n[MENTAT DEBUG] Aggregation result:",
            result
        )

        if result is not None:

            return {
                "method": "structured_aggregation",
                "answer": str(result["answer"]),
                "result": result
            }

        return {
            "method": "aggregation_failed",
            "answer": "Unable to compute aggregation from structured corpus data.",
            "result": {
                "parsed": parse_aggregation_question(question)
            }
        }


    # Superlative
    
    if qtype == "superlative":

        result = solve_superlative(
            question
        )


        if result is not None:

            winners = result[
                "winners"
            ]


            if len(winners) > 0:

                answer = (
                    winners
                    .iloc[0]
                    ["title"]
                )

                answer = (
                    f"{answer} "
                    f"({result['best_value']} competitors)"
                )

            else:

                answer = (
                    "No answer found."
                )


            return {

                "method":
                    "structured_superlative",

                "answer":
                    answer,

                "result":
                    result
            }


    # Normal Hybrid RAG
    
    result = rag_v2_retrieve(
        question
    )


    context = format_context(
        result[
            "expanded_context"
        ]
    )


    answer = generate_answer(
        question,
        context
    )


    return {

        "method":
            "hybrid_rag",

        "answer":
            answer,

        "result":
            result
    }


# INTERACTIVE CLI

if __name__ == "__main__":

    print(
        "\nRAG-v2 READY"
    )

    print(
        "Type 'exit' to stop.\n"
    )


    while True:

        question = input(
            "Question: "
        ).strip()


        if question.lower() in {
            "exit",
            "quit"
        }:

            break


        if not question:

            continue


        response = rag_v2_route(
            question
        )


        print(
            "\nMETHOD:"
        )

        print(
            response[
                "method"
            ]
        )


        print(
            "\nANSWER:"
        )

        print(
            response[
                "answer"
            ]
        )


        print(
            "\n" + "-" * 80
        )
