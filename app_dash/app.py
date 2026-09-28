# ============================================================
# MENTAT
# Multi-agent Evidence & Neural Thinking for Analysis & Truth
# Main Dash Application
# ============================================================

import os

from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
    State,
    no_update,
    ctx
)

from dotenv import load_dotenv
import importlib
import time
import re

# ENVIRONMENT

load_dotenv()

# PIPELINE CACHE

_pipeline_cache = {}


def load_pipeline(name):

    if name in _pipeline_cache:
        return _pipeline_cache[name]

    if name == "RAG":
        module = importlib.import_module("final_rag")

    elif name == "GraphRAG":
        module = importlib.import_module("final_graphrag")

    elif name == "Agentic GraphRAG":
        module = importlib.import_module("agentic_graphrag")

    else:
        raise ValueError(
            f"Unknown pipeline: {name}"
        )

    _pipeline_cache[name] = module

    return module

# MENTAT QUESTION ANALYZER

def analyze_question(question):

    q = question.lower().strip()

    # Superlative

    if re.search(
        r"\b("
        r"highest|lowest|most|least|largest|smallest|"
        r"maximum|minimum"
        r")\b",
        q
    ):

        return {
            "question_type": "superlative",
            "complexity": "high"
        }

    # Aggregation

    if re.search(
        r"\bhow many\b.*\bevents?\b"
        r".*\b(had|have|with|more than|less than|"
        r"greater than|under)\b"
        r"|"
        r"\bcount\b.*\bevents?\b"
        r"|"
        r"\bnumber of\b.*\bevents?\b",
        q
    ):

        return {
            "question_type": "aggregation",
            "complexity": "high"
        }

    # Temporal

    if re.search(
        r"\b("
        r"immediately before|"
        r"immediately after|"
        r"previous.*olympics|"
        r"next.*olympics|"
        r"preceding.*olympics|"
        r"following.*olympics|"
        r"prior to.*olympics|"
        r"after.*\d{4}|"
        r"before.*\d{4}"
        r")\b",
        q
    ):

        return {
            "question_type": "temporal",
            "complexity": "medium"
        }

    # Multi-hop

    if re.search(
        r"\bevent held at\b"
        r"|"
        r"\bheld at\b"
        r"|"
        r"\bon \d{1,2} [a-z]+ \d{4}\b",
        q
    ):

        return {
            "question_type": "multi_hop",
            "complexity": "high"
        }

    # Default

    return {
        "question_type": "lookup",
        "complexity": "low"
    }


# MENTAT ADAPTIVE ROUTER

def select_pipeline(analysis):

    qtype = analysis["question_type"]
    complexity = analysis["complexity"]

    # Simple factual questions

    if qtype == "lookup":

        return {
            "pipeline": "RAG",
            "reason": (
                "The question is a direct factual lookup "
                "and can initially be handled through "
                "hybrid retrieval."
            )
        }

    # Temporal questions

    if qtype == "temporal":

        return {
            "pipeline": "RAG",
            "reason": (
                "The question contains temporal constraints "
                "that can first be resolved through the "
                "existing RAG-v2 temporal retrieval."
            )
        }

    # Aggregation

    if qtype == "aggregation":

        return {
            "pipeline": "RAG",
            "reason": (
                "The question requires deterministic "
                "structured aggregation available in RAG-v2."
            )
        }

    # Superlative

    if qtype == "superlative":

        return {
            "pipeline": "RAG",
            "reason": (
                "The question requires deterministic "
                "comparison available in RAG-v2."
            )
        }

    # Multi-hop

    if qtype == "multi_hop":

        return {
            "pipeline": "GraphRAG",
            "reason": (
                "The question requires resolving relationships "
                "between an event, location, date, Olympic "
                "edition and source document."
            )
        }

    return {
        "pipeline": "RAG",
        "reason": "Default retrieval route."
    }


# PIPELINE EXECUTION

def execute_pipeline(
    pipeline,
    question
):

    start_time = time.perf_counter()

    # RAG

    if pipeline == "RAG":

        module = load_pipeline("RAG")

        result = module.rag_v2_route(
            question
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "pipeline": "RAG",
            "result": result,
            "latency": elapsed
        }

    # GRAPHRAG

    if pipeline == "GraphRAG":

        module = load_pipeline("GraphRAG")

        result = module.graph_rag(
            question,
            generate_answer=True
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "pipeline": "GraphRAG",
            "result": result,
            "latency": elapsed
        }

    # AGENTIC GRAPHRAG

    if pipeline == "Agentic GraphRAG":

        module = load_pipeline(
            "Agentic GraphRAG"
        )

        result = module.trace_run(
            question,
            max_steps=6
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "pipeline": "Agentic GraphRAG",
            "result": result,
            "latency": elapsed
        }

    raise ValueError(
        f"Unknown pipeline: {pipeline}"
    )

# EVIDENCE SUFFICIENCY

def is_sufficient(
    pipeline,
    result
):

    if not isinstance(
        result,
        dict
    ):
        return False

    # RAG

    if pipeline == "RAG":

        answer = result.get(
            "answer"
        )

        if not answer:
            return False

        answer_text = str(
            answer
        ).lower()

        failure_terms = [
            "no answer found",
            "insufficient evidence",
            "could not find",
            "not enough evidence",
            "unable to answer"
        ]

        return not any(
            term in answer_text
            for term in failure_terms
        )

    # GraphRAG

    if pipeline == "GraphRAG":

        return (
            result.get("status")
            == "success"
            and bool(result.get("answer"))
            and bool(result.get("event"))
            and bool(result.get("source_document"))
        )

    # Agentic GraphRAG

    if pipeline == "Agentic GraphRAG":

        return (
            result.get("finished", False)
            and bool(result.get("final_answer"))
        )

    return False

# ADAPTIVE EXECUTION

def mentat_execute(
    question
):

    analysis = analyze_question(
        question
    )

    first_route = select_pipeline(
        analysis
    )

    pipeline = first_route[
        "pipeline"
    ]

    execution_log = []

    # FIRST PIPELINE

    first_execution = execute_pipeline(
        pipeline,
        question
    )

    first_result = first_execution[
        "result"
    ]

    execution_log.append(
        {
            "pipeline": pipeline,
            "status": (
                "SUFFICIENT"
                if is_sufficient(
                    pipeline,
                    first_result
                )
                else "INSUFFICIENT"
            ),
            "latency": first_execution[
                "latency"
            ]
        }
    )

    # STOP IF EVIDENCE IS SUFFICIENT

    if is_sufficient(
        pipeline,
        first_result
    ):

        return {
            "analysis": analysis,
            "selected_pipeline": pipeline,
            "final_pipeline": pipeline,
            "result": first_result,
            "execution_log": execution_log,
            "escalated": False,
            "latency": first_execution[
                "latency"
            ]
        }

    # RAG → GRAPHRAG

    if pipeline == "RAG":

        graph_execution = execute_pipeline(
            "GraphRAG",
            question
        )

        graph_result = graph_execution[
            "result"
        ]

        execution_log.append(
            {
                "pipeline": "GraphRAG",
                "status": (
                    "SUFFICIENT"
                    if is_sufficient(
                        "GraphRAG",
                        graph_result
                    )
                    else "INSUFFICIENT"
                ),
                "latency": graph_execution[
                    "latency"
                ]
            }
        )

        if is_sufficient(
            "GraphRAG",
            graph_result
        ):

            return {
                "analysis": analysis,
                "selected_pipeline": pipeline,
                "final_pipeline": "GraphRAG",
                "result": graph_result,
                "execution_log": execution_log,
                "escalated": True,
                "latency": (
                    first_execution["latency"]
                    +
                    graph_execution["latency"]
                )
            }

        pipeline = "GraphRAG"

    # GRAPHRAG → AGENTIC GRAPHRAG

    agent_execution = execute_pipeline(
        "Agentic GraphRAG",
        question
    )

    agent_result = agent_execution[
        "result"
    ]

    execution_log.append(
        {
            "pipeline": "Agentic GraphRAG",
            "status": (
                "SUFFICIENT"
                if is_sufficient(
                    "Agentic GraphRAG",
                    agent_result
                )
                else "INSUFFICIENT"
            ),
            "latency": agent_execution[
                "latency"
            ]
        }
    )

    return {
        "analysis": analysis,
        "selected_pipeline": (
            first_route["pipeline"]
        ),
        "final_pipeline": "Agentic GraphRAG",
        "result": agent_result,
        "execution_log": execution_log,
        "escalated": True,
        "latency": (
            first_execution["latency"]
            +
            agent_execution["latency"]
        )
    }


ARCHITECTURE_URL = os.getenv(
    "ARCHITECTURE_URL",
    "https://YOUR-ARCHITECTURE-PAGE"
)

COMPARISON_URL = os.getenv(
    "COMPARISON_URL",
    "https://YOUR-COMPARISON-PAGE"
)


# DASH APP

app = Dash(
    __name__,
    title="MENTAT | Adaptive Agentic GraphRAG",
    suppress_callback_exceptions=True
)

server = app.server


# COLORS

COLORS = {
    "background": "#F5F7FB",
    "surface": "#FFFFFF",
    "primary": "#3157D5",
    "primary_dark": "#203A9A",
    "secondary": "#6C4CE8",
    "success": "#16A34A",
    "warning": "#F59E0B",
    "danger": "#DC2626",
    "text": "#172033",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "rag": "#2563EB",
    "graph": "#059669",
    "agent": "#7C3AED",
}


# REUSABLE STYLES

CARD_STYLE = {
    "backgroundColor": COLORS["surface"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "16px",
    "padding": "22px",
    "boxShadow": "0 4px 16px rgba(15, 23, 42, 0.05)",
}


PIPELINE_CARD_STYLE = {
    **CARD_STYLE,
    "minHeight": "150px",
}


# HEADER

def create_header():

    return html.Div(
        [

            html.Div(
                [

                    html.Div(
                        "M",
                        style={
                            "width": "52px",
                            "height": "52px",
                            "borderRadius": "14px",
                            "background": (
                                "linear-gradient("
                                "135deg, #3157D5, #7C4DFF)"
                            ),
                            "color": "white",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "28px",
                            "fontWeight": "800",
                        },
                    ),

                    html.Div(
                        [

                            html.H1(
                                "MENTAT",
                                style={
                                    "margin": "0",
                                    "fontSize": "30px",
                                    "fontWeight": "800",
                                    "color": COLORS["text"],
                                    "letterSpacing": "-0.5px",
                                },
                            ),

                            html.Div(
                                "Multi-agent Evidence & Neural Thinking for Analysis & Truth",
                                style={
                                    "fontSize": "13px",
                                    "color": COLORS["muted"],
                                    "marginTop": "2px",
                                },
                            ),

                        ],
                    ),

                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "14px",
                },
            ),

            html.Div(
                [

                    html.Div(
                        [
                            html.Span(
                                "●",
                                style={
                                    "color": COLORS["success"],
                                    "fontSize": "15px",
                                },
                            ),

                            html.Span(
                                " SYSTEM ONLINE",
                                style={
                                    "fontSize": "12px",
                                    "fontWeight": "700",
                                    "color": COLORS["success"],
                                },
                            ),
                        ],
                        style={
                            "backgroundColor": "#ECFDF3",
                            "border": "1px solid #BBF7D0",
                            "padding": "9px 14px",
                            "borderRadius": "999px",
                        },
                    ),

                ],
            ),

        ],
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "marginBottom": "28px",
        },
    )


# INTRODUCTION

def create_intro():

    return html.Div(
        [

            html.H2(
                "Adaptive Intelligence for Evidence-Grounded Answers",
                style={
                    "fontSize": "28px",
                    "fontWeight": "750",
                    "color": COLORS["text"],
                    "marginBottom": "10px",
                },
            ),

            html.P(
                (
                    "MENTAT intelligently routes questions across "
                    "RAG-v2, GraphRAG, and Agentic GraphRAG based "
                    "on query complexity and evidence requirements. "
                    "It escalates reasoning only when the available "
                    "evidence is insufficient."
                ),
                style={
                    "fontSize": "15px",
                    "lineHeight": "1.7",
                    "color": COLORS["muted"],
                    "maxWidth": "950px",
                    "marginBottom": "0",
                },
            ),

        ],
        style={
            "marginBottom": "25px",
        },
    )


# ============================================================
# PIPELINE CARD
# ============================================================

def pipeline_card(
    title,
    subtitle,
    description,
    color,
    icon,
):

    return html.Div(
        [

            html.Div(
                [

                    html.Div(
                        icon,
                        style={
                            "width": "42px",
                            "height": "42px",
                            "borderRadius": "12px",
                            "backgroundColor": f"{color}15",
                            "color": color,
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "20px",
                            "fontWeight": "700",
                        },
                    ),

                    html.Div(
                        [

                            html.Div(
                                title,
                                style={
                                    "fontSize": "18px",
                                    "fontWeight": "750",
                                    "color": COLORS["text"],
                                },
                            ),

                            html.Div(
                                subtitle,
                                style={
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                    "color": color,
                                    "marginTop": "2px",
                                },
                            ),

                        ],
                    ),

                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "12px",
                    "marginBottom": "15px",
                },
            ),

            html.P(
                description,
                style={
                    "fontSize": "13px",
                    "lineHeight": "1.6",
                    "color": COLORS["muted"],
                    "margin": "0",
                },
            ),

        ],
        style=PIPELINE_CARD_STYLE,
    )


# PIPELINE SECTION

def create_pipeline_section():

    return html.Div(
        [

            pipeline_card(
                "RAG-v2",
                "HYBRID RETRIEVAL",
                (
                    "Semantic vector search, BM25 retrieval, "
                    "Reciprocal Rank Fusion, cross-encoder "
                    "reranking, and context expansion."
                ),
                COLORS["rag"],
                "R",
            ),

            pipeline_card(
                "GraphRAG",
                "STRUCTURED REASONING",
                (
                    "TigerGraph-based event relationships "
                    "combined with source-document grounding "
                    "for structured reasoning."
                ),
                COLORS["graph"],
                "G",
            ),

            pipeline_card(
                "Agentic GraphRAG",
                "ADAPTIVE INVESTIGATION",
                (
                    "Groq-powered planning, specialized tools, "
                    "multi-hop investigation, evidence evaluation, "
                    "and adaptive re-planning."
                ),
                COLORS["agent"],
                "A",
            ),

        ],
        style={
            "display": "grid",
            "gridTemplateColumns": (
                "repeat(3, minmax(0, 1fr))"
            ),
            "gap": "18px",
            "marginBottom": "25px",
        },
    )


# NAVIGATION CARDS

def navigation_section():

    return html.Div(
        [

            html.A(
                [

                    html.Div(
                        "⌬",
                        style={
                            "fontSize": "30px",
                            "color": COLORS["primary"],
                        },
                    ),

                    html.Div(
                        [

                            html.Div(
                                "View Architecture",
                                style={
                                    "fontSize": "16px",
                                    "fontWeight": "750",
                                    "color": COLORS["text"],
                                },
                            ),

                            html.Div(
                                "Explore the complete MENTAT architecture.",
                                style={
                                    "fontSize": "12px",
                                    "color": COLORS["muted"],
                                    "marginTop": "4px",
                                },
                            ),

                        ],
                    ),

                    html.Div(
                        "→",
                        style={
                            "marginLeft": "auto",
                            "fontSize": "22px",
                            "color": COLORS["primary"],
                        },
                    ),

                ],
                href=ARCHITECTURE_URL,
                target="_blank",
                style={
                    **CARD_STYLE,
                    "textDecoration": "none",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "14px",
                    "cursor": "pointer",
                },
            ),

            html.A(
                [

                    html.Div(
                        "▥",
                        style={
                            "fontSize": "30px",
                            "color": COLORS["secondary"],
                        },
                    ),

                    html.Div(
                        [

                            html.Div(
                                "Compare Pipelines",
                                style={
                                    "fontSize": "16px",
                                    "fontWeight": "750",
                                    "color": COLORS["text"],
                                },
                            ),

                            html.Div(
                                "Compare RAG, GraphRAG and Agentic GraphRAG.",
                                style={
                                    "fontSize": "12px",
                                    "color": COLORS["muted"],
                                    "marginTop": "4px",
                                },
                            ),

                        ],
                    ),

                    html.Div(
                        "→",
                        style={
                            "marginLeft": "auto",
                            "fontSize": "22px",
                            "color": COLORS["secondary"],
                        },
                    ),

                ],
                href=COMPARISON_URL,
                target="_blank",
                style={
                    **CARD_STYLE,
                    "textDecoration": "none",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "14px",
                    "cursor": "pointer",
                },
            ),

        ],
        style={
            "display": "grid",
            "gridTemplateColumns": (
                "repeat(2, minmax(0, 1fr))"
            ),
            "gap": "18px",
            "marginBottom": "25px",
        },
    )


# QUERY SECTION

def create_query_section():

    return html.Div(
        [

            html.Div(
                [

                    html.Div(
                        "Ask MENTAT",
                        style={
                            "fontSize": "21px",
                            "fontWeight": "750",
                            "color": COLORS["text"],
                        },
                    ),

                    html.Div(
                        (
                            "Enter a question and let MENTAT "
                            "decide the required reasoning level."
                        ),
                        style={
                            "fontSize": "13px",
                            "color": COLORS["muted"],
                            "marginTop": "4px",
                        },
                    ),

                ],
                style={
                    "marginBottom": "15px",
                },
            ),

            html.Div(
                [

                    dcc.Input(
                        id="question-input",
                        type="text",
                        placeholder=(
                            "Ask an Olympic knowledge question..."
                        ),
                        debounce=True,
                        style={
                            "flex": "1",
                            "height": "52px",
                            "border": (
                                f"1px solid {COLORS['border']}"
                            ),
                            "borderRadius": "12px",
                            "padding": "0 17px",
                            "fontSize": "14px",
                            "outline": "none",
                            "color": COLORS["text"],
                            "backgroundColor": "#FFFFFF",
                        },
                    ),

                    html.Button(
                        "Investigate",
                        id="investigate-button",
                        n_clicks=0,
                        style={
                            "height": "52px",
                            "padding": "0 28px",
                            "border": "none",
                            "borderRadius": "12px",
                            "background": (
                                "linear-gradient("
                                "135deg, #3157D5, #6C4CE8)"
                            ),
                            "color": "white",
                            "fontSize": "14px",
                            "fontWeight": "700",
                            "cursor": "pointer",
                            "boxShadow": (
                                "0 5px 14px "
                                "rgba(49, 87, 213, 0.25)"
                            ),
                        },
                    ),

                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                },
            ),

            html.Div(
                [

                    html.Button(
                        "Who hosted the 2012 Summer Olympics?",
                        id="example-1",
                        n_clicks=0,
                        className="example-button",
                    ),

                    html.Button(
                        (
                            "How many biathlon events at the "
                            "2018 Winter Olympics had more than "
                            "73 competitors?"
                        ),
                        id="example-2",
                        n_clicks=0,
                        className="example-button",
                    ),

                    html.Button(
                        (
                            "Which athletics event at the 2008 "
                            "Olympics had the most competitors?"
                        ),
                        id="example-3",
                        n_clicks=0,
                        className="example-button",
                    ),

                    html.Button(
                        (
                            "Who won the gold medal in the event "
                            "held at Olympic Weightlifting Gymnasium "
                            "on 20 September 1988?"
                        ),
                        id="example-4",
                        n_clicks=0,
                        className="example-button",
                    ),

                ],
                style={
                    "display": "flex",
                    "gap": "8px",
                    "flexWrap": "wrap",
                    "marginTop": "14px",
                },
            ),

        ],
        style={
            **CARD_STYLE,
            "marginBottom": "25px",
        },
    )


# RESULT PLACEHOLDERS

def create_results_section():

    return html.Div(
        [

            # LEFT COLUMN

            html.Div(
                [

                    html.Div(
                        [

                            html.Div(
                                "◉",
                                style={
                                    "fontSize": "23px",
                                    "color": COLORS["primary"],
                                },
                            ),

                            html.Div(
                                "MENTAT Decision",
                                style={
                                    "fontSize": "18px",
                                    "fontWeight": "750",
                                    "color": COLORS["text"],
                                },
                            ),

                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "10px",
                            "marginBottom": "18px",
                        },
                    ),

                    html.Div(
                        id="decision-content",
                        children=html.Div(
                            "Submit a question to see the selected reasoning strategy.",
                            style={
                                "color": COLORS["muted"],
                                "fontSize": "13px",
                                "lineHeight": "1.6",
                            },
                        ),
                    ),

                    html.Hr(
                        style={
                            "border": "none",
                            "borderTop": (
                                f"1px solid {COLORS['border']}"
                            ),
                            "margin": "20px 0",
                        }
                    ),

                    html.Div(
                        "Query Statistics",
                        style={
                            "fontSize": "16px",
                            "fontWeight": "750",
                            "color": COLORS["text"],
                            "marginBottom": "12px",
                        },
                    ),

                    html.Div(
                        id="statistics-content",
                        children=html.Div(
                            "Waiting for query...",
                            style={
                                "color": COLORS["muted"],
                                "fontSize": "13px",
                            },
                        ),
                    ),

                ],
                style={
                    **CARD_STYLE,
                },
            ),

            # CENTER COLUMN

            html.Div(
                [

                    html.Div(
                        [

                            html.Div(
                                "◎",
                                style={
                                    "fontSize": "24px",
                                    "color": COLORS["secondary"],
                                },
                            ),

                            html.Div(
                                "Reasoning Trace",
                                style={
                                    "fontSize": "18px",
                                    "fontWeight": "750",
                                    "color": COLORS["text"],
                                },
                            ),

                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "10px",
                            "marginBottom": "18px",
                        },
                    ),

                    html.Div(
                        id="trace-content",
                        children=html.Div(
                            "MENTAT investigation steps will appear here.",
                            style={
                                "color": COLORS["muted"],
                                "fontSize": "13px",
                            },
                        ),
                    ),

                ],
                style={
                    **CARD_STYLE,
                },
            ),

            # RIGHT COLUMN

            html.Div(
                [

                    html.Div(
                        [

                            html.Div(
                                "▣",
                                style={
                                    "fontSize": "24px",
                                    "color": COLORS["graph"],
                                },
                            ),

                            html.Div(
                                "Evidence & Answer",
                                style={
                                    "fontSize": "18px",
                                    "fontWeight": "750",
                                    "color": COLORS["text"],
                                },
                            ),

                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "10px",
                            "marginBottom": "18px",
                        },
                    ),

                    html.Div(
                        id="evidence-content",
                        children=html.Div(
                            "Evidence will appear after investigation.",
                            style={
                                "color": COLORS["muted"],
                                "fontSize": "13px",
                            },
                        ),
                    ),

                    html.Div(
                        id="answer-content",
                        style={
                            "marginTop": "18px",
                        },
                    ),

                ],
                style={
                    **CARD_STYLE,
                },
            ),

        ],
        style={
            "display": "grid",
            "gridTemplateColumns": (
                "1fr 1.25fr 1.25fr"
            ),
            "gap": "18px",
        },
    )


# MAIN LAYOUT

app.layout = html.Div(
    [

        dcc.Store(
            id="query-result-store"
        ),

        html.Div(
            [

                create_header(),

                create_intro(),

                create_pipeline_section(),

                navigation_section(),

                create_query_section(),

                create_results_section(),

            ],
            style={
                "maxWidth": "1500px",
                "margin": "0 auto",
                "padding": "30px 35px 50px 35px",
            },
        ),

    ],
    style={
        "minHeight": "100vh",
        "backgroundColor": COLORS["background"],
        "fontFamily": (
            "Inter, Segoe UI, Arial, sans-serif"
        ),
    },
)


# EXAMPLE QUESTION CALLBACKS

@app.callback(
    Output("question-input", "value"),

    Input("example-1", "n_clicks"),
    Input("example-2", "n_clicks"),
    Input("example-3", "n_clicks"),
    Input("example-4", "n_clicks"),

    prevent_initial_call=True,
)
def load_example(
    n1,
    n2,
    n3,
    n4,
):

    button_id = ctx.triggered_id

    if not button_id:
        return no_update

    examples = {
        "example-1":
            "Who hosted the 2012 Summer Olympics?",

        "example-2":
            (
                "How many biathlon events at the "
                "2018 Winter Olympics had more than "
                "73 competitors?"
            ),

        "example-3":
            (
                "Which athletics event at the 2008 "
                "Olympics had the most competitors?"
            ),

        "example-4":
            (
                "Who won the gold medal in the event "
                "held at Olympic Weightlifting Gymnasium "
                "on 20 September 1988?"
            ),
    }

    return examples.get(
        button_id,
        no_update
    )

# REAL MENTAT INVESTIGATION CALLBACK

@app.callback(
    Output(
        "decision-content",
        "children"
    ),

    Output(
        "trace-content",
        "children"
    ),

    Output(
        "evidence-content",
        "children"
    ),

    Output(
        "answer-content",
        "children"
    ),

    Output(
        "statistics-content",
        "children"
    ),

    Input(
        "investigate-button",
        "n_clicks"
    ),

    State(
        "question-input",
        "value"
    ),

    prevent_initial_call=True,
)
def investigate(
    n_clicks,
    question
):

    if not question or not question.strip():

        warning = html.Div(
            "Please enter a question first.",
            style={
                "color": COLORS["warning"],
                "fontWeight": "600",
            },
        )

        return (
            warning,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    try:

        # RUN MENTAT

        response = mentat_execute(
            question.strip()
        )

        analysis = response[
            "analysis"
        ]

        result = response[
            "result"
        ]

        final_pipeline = response[
            "final_pipeline"
        ]

        execution_log = response[
            "execution_log"
        ]

        # DECISION PANEL

        decision = html.Div(
            [

                html.Div(
                    [
                        html.Span(
                            "QUESTION TYPE",
                            style={
                                "fontSize": "10px",
                                "fontWeight": "800",
                                "color": COLORS["muted"],
                            },
                        ),

                        html.Div(
                            analysis[
                                "question_type"
                            ].replace(
                                "_",
                                " "
                            ).upper(),
                            style={
                                "fontSize": "17px",
                                "fontWeight": "800",
                                "color": COLORS["primary"],
                                "marginTop": "4px",
                            },
                        ),
                    ],
                    style={
                        "marginBottom": "16px"
                    },
                ),

                html.Div(
                    [
                        html.Span(
                            "COMPLEXITY",
                            style={
                                "fontSize": "10px",
                                "fontWeight": "800",
                                "color": COLORS["muted"],
                            },
                        ),

                        html.Div(
                            analysis[
                                "complexity"
                            ].upper(),
                            style={
                                "fontSize": "14px",
                                "fontWeight": "700",
                                "color": COLORS["text"],
                                "marginTop": "4px",
                            },
                        ),
                    ],
                    style={
                        "marginBottom": "16px"
                    },
                ),

                html.Div(
                    [
                        html.Span(
                            "FINAL PIPELINE",
                            style={
                                "fontSize": "10px",
                                "fontWeight": "800",
                                "color": COLORS["muted"],
                            },
                        ),

                        html.Div(
                            final_pipeline,
                            style={
                                "fontSize": "17px",
                                "fontWeight": "800",
                                "color": (
                                    COLORS["agent"]
                                    if final_pipeline
                                    == "Agentic GraphRAG"
                                    else (
                                        COLORS["graph"]
                                        if final_pipeline
                                        == "GraphRAG"
                                        else COLORS["rag"]
                                    )
                                ),
                                "marginTop": "4px",
                            },
                        ),
                    ],
                    style={
                        "marginBottom": "16px"
                    },
                ),

                html.Div(
                    (
                        "Reasoning escalated because "
                        "additional evidence was required."
                        if response["escalated"]
                        else
                        "Evidence was sufficient at the "
                        "initial reasoning level."
                    ),
                    style={
                        "fontSize": "12px",
                        "lineHeight": "1.6",
                        "color": COLORS["muted"],
                        "backgroundColor": "#F8FAFC",
                        "borderRadius": "10px",
                        "padding": "11px",
                    },
                ),

            ]
        )

        # REASONING TRACE

        trace_items = []

        for index, item in enumerate(
            execution_log,
            start=1
        ):

            status = item[
                "status"
            ]

            successful = (
                status == "SUFFICIENT"
            )

            trace_items.append(
                html.Div(
                    [

                        html.Div(
                            str(index),
                            style={
                                "width": "28px",
                                "height": "28px",
                                "borderRadius": "50%",
                                "backgroundColor": (
                                    "#DCFCE7"
                                    if successful
                                    else "#EEF2FF"
                                ),
                                "color": (
                                    COLORS["success"]
                                    if successful
                                    else COLORS["primary"]
                                ),
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "fontSize": "11px",
                                "fontWeight": "800",
                                "flexShrink": "0",
                            },
                        ),

                        html.Div(
                            [

                                html.Div(
                                    item[
                                        "pipeline"
                                    ],
                                    style={
                                        "fontSize": "13px",
                                        "fontWeight": "750",
                                        "color": COLORS["text"],
                                    },
                                ),

                                html.Div(
                                    status,
                                    style={
                                        "fontSize": "10px",
                                        "fontWeight": "700",
                                        "color": (
                                            COLORS["success"]
                                            if successful
                                            else COLORS["warning"]
                                        ),
                                        "marginTop": "3px",
                                    },
                                ),

                            ],
                        ),

                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "10px",
                        "padding": "10px 0",
                        "borderBottom": (
                            f"1px solid "
                            f"{COLORS['border']}"
                        ),
                    },
                )
            )

        trace = html.Div(
            trace_items
        )

        # EVIDENCE

        evidence_items = []

        if final_pipeline == "RAG":

            rag_result = result

            method = rag_result.get(
                "method",
                "hybrid_rag"
            )

            evidence_items.append(
                html.Div(
                    [
                        html.Div(
                            "Retrieval Method",
                            style={
                                "fontSize": "10px",
                                "fontWeight": "800",
                                "color": COLORS["muted"],
                            },
                        ),

                        html.Div(
                            method,
                            style={
                                "fontSize": "13px",
                                "fontWeight": "700",
                                "color": COLORS["text"],
                                "marginTop": "3px",
                            },
                        ),
                    ],
                    style={
                        "marginBottom": "14px"
                    },
                )
            )

            rag_data = rag_result.get(
                "result",
                {}
            )

            evidence_count = len(
                rag_data.get(
                    "expanded_context",
                    []
                )
            )

            evidence_items.append(
                html.Div(
                    f"{evidence_count} retrieved evidence units",
                    style={
                        "fontSize": "12px",
                        "color": COLORS["muted"],
                    },
                )
            )

        elif final_pipeline == "GraphRAG":

            event = result.get(
                "event"
            )

            source = result.get(
                "source_document"
            )

            graph = result.get(
                "graph",
                {}
            )

            if event:

                evidence_items.append(
                    html.Div(
                        [
                            html.Div(
                                "RESOLVED EVENT",
                                style={
                                    "fontSize": "10px",
                                    "fontWeight": "800",
                                    "color": COLORS["muted"],
                                },
                            ),

                            html.Div(
                                event.get(
                                    "event_name",
                                    "Unknown"
                                ),
                                style={
                                    "fontSize": "13px",
                                    "fontWeight": "700",
                                    "color": COLORS["text"],
                                    "marginTop": "4px",
                                },
                            ),
                        ],
                        style={
                            "marginBottom": "14px"
                        },
                    )
                )

            evidence_items.append(
                html.Div(
                    f"Graph relationships: "
                    f"{len(graph.get('part_of', [])) + len(graph.get('held_at', []))}",
                    style={
                        "fontSize": "12px",
                        "color": COLORS["muted"],
                        "marginBottom": "7px",
                    },
                )
            )

            if source:

                evidence_items.append(
                    html.Div(
                        f"Source: {source.get('title', 'Unknown')}",
                        style={
                            "fontSize": "12px",
                            "color": COLORS["muted"],
                            "lineHeight": "1.5",
                        },
                    )
                )

        elif final_pipeline == "Agentic GraphRAG":

            evidence_items.append(
                html.Div(
                    "Agentic investigation completed.",
                    style={
                        "fontSize": "12px",
                        "fontWeight": "700",
                        "color": COLORS["success"],
                        "marginBottom": "12px",
                    },
                )
            )

            tool_history = result.get(
                "tool_history",
                []
            )

            evidence_items.append(
                html.Div(
                    f"Tool operations: {len(tool_history)}",
                    style={
                        "fontSize": "12px",
                        "color": COLORS["muted"],
                        "marginBottom": "7px",
                    },
                )
            )

            evidence_items.append(
                html.Div(
                    f"Final stage: "
                    f"{result.get('final_stage', 'UNKNOWN')}",
                    style={
                        "fontSize": "12px",
                        "color": COLORS["muted"],
                    },
                )
            )

        evidence = html.Div(
            evidence_items
        )

    
        # FINAL ANSWER

        if final_pipeline == "Agentic GraphRAG":

            final_answer = result.get(
                "final_answer"
            )

        else:

            final_answer = result.get(
                "answer"
            )

        if final_answer:
            final_answer = str(final_answer).strip()
        else:
            final_answer = "No final answer was produced."

        if not final_answer:

            final_answer = (
                "No final answer was produced."
            )

        answer = html.Div(
            [

                html.Div(
                    "FINAL ANSWER",
                    style={
                        "fontSize": "10px",
                        "fontWeight": "800",
                        "color": COLORS["muted"],
                        "marginBottom": "7px",
                    },
                ),

                html.Div(
                    final_answer,
                    style={
                        "fontSize": "16px",
                        "fontWeight": "750",
                        "lineHeight": "1.6",
                        "color": COLORS["text"],
                        "backgroundColor": "#F8FAFC",
                        "border": (
                            f"1px solid "
                            f"{COLORS['border']}"
                        ),
                        "borderRadius": "12px",
                        "padding": "14px",
                    },
                ),

                html.Div(
                    "✓ Evidence sufficient",
                    style={
                        "display": "inline-block",
                        "marginTop": "10px",
                        "padding": "6px 10px",
                        "borderRadius": "999px",
                        "backgroundColor": "#ECFDF3",
                        "color": COLORS["success"],
                        "fontSize": "10px",
                        "fontWeight": "800",
                    },
                ),

            ]
        )

        # STATISTICS

        total_steps = len(
            execution_log
        )

        statistics = html.Div(
            [

                html.Div(
                    f"Pipeline: {final_pipeline}",
                    style={
                        "fontSize": "12px",
                        "marginBottom": "8px",
                    },
                ),

                html.Div(
                    f"Reasoning steps: {total_steps}",
                    style={
                        "fontSize": "12px",
                        "marginBottom": "8px",
                    },
                ),

                html.Div(
                    f"Latency: "
                    f"{response['latency']:.2f}s",
                    style={
                        "fontSize": "12px",
                        "marginBottom": "8px",
                    },
                ),

                html.Div(
                    (
                        "Escalated: YES"
                        if response["escalated"]
                        else "Escalated: NO"
                    ),
                    style={
                        "fontSize": "12px",
                        "fontWeight": "700",
                        "color": (
                            COLORS["warning"]
                            if response["escalated"]
                            else COLORS["success"]
                        ),
                    },
                ),

            ]
        )

        return (
            decision,
            trace,
            evidence,
            answer,
            statistics,
        )

    except Exception as exc:

        error = html.Div(
            [

                html.Div(
                    "Pipeline Execution Error",
                    style={
                        "fontWeight": "800",
                        "color": COLORS["danger"],
                        "marginBottom": "7px",
                    },
                ),

                html.Div(
                    str(exc),
                    style={
                        "fontSize": "12px",
                        "color": COLORS["muted"],
                        "lineHeight": "1.5",
                    },
                ),

            ]
        )

        return (
            error,
            no_update,
            no_update,
            no_update,
            no_update,
        )


# RUN

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050,
    )
