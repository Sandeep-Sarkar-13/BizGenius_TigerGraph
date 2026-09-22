import os, re, json, argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from rapidfuzz import fuzz
from groq import Groq
from pyTigerGraph import TigerGraphConnection

# ============================================================
# CONFIGURATION
# ============================================================
# This is the TigerGraph Cloud instance used by the existing
# Olympic_GraphRAG setup. Override TG_HOST in .env if needed.
TG_HOST = os.getenv(
    "TG_HOST",
    "https://tg-c65a7a6e-8a3b-4f80-9678-04eb00845868.tg-2635877100.i.tgcloud.io",
)
TG_GRAPH = os.getenv("TG_GRAPH", "Olympic_GraphRAG")
TG_SECRET = os.getenv("TG_SECRET", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
CORPUS_PATH = os.getenv("CORPUS_PATH", "./data/corpus.jsonl")
MAX_STEPS = int(os.getenv("TRACE_MAX_STEPS", "6"))

# ============================================================
# VALIDATION / CONNECTIONS
# ============================================================
def require_config():
    missing = []
    if not TG_SECRET: missing.append("TG_SECRET")
    if not GROQ_API_KEY: missing.append("GROQ_API_KEY")
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

require_config()

print("Connecting to TigerGraph Cloud...")
conn = TigerGraphConnection(host=TG_HOST, graphname=TG_GRAPH, gsqlSecret=TG_SECRET)
conn.getToken(TG_SECRET)
print(f"TigerGraph: connected | graph={TG_GRAPH}")

print(f"Initializing Groq | model={GROQ_MODEL}")
groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# CORPUS
# ============================================================
def load_corpus(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    required = ["doc_id", "title", "url", "wikidata_qid", "wikipedia_pageid", "approx_tokens", "text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing corpus columns: {missing}")
    df["doc_id"] = df["doc_id"].astype(str)
    return df

corpus_df = load_corpus(CORPUS_PATH)
corpus_lookup = {str(r["doc_id"]): r.to_dict() for _, r in corpus_df.iterrows()}
print(f"Corpus loaded: {len(corpus_df):,} documents")

# ============================================================
# EVENT INDEX
# ============================================================
def normalize_text(x: Any) -> str:
    x = str(x or "").replace("–", "-").replace("—", "-").replace("−", "-")
    x = re.sub(r"\s+", " ", x)
    return x.strip().lower()

def first_value(text: str, keys: List[str]) -> str:
    for key in keys:
        m = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text)
        if m:
            return m.group(1).strip()
    return ""

def parse_event_row(row: Dict[str, Any]) -> Dict[str, Any]:
    text = str(row.get("text", ""))
    title = str(row.get("title", ""))
    # The corpus titles are the most reliable event-document label.
    event_name = title
    sport = first_value(text, ["sport"])
    games = first_value(text, ["games", "olympics"])
    year = None
    season = ""
    m = re.search(r"\b(18|19|20)\d{2}\b", title)
    if m:
        year = int(m.group(0))
    if "summer olympics" in title.lower() or "summer olympics" in text[:5000].lower():
        season = "Summer"
    elif "winter olympics" in title.lower() or "winter olympics" in text[:5000].lower():
        season = "Winter"
    if not year:
        m = re.search(r"\b(18|19|20)\d{2}\b", text[:6000])
        if m: year = int(m.group(0))
    # Prefer an explicit sport in the corpus; otherwise infer from title.
    if not sport:
        common = ["athletics","swimming","cycling","weightlifting","rowing","canoeing","sailing","gymnastics","judo","boxing","fencing","shooting","archery","wrestling","tennis","football","basketball","hockey"]
        tl = title.lower()
        sport = next((s for s in common if s in tl), "")
    return {
        "event_id": f"event_{row['doc_id']}",
        "doc_id": str(row["doc_id"]),
        "event_name": event_name,
        "sport": sport,
        "games": games,
        "year": year,
        "season": season,
        "title": title,
    }

event_index_df = pd.DataFrame([parse_event_row(r) for r in corpus_lookup.values()])
print(f"Event index: {len(event_index_df):,} candidate documents")

# ============================================================
# TIGERGRAPH TOOLS
# ============================================================
def search_graph_event(event_id: str) -> Dict[str, Any]:
    try:
        return {
            "success": True,
            "tool": "search_graph_event",
            "event_id": event_id,
            "games": conn.getEdges("Event", event_id, "PART_OF"),
            "locations": conn.getEdges("Event", event_id, "HELD_AT"),
        }
    except Exception as e:
        return {"success": False, "tool": "search_graph_event", "event_id": event_id, "error": str(e)}

def search_graph_document(doc_id: str) -> Dict[str, Any]:
    try:
        return {
            "success": True,
            "tool": "search_graph_document",
            "doc_id": doc_id,
            "games": conn.getEdges("Document", doc_id, "DOCUMENT_MENTIONS_GAMES"),
            "events": conn.getEdges("Document", doc_id, "DOCUMENT_MENTIONS_EVENT"),
            "locations": conn.getEdges("Document", doc_id, "DOCUMENT_MENTIONS_LOCATION"),
        }
    except Exception as e:
        return {"success": False, "tool": "search_graph_document", "doc_id": doc_id, "error": str(e)}

def inspect_source_document(doc_id: str) -> Dict[str, Any]:
    row = corpus_lookup.get(str(doc_id))
    if row is None:
        return {"success": False, "tool": "inspect_source_document", "doc_id": str(doc_id), "error": "Document not found."}
    return {"success": True, "tool": "inspect_source_document", "doc_id": str(doc_id), **row}

# ============================================================
# EVENT DISCOVERY / INVESTIGATION
# ============================================================
def discover_event(question: str) -> Dict[str, Any]:
    q = normalize_text(question)
    candidates = event_index_df.copy()
    ym = re.search(r"\b(?:18|19|20)\d{2}\b", q)
    if ym:
        y = int(ym.group(0)); c = candidates[candidates["year"] == y]
        if len(c): candidates = c
    if "summer olympics" in q:
        c = candidates[candidates["season"].astype(str).str.lower() == "summer"]
        if len(c): candidates = c
    if "winter olympics" in q:
        c = candidates[candidates["season"].astype(str).str.lower() == "winter"]
        if len(c): candidates = c
    venue = None
    vm = re.search(r"held at\s+(.+?)(?:\s+on\s+|\s+between\s+|\s+during\s+|$)", question, re.I)
    if vm: venue = vm.group(1).strip()
    scored = []
    for _, row in candidates.iterrows():
        event_text = normalize_text(" ".join([str(row.event_name), str(row.title), str(row.sport)]))
        score = fuzz.token_set_ratio(q, event_text) * 0.01
        doc = corpus_lookup.get(str(row.doc_id))
        if venue and doc:
            score += fuzz.partial_ratio(normalize_text(venue), normalize_text(doc["text"][:10000])) * 0.50
        # date evidence
        nums = re.findall(r"\b\d{1,2}\b", q)
        if doc and nums:
            dt = str(doc["text"][:8000]).lower()
            score += min(sum(1 for n in nums if re.search(rf"\b{re.escape(n)}\b", dt)), 4) * 10
        scored.append({**row.to_dict(), "score": float(score)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"success": bool(scored), "tool": "discover_event", "query": question, "candidates": scored[:5]}

def investigate_event_candidates(question: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    vm = re.search(r"held at\s+(.+?)\s+on\s+", question, re.I)
    requested_venue = vm.group(1).strip() if vm else None
    dm = re.search(r"(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", question, re.I)
    requested_dates = dm.group(0) if dm else None
    out = []
    for c in candidates:
        source = inspect_source_document(str(c["doc_id"]))
        if not source.get("success"): continue
        text = str(source.get("text", "")); nt = normalize_text(text)
        venue_score = fuzz.partial_ratio(normalize_text(requested_venue), nt[:15000]) / 100 if requested_venue else 0
        date_score = 0
        if requested_dates:
            nums = re.findall(r"\d+", requested_dates)
            date_score = sum(1 for n in nums if n in text) / max(1, len(nums))
        score = venue_score*0.50 + date_score*0.40 + float(c.get("score",0))/100*0.10
        out.append({**c, "requested_venue": requested_venue, "requested_dates": requested_dates, "venue_score": round(venue_score,3), "date_score": round(date_score,3), "evidence_score": round(score,3), "source_text": text[:5000]})
    out.sort(key=lambda x:(x.get("evidence_score",0),x.get("date_score",0),x.get("venue_score",0)), reverse=True)
    return {"success": True, "tool":"investigate_event_candidates", "results":out, "best_candidate": out[0] if out else None}

# ============================================================
# SPECIALIZED DETERMINISTIC TOOLS
# ============================================================
def lookup_event_fact(question: str, event_id: str = "") -> Dict[str, Any]:
    if event_id:
        if not event_id.startswith("event_"):
            return {"success": False, "tool":"lookup_event_fact", "reason":"Invalid event_id."}
        doc_id = event_id[len("event_"):]
        src = inspect_source_document(doc_id)
        if not src.get("success"): return src
        matches = event_index_df[event_index_df.event_id.astype(str) == event_id]
        candidate = matches.iloc[0].to_dict() if len(matches) else None
        return {"success":True,"tool":"lookup_event_fact","event_id":event_id,"doc_id":doc_id,"source":src,"candidate":candidate}
    d = discover_event(question)
    if not d.get("candidates"): return {"success":False,"tool":"lookup_event_fact","reason":"No event candidate found."}
    inv = investigate_event_candidates(question,d["candidates"])
    best = inv.get("best_candidate")
    if not best: return {"success":False,"tool":"lookup_event_fact","reason":"No source candidate found."}
    src = inspect_source_document(best["doc_id"])
    return {"success":True,"tool":"lookup_event_fact","event_id":best["event_id"],"doc_id":best["doc_id"],"source":src,"candidate":best}

def resolve_temporal_edition(question: str) -> Dict[str, Any]:
    q = question.lower(); ym = re.findall(r"\b(?:18|19|20)\d{2}\b", question)
    if not ym: return {"success":False,"tool":"resolve_temporal_edition","reason":"No target year detected."}
    target = int(ym[-1]); season = "Winter" if "winter" in q else ("Summer" if "summer" in q else None)
    games = event_index_df[["year","season","games"]].drop_duplicates()
    games = games[games.year.notna()]
    if season: games = games[games.season.astype(str).str.lower() == season.lower()]
    games = games[games.year.astype(int) < target]
    if games.empty: return {"success":False,"tool":"resolve_temporal_edition","reason":"No earlier Olympic edition found."}
    r = games.sort_values("year", ascending=False).iloc[0]
    return {"success":True,"tool":"resolve_temporal_edition","target_year":target,"resolved_year":int(r.year),"season":r.season,"games":r.games}

def resolve_temporal_event(question: str) -> Dict[str, Any]:
    ed = resolve_temporal_edition(question)
    if not ed.get("success"): return {"success":False,"tool":"resolve_temporal_event","reason":ed.get("reason")}
    c = event_index_df[event_index_df.year.astype(str) == str(ed["resolved_year"])].copy()
    if ed.get("season"): c = c[c.season.astype(str).str.lower() == str(ed["season"]).lower()]
    q = question.lower(); score_rows=[]
    sports=["athletics","swimming","cycling","weightlifting","rowing","canoeing","sailing","gymnastics","judo","boxing","fencing","shooting","archery","wrestling"]
    sport=next((s for s in sports if s in q),None)
    patterns=["20 kilometres walk","20 kilometre walk","20 km walk","50 kilometres walk","50 kilometre walk","100 metres","200 metres","400 metres","800 metres","1500 metres","5000 metres","10000 metres"]
    pattern=next((p for p in patterns if p in q),None)
    for _,r in c.iterrows():
        en=str(r.event_name).lower(); s=0
        if pattern and pattern in en: s+=100
        if "walk" in q and "walk" in en: s+=30
        if "men's" in q and "men's" in en: s+=20
        if "women's" in q and "women's" in en: s+=20
        if sport and sport in str(r.sport).lower(): s+=20
        score_rows.append({**r.to_dict(),"score":s})
    score_rows.sort(key=lambda x:x["score"],reverse=True)
    if not score_rows: return {"success":False,"tool":"resolve_temporal_event","reason":"No event candidates found."}
    b=score_rows[0]
    return {"success":True,"tool":"resolve_temporal_event","resolved_year":ed["resolved_year"],"season":ed["season"],"event_id":b["event_id"],"doc_id":b["doc_id"],"event_name":b["event_name"],"candidate":b}

def extract_competitors(text: str) -> Optional[int]:
    pats=[r"number of competitors\s*[:=]\s*(\d+)",r"competitors\s*[:=]\s*(\d+)",r"entries\s*[:=]\s*(\d+)"]
    for p in pats:
        m=re.search(p,text,re.I)
        if m:return int(m.group(1))
    return None

def aggregate_events(question: str) -> Dict[str, Any]:
    q=question.lower(); c=event_index_df.copy()
    ym=re.search(r"\b(?:18|19|20)\d{2}\b",q)
    if ym: c=c[c.year==int(ym.group(0))]
    if "summer olympics" in q:c=c[c.season.astype(str).str.lower()=="summer"]
    if "winter olympics" in q:c=c[c.season.astype(str).str.lower()=="winter"]
    for sport in sorted([str(x) for x in event_index_df.sport.dropna().unique()],key=len,reverse=True):
        if sport.lower() in q:c=c[c.sport.astype(str).str.lower()==sport.lower()];break
    threshold=None
    m=re.search(r"(?:more than|greater than|over)\s+(\d+)",q)
    if m: op,threshold="gt",int(m.group(1))
    else:
        m=re.search(r"(?:less than|under|fewer than)\s+(\d+)",q)
        if m: op,threshold="lt",int(m.group(1))
    vals=[]
    for _,r in c.iterrows():
        src=corpus_lookup.get(str(r.doc_id)); n=extract_competitors(str(src.get("text",""))) if src else None
        if n is not None: vals.append((r.to_dict(),n))
    if threshold is not None:
        vals=[x for x in vals if (x[1]>threshold if op=="gt" else x[1]<threshold)]
    return {"success":True,"tool":"aggregate_events","answer":len(vals),"candidate_count":len(vals),"reason":"Deterministic corpus aggregation completed."}

def find_event_extreme(question: str) -> Dict[str, Any]:
    q=question.lower(); c=event_index_df.copy(); ym=re.search(r"\b(?:18|19|20)\d{2}\b",q)
    if ym:c=c[c.year==int(ym.group(0))]
    if "summer olympics" in q:c=c[c.season.astype(str).str.lower()=="summer"]
    if "winter olympics" in q:c=c[c.season.astype(str).str.lower()=="winter"]
    for sport in sorted([str(x) for x in event_index_df.sport.dropna().unique()],key=len,reverse=True):
        if sport.lower() in q:c=c[c.sport.astype(str).str.lower()==sport.lower()];break
    vals=[]
    for _,r in c.iterrows():
        src=corpus_lookup.get(str(r.doc_id)); n=extract_competitors(str(src.get("text",""))) if src else None
        if n is not None: vals.append({**r.to_dict(),"competitors":n})
    if not vals:return {"success":False,"tool":"find_event_extreme","reason":"No qualifying events found."}
    best=max(vals,key=lambda x:x["competitors"]) if "lowest" not in q and "fewest" not in q and "minimum" not in q else min(vals,key=lambda x:x["competitors"])
    return {"success":True,"tool":"find_event_extreme","answer":best["title"],"event":best,"extreme_value":best["competitors"]}

# ============================================================
# TRACE STATE
# ============================================================
@dataclass
class TraceState:
    question: str
    question_type: str = "lookup"
    complexity: str = "low"
    initial_route: str = "RAG"
    step: int = 0
    evidence: List[Dict[str,Any]] = field(default_factory=list)
    graph_evidence: List[Dict[str,Any]] = field(default_factory=list)
    tool_history: List[Dict[str,Any]] = field(default_factory=list)
    answer_candidate: Optional[str] = None
    finished: bool = False
    current_stage: str = "INITIAL"
    stop_reason: Optional[str] = None

def analyze_question(question: str) -> Dict[str,str]:
    q=question.lower().strip()
    if re.search(r"\b(highest|lowest|most|least|largest|smallest|maximum|minimum)\b",q): return {"question_type":"superlative","complexity":"high"}
    if re.search(r"\bhow many\b.*\bevents?\b.*\b(had|have|with|more than|less than|greater than|under)\b|\bcount\b.*\bevents?\b|\bnumber of\b.*\bevents?\b",q): return {"question_type":"aggregation","complexity":"high"}
    if re.search(r"\b(immediately before|immediately after|previous.*olympics|next.*olympics|preceding.*olympics|following.*olympics|prior to.*olympics|after.*\d{4}|before.*\d{4})\b",q): return {"question_type":"temporal","complexity":"medium"}
    if re.search(r"\bevent held at\b|\bheld at\b|\bon \d{1,2} [a-z]+ \d{4}\b",q): return {"question_type":"multi_hop","complexity":"high"}
    return {"question_type":"lookup","complexity":"low"}

def create_state(question:str)->TraceState:
    a=analyze_question(question); route="RAG" if a["question_type"]=="lookup" else "GraphRAG"
    return TraceState(question=question,question_type=a["question_type"],complexity=a["complexity"],initial_route=route)

def add_evidence(state,action,result):
    success=bool(result.get("success")) if isinstance(result,dict) else False
    state.tool_history.append({"action":action,"target":result.get("event_id") or result.get("doc_id") or "","success":success})
    state.evidence.append({"type":action,"tool":action,"status":"SUCCESS" if success else "FAILED","result":result,"event_id":result.get("event_id") if isinstance(result,dict) else None,"doc_id":result.get("doc_id") if isinstance(result,dict) else None})
    if action.startswith("search_graph") and success: state.graph_evidence.append(result)

def has_tool(state,name):
    return any(x.get("tool")==name and x.get("status")=="SUCCESS" for x in state.evidence)

def source_available(state):
    for x in state.evidence:
        r=x.get("result",{})
        if isinstance(r,dict):
            s=r.get("source")
            if isinstance(s,dict) and s.get("text"): return True
            if x.get("tool")=="inspect_source_document" and r.get("text"): return True
    return False

def selected_event(state):
    for x in reversed(state.evidence):
        r=x.get("result",{})
        if x.get("tool")=="investigate_event_candidates" and r.get("best_candidate"):
            return r["best_candidate"].get("event_id")
        if r.get("event_id","").startswith("event_"): return r["event_id"]
    return ""

def evaluate(state):
    q=state.question_type
    if q=="lookup":
        return {"status":"SUFFICIENT","next_action":"STOP"} if has_tool(state,"lookup_event_fact") and source_available(state) else {"status":"INSUFFICIENT","next_action":"lookup_event_fact"}
    if q=="temporal":
        if has_tool(state,"lookup_event_fact") and source_available(state): return {"status":"SUFFICIENT","next_action":"STOP"}
        if not has_tool(state,"resolve_temporal_edition"): return {"status":"INSUFFICIENT","next_action":"resolve_temporal_edition"}
        if not has_tool(state,"resolve_temporal_event"): return {"status":"INSUFFICIENT","next_action":"resolve_temporal_event"}
        return {"status":"INSUFFICIENT","next_action":"lookup_event_fact"}
    if q=="multi_hop":
        if has_tool(state,"search_graph_event") and source_available(state): return {"status":"SUFFICIENT","next_action":"STOP"}
        if has_tool(state,"lookup_event_fact") and source_available(state): return {"status":"SUFFICIENT","next_action":"STOP"}
        if has_tool(state,"search_graph_event"): return {"status":"ESCALATE_TO_SOURCE","next_action":"lookup_event_fact"}
        if has_tool(state,"investigate_event_candidates"): return {"status":"ESCALATE_TO_GRAPHRAG","next_action":"search_graph_event"}
        if has_tool(state,"discover_event"): return {"status":"INSUFFICIENT","next_action":"investigate_event_candidates"}
        return {"status":"INSUFFICIENT","next_action":"discover_event"}
    if q=="aggregation": return {"status":"SUFFICIENT","next_action":"STOP"} if has_tool(state,"aggregate_events") else {"status":"INSUFFICIENT","next_action":"aggregate_events"}
    if q=="superlative": return {"status":"SUFFICIENT","next_action":"STOP"} if has_tool(state,"find_event_extreme") else {"status":"INSUFFICIENT","next_action":"find_event_extreme"}
    return {"status":"INSUFFICIENT","next_action":"discover_event"}

# ============================================================
# GROQ PLANNER
# ============================================================
TOOLS = {
    "lookup_event_fact":"Retrieve source evidence for a specific event.",
    "resolve_temporal_edition":"Resolve the relevant Olympic edition.",
    "resolve_temporal_event":"Resolve the event inside the Olympic edition.",
    "aggregate_events":"Count qualifying corpus events deterministically.",
    "find_event_extreme":"Find the maximum/minimum event deterministically.",
    "discover_event":"Discover candidate events from corpus evidence.",
    "investigate_event_candidates":"Inspect candidate documents using venue/date evidence.",
    "search_graph_event":"Verify Event -> Olympic_Games and Event -> Location relationships in TigerGraph.",
    "search_graph_document":"Verify Document relationships in TigerGraph.",
    "inspect_source_document":"Retrieve the source document text.",
}

def groq_plan(state: TraceState) -> Dict[str,str]:
    ev=evaluate(state)
    if ev["status"] in {"INSUFFICIENT","ESCALATE_TO_GRAPHRAG","ESCALATE_TO_SOURCE"}:
        # The evaluator is authoritative for the evidence gap.
        # Groq is still used for the agentic decision when there is a choice.
        if state.question_type in {"multi_hop","lookup"} and ev["next_action"] in {"discover_event","investigate_event_candidates","search_graph_event","lookup_event_fact"}:
            mandatory=ev["next_action"]
        elif state.question_type in {"temporal"}:
            mandatory=ev["next_action"]
        else:
            mandatory=ev["next_action"]
    else:
        return {"action":"STOP","target":"","reason":"Evidence is sufficient."}
    prompt=f"""You are the TRACE investigation planner. Choose ONE next action only. Do not answer the user.\nQuestion: {state.question}\nType: {state.question_type}\nComplexity: {state.complexity}\nCurrent evidence: {json.dumps(state.evidence[-5:],default=str)[:7000]}\nGraph evidence: {json.dumps(state.graph_evidence[-5:],default=str)[:4000]}\nEvaluator-required action: {mandatory}\nTools: {json.dumps(TOOLS)}\nRules: do not repeat an action without new evidence; use graph verification for relational multi-hop questions; use source text for factual answers; return JSON only.\n{{\"action\":\"...\",\"target\":\"\",\"reason\":\"...\"}}"""
    try:
        r=groq_client.chat.completions.create(model=GROQ_MODEL,messages=[{"role":"system","content":"Return valid JSON only."},{"role":"user","content":prompt}],temperature=0,max_tokens=300)
        txt=r.choices[0].message.content.strip(); m=re.search(r"\{.*\}",txt,re.S); data=json.loads(m.group(0)) if m else {}
        action=data.get("action",mandatory); target=str(data.get("target","") or "")
        if action not in set(TOOLS)|{"STOP"}: action=mandatory
        # Never let Groq override a mandatory evidence transition.
        if mandatory and mandatory not in {"STOP"} and action not in {mandatory}:
            action=mandatory
        return {"action":action,"target":target,"reason":data.get("reason","")}
    except Exception as e:
        return {"action":mandatory,"target":"","reason":f"Groq planner fallback: {e}"}

# ============================================================
# TARGET RESOLUTION
# ============================================================
def resolve_target(state, action, target=""):
    if target: return target
    if action=="lookup_event_fact":
        # temporal event result has priority
        for x in reversed(state.evidence):
            r=x.get("result",{})
            if x.get("tool")=="resolve_temporal_event" and r.get("event_id"): return r["event_id"]
        return selected_event(state)
    if action=="search_graph_event": return selected_event(state)
    if action=="inspect_source_document":
        for x in reversed(state.evidence):
            r=x.get("result",{})
            if r.get("doc_id"): return str(r["doc_id"])
        ev=selected_event(state); return ev[len("event_"):] if ev.startswith("event_") else ""
    if action=="search_graph_document":
        for x in reversed(state.evidence):
            r=x.get("result",{})
            if r.get("doc_id"): return str(r["doc_id"])
    return ""

# ============================================================
# TOOL EXECUTION
# ============================================================
def execute(action, question, target=""):
    if action=="discover_event": return discover_event(question)
    if action=="investigate_event_candidates":
        d=discover_event(question)
        return investigate_event_candidates(question,d.get("candidates",[]))
    if action=="search_graph_event": return search_graph_event(target)
    if action=="search_graph_document": return search_graph_document(target)
    if action=="inspect_source_document": return inspect_source_document(target)
    if action=="lookup_event_fact": return lookup_event_fact(question,target)
    if action=="resolve_temporal_edition": return resolve_temporal_edition(question)
    if action=="resolve_temporal_event": return resolve_temporal_event(question)
    if action=="aggregate_events": return aggregate_events(question)
    if action=="find_event_extreme": return find_event_extreme(question)
    return {"success":False,"error":f"Unknown action: {action}"}

# ============================================================
# FINAL ANSWER
# ============================================================
def generate_final_answer(question: str, state: TraceState) -> str:
    # Deterministic answers are returned directly where possible.
    for x in reversed(state.evidence):
        r=x.get("result",{})
        if x.get("tool") in {"aggregate_events","find_event_extreme"} and r.get("success"):
            return str(r.get("answer"))
    source=None; graph=[]
    for x in reversed(state.evidence):
        r=x.get("result",{})
        if x.get("tool")=="lookup_event_fact" and r.get("success"):
            source=r.get("source"); break
        if x.get("tool")=="inspect_source_document" and r.get("success"):
            source=r; break
    graph=state.graph_evidence[-3:]
    if not source: return "Evidence was insufficient to produce a grounded answer."
    text=str(source.get("text", ""))
    prompt=f"""Answer the user's question using ONLY the supplied source and graph evidence. Give the direct answer first. Do not invent facts.\nQUESTION:\n{question}\nSOURCE:\n{text[:8000]}\nGRAPH:\n{json.dumps(graph,default=str)[:5000]}"""
    r=groq_client.chat.completions.create(model=GROQ_MODEL,messages=[{"role":"user","content":prompt}],temperature=0,max_tokens=300)
    return r.choices[0].message.content.strip()

# ============================================================
# MAIN AGENT
# ============================================================
def trace_run(question: str, max_steps: int = MAX_STEPS) -> Dict[str,Any]:
    state=create_state(question)
    execution=[]
    for step in range(max_steps):
        state.step=step
        ev=evaluate(state)
        if ev["status"]=="SUFFICIENT":
            state.finished=True; state.current_stage="VERIFIED"; break
        decision=groq_plan(state)
        action=decision.get("action",""); target=decision.get("target","")
        if action=="STOP":
            state.current_stage="STOPPED"; state.stop_reason=decision.get("reason","Planner stopped."); break
        target=resolve_target(state,action,target)
        # Guard: actions requiring targets must have them.
        if action in {"search_graph_event","lookup_event_fact","inspect_source_document","search_graph_document"} and not target:
            state.current_stage="STOPPED"; state.stop_reason=f"No target available for {action}."; break
        result=execute(action,question,target)
        add_evidence(state,action,result)
        execution.append({"step":step,"action":action,"target":target,"success":bool(result.get("success"))})
        # Deterministic tools can finish immediately.
        if action in {"aggregate_events","find_event_extreme"} and result.get("success"):
            state.answer_candidate=str(result.get("answer")); state.finished=True; state.current_stage="ANSWERED"; break
        if action=="lookup_event_fact" and result.get("success") and result.get("source",{}).get("text"):
            state.finished=True; state.current_stage="ANSWERED"; break
    if not state.finished and state.current_stage!="STOPPED":
        state.current_stage="STOPPED"; state.stop_reason="Maximum steps reached."
    final=generate_final_answer(question,state) if state.finished else None
    return {"final_answer":final,"question_type":state.question_type,"initial_route":state.initial_route,"finished":state.finished,"final_stage":state.current_stage,"tool_history":state.tool_history,"execution_log":execution,"stop_reason":state.stop_reason}

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="TRACE Agentic GraphRAG using Groq + TigerGraph Cloud")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args=parser.parse_args()
    question=args.question or input("Question: ").strip()
    result=trace_run(question,args.max_steps)
    print("\n"+"="*80); print("TRACE RESULT"); print("="*80)
    print(json.dumps(result,indent=2,default=str))
