# TRACE — Adaptive Agentic GraphRAG

## 1. Overview

TRACE, or **The Adaptive Reasoning & Context Engine**, is an adaptive Agentic GraphRAG framework designed to answer complex knowledge questions by intelligently selecting the appropriate level of retrieval and reasoning. The system compares three progressively capable approaches: **RAG-v2, GraphRAG, and Agentic GraphRAG**, with the objective of understanding when conventional retrieval is sufficient, when structured graph reasoning is beneficial, and when multi-step agentic investigation is actually required.

The core idea behind TRACE is simple: **not every question needs an agent**. A straightforward factual lookup can be handled through direct source retrieval, while relational or multi-hop questions may require graph traversal and deeper investigation. TRACE therefore treats reasoning as an adaptive resource and escalates the query only when the available evidence is insufficient to produce a reliable answer.

## 2. RAG-v2: Hybrid Retrieval Foundation

The **RAG-v2 pipeline** acts as the enhanced text-retrieval baseline. It combines semantic vector retrieval with BM25 lexical retrieval and uses Reciprocal Rank Fusion to merge the retrieved candidates. A cross-encoder reranker then improves the ordering of relevant passages, followed by context expansion to provide additional surrounding evidence before answer generation. This creates a strong retrieval baseline without introducing graph-based reasoning or autonomous planning.

RAG-v2 is particularly useful for questions where the required information exists directly within the textual corpus and does not require complex relationships between multiple entities. It provides the foundation against which the additional capabilities of GraphRAG and TRACE can be evaluated.

## 3. GraphRAG: Structured Knowledge Reasoning

The **GraphRAG pipeline** extends conventional retrieval by introducing structured relationships through a TigerGraph knowledge graph. The Olympic knowledge graph contains entities such as documents, events, Olympic Games, locations, people, countries, medals, and organizations, together with relationships such as `PART_OF`, `HELD_AT`, and document-mention relationships. GraphRAG uses these relationships to establish structural context around an event or entity and then combines that graph-derived context with the original source document before generating the final answer.

The current graph is primarily **event-centric and source-document grounded**. In particular, the graph provides reliable relationships between events, Olympic Games, locations, and source documents, while detailed factual information such as medal results is obtained from the underlying corpus. This separation is intentional: the knowledge graph establishes structural relationships, while the original documents remain the authoritative textual evidence layer.

## 4. TRACE: Agentic Reasoning Layer

**TRACE introduces the agentic layer on top of these retrieval capabilities.** Instead of following a fixed retrieval sequence, TRACE maintains an investigation state containing the question, question type, current evidence, previous tool results, selected targets, and reasoning history. A Groq-powered planner analyzes this state and selects the next action from a collection of specialized tools. The system can therefore change its investigation strategy dynamically as new evidence becomes available.

The agent uses **Groq with the `openai/gpt-oss-120b` model** for planning and answer synthesis. The model is not expected to perform every operation itself. Instead, it determines which operation should happen next and interprets the resulting evidence. Deterministic operations such as aggregation, event comparison, and structured filtering are delegated to dedicated tools, reducing unnecessary generative reasoning and making those operations more reproducible.

## 5. Specialized Investigation Tools

TRACE contains specialized capabilities for direct event lookup, event discovery, candidate investigation, temporal event resolution, TigerGraph event search, document-level graph search, source-document inspection, aggregation, and superlative queries. This tool-based design allows the system to break a complex question into smaller verifiable operations instead of relying on a single LLM response.

Each tool performs a specific role within the investigation process. Rather than asking the language model to solve the entire problem internally, TRACE allows the planner to delegate factual retrieval, graph traversal, filtering, comparison, and source inspection to specialized components.

## 6. Multi-Hop Reasoning

One of the most important components is the **multi-hop event resolver**. Questions may describe an event indirectly through a combination of venue, date, Olympic edition, sport, or other contextual information. TRACE first extracts these constraints, discovers potential candidates, evaluates their compatibility, and identifies the most appropriate event. The selected event can then be verified against TigerGraph relationships before the system retrieves the corresponding source document.

This approach allows TRACE to handle questions where the answer cannot be obtained reliably through a single retrieval operation. The system progressively connects different pieces of evidence until the original question can be grounded in a specific event and its corresponding source.

## 7. Temporal Reasoning

Temporal reasoning is handled as a separate capability because Olympic questions frequently contain relative or historical references. A question such as asking about the Summer Olympics immediately before a particular year requires the system to resolve the correct Olympic edition before attempting to identify the specific event or answer.

TRACE therefore separates temporal resolution from ordinary lexical retrieval and uses the resolved temporal context during subsequent investigation. This allows historical relationships and relative time references to become explicit constraints within the reasoning process.

## 8. Evidence Evaluation

The **Evidence Evaluator** is the central control mechanism that distinguishes TRACE from a simple tool-calling pipeline. After every significant investigation step, TRACE evaluates whether the collected evidence is sufficient to answer the original question.

It checks whether the required event or entity has been resolved, whether the graph relationships support the candidate, whether source-level evidence is available, and whether important constraints from the question have been satisfied. If the evidence is sufficient, the system stops and generates the answer; otherwise, it sends the current state back to the planner for another investigation step.

## 9. Adaptive Re-Planning

TRACE creates a closed-loop reasoning process in which the agent does not simply execute a predetermined chain of tools. The system can discover an event, inspect its graph relationships, retrieve the source document, recognize that additional evidence is required, and then select another tool.

This dynamic re-planning is particularly important for multi-hop questions where the correct sequence of operations cannot always be known in advance. The agent continuously evaluates the current evidence and determines whether another reasoning step is justified.

## 10. Intelligent Query Routing

TRACE introduces a **question-specific routing strategy**. Lookup questions can be resolved through direct evidence retrieval, temporal questions can invoke temporal resolution, aggregation questions can use deterministic counting, superlative questions can use deterministic comparison, and multi-hop questions can trigger event discovery, investigation, graph verification, and source inspection.

The system therefore attempts to match reasoning complexity with question complexity rather than treating every query identically. This is central to TRACE's adaptive design because unnecessary agentic reasoning increases computational cost without necessarily improving the answer.

## 11. Knowledge Corpus

The underlying Olympic corpus contains approximately **2,951 documents**, with each document containing identifiers, title, URL, Wikidata information, Wikipedia page information, approximate token count, and source text. This corpus acts as the textual foundation shared across the retrieval systems and allows the three pipelines to be evaluated against the same underlying knowledge source.

Using the same corpus across RAG-v2, GraphRAG, and TRACE ensures that the comparison focuses on differences in retrieval, graph reasoning, and agentic orchestration rather than differences in the underlying information source.

## 12. Three Levels of Reasoning

The three approaches represent three different levels of reasoning capability. **RAG-v2 focuses on retrieving the right textual evidence efficiently, GraphRAG adds structured relationships to improve contextual understanding, and TRACE adds adaptive planning and iterative investigation on top of both.**

This progression makes it possible to study the trade-off between answer quality and reasoning complexity. A simple query may require only textual retrieval, while a relational question may benefit from graph reasoning, and a difficult multi-hop question may require iterative agentic investigation.

## 13. Benchmarking and Evaluation

The benchmark evaluates the pipelines using common evaluation questions covering **lookup, temporal, multi-hop, aggregation, and superlative reasoning**. The primary metrics include accuracy, completeness, retrieval performance, token usage, latency, and, for TRACE, tool success and reasoning-step behaviour.

Using the same evaluation corpus across all three systems allows the project to examine whether additional graph and agentic reasoning provides measurable benefits relative to the increased computational and token cost.

## 14. Core Contribution

The key contribution of TRACE is not simply the addition of an LLM agent to GraphRAG. Its contribution is the introduction of **adaptive reasoning orchestration**, where retrieval, graph reasoning, specialized computation, evidence evaluation, and agentic investigation are combined into a single decision-making framework.

TRACE attempts to answer not only *what information should be retrieved*, but also *whether more reasoning is actually necessary*. This makes the framework focused on evidence-driven reasoning rather than blindly applying the most complex architecture to every query.

## 15. Overall System Philosophy

Ultimately, TRACE represents a progression from **retrieval to structured reasoning to adaptive investigation**. RAG-v2 retrieves evidence, GraphRAG connects evidence through relationships, and TRACE determines how much reasoning is required, selects the appropriate tools, evaluates the resulting evidence, and continues investigating only when necessary.

The goal is a system that is not merely more complex than conventional RAG, but **more selective, evidence-driven, and context-aware in deciding when additional reasoning is justified**.
