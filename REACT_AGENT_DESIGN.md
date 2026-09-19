# ReAct Agent Architecture — Design Notes

This is a design document, not an implemented feature yet. It describes how Tracer's
retrieval/answering pipeline would be rearchitected from a fixed, hand-coded workflow
into a single ReAct-style agent, orchestrated with LangGraph/LangChain and observed via
LangSmith. Nothing in this document has been built - it's the plan to build against.

**Verification note (2026-09-19):** the implementation sections below were checked
against current, live documentation before being written, specifically because
LangChain's agent-building API has changed under itself recently and stale tutorials
are common. The single most important correction this produced: `create_react_agent`
(from `langgraph.prebuilt`) - the function most tutorials, StackOverflow answers, and
older training data default to - **is deprecated**. The current, correct entrypoint is
`create_agent` from the `langchain` package itself. Every code shape below reflects the
verified, current API, not the deprecated one. Sources are listed at the bottom of this
document.

---

## 1. Why change anything

**Today's architecture is a fixed decision tree, decided once, upfront, per turn:**

```
query -> intent_parser.classify (1 of 8 labels) -> retrieval_router.route (1 of 4 mechanisms)
       -> generate.build_messages (1 of 4 prompt shapes) -> one LLM call -> done
```

Every decision is made *before* seeing what retrieval actually returns. This has caused
real, concrete problems already found in this project:

- **The 8-category classifier mostly collapses.** 4 of 8 labels (`general`,
  `broad_comparison`, `recommendation`, an unresolvable `named_comparison`) route through
  the identical retrieval call. The taxonomy adds classification cost and fragility
  without adding proportional behavioral difference.
- **No recovery from a bad first retrieval.** If a broad search comes back weak, the
  fixed pipeline either refuses or answers thinly - it cannot try a different phrasing,
  narrow the scope, or ask a clarifying question. This caused a real bug: "recommend me
  an episode" (no topic) embedded as a near-empty vector, scored below the refusal
  threshold, and got a flat "not covered" refusal - even though nothing was actually
  uncovered, there was just no topic to search for yet.
- **Follow-up handling is a hard-coded special case**, not a capability. `query_type ==
  "follow_up"` triggers a fixed reuse-and-expand rule (`expand_context()`), which needed
  three separate bug fixes (history truncation, a too-narrow prompt definition, and a
  budget-trim ordering bug) to work correctly for one specific scenario. Each fix patched
  a fixed rule that was trying to simulate judgment a reasoning loop could exercise
  directly.
- **Retrieval depth is fixed at 0 or 1 calls per turn.** A question that genuinely
  benefits from checking two different scopes (e.g. "is this covered anywhere, and if
  so which episode is best") cannot do that today.

A ReAct loop replaces "decide once, upfront" with "decide, act, observe, decide again -
as many or as few times as the question actually needs."

---

## 2. The core loop

**Reason -> Act -> Observe, repeated until the agent decides it has enough grounding to
answer (or that nothing will help, and it should say so).**

```
┌─────────────────────────────────────────────────────────┐
│  agent node (LLM + tools bound)                          │
│  - reads: conversation history, accumulated observations │
│  - decides: call a tool, or produce the final answer      │
└───────────────┬────────────────────────┬────────────────┘
                │ tool call requested      │ no tool call
                ▼                          ▼
      ┌───────────────────┐         ┌─────────────┐
      │  tools node        │         │    END      │
      │  executes the tool,│         │ (final      │
      │  appends result as │         │  answer)    │
      │  an observation    │         └─────────────┘
      └─────────┬──────────┘
                │ loop back
                ▼
        (back to agent node)
```

No upfront classification step. Tool *selection*, made fresh by the agent each time,
replaces what `intent_parser.py` used to decide once and lock in.

---

## 3. Tools — thin wrappers over what already exists

The existing retrieval functions barely change; they get exposed as callable tools via
LangChain's `@tool` decorator (`from langchain.tools import tool`) instead of being
dispatched from a fixed lookup table. The decorator reads the function's type-annotated
signature and docstring to build the tool schema the model sees - so the docstring isn't
just documentation here, it's the model's only description of what the tool does.

```python
from langchain.tools import tool
import retrieve as retrieve_mod
import retrieval_router

@tool
def search_episode(episode_number: int, query: str) -> str:
    """Search for content within one specific episode. Use this when the user named
    a specific episode, or you've narrowed a broad search down to one episode and want
    more detail from it. Returns up to 5 matching excerpts with their timestamps and
    a similarity score for each (0-1, higher is more relevant)."""
    episode_id = resolve_episode_id(episode_number)  # existing episode_resolver.py logic
    chunks = retrieve_mod.retrieve_scoped(query, episode_id)
    return format_chunks_for_agent(chunks)  # includes chunk_id, timestamps, similarity, text

@tool
def search_all(query: str) -> str:
    """Search across the entire episode catalogue when no specific episode is implied,
    or for a broad/comparison question. Returns up to 25 matches, capped at 3 per
    episode so one episode can't dominate the results, each with a similarity score."""
    chunks = retrieval_router.rebalance(retrieve_mod.retrieve(query, n_results=25))
    return format_chunks_for_agent(chunks)

@tool
def search_timerange(episode_number: int, query: str, start_seconds: float, end_seconds: float) -> str:
    """Search within one episode, restricted to a specific time window. Use only when
    the user named an explicit time range."""
    ...

@tool
def get_neighboring_chunks(chunk_id: str) -> str:
    """Fetch the chunk immediately before and after a given chunk_id, for when you need
    more surrounding context around something you already found - e.g. the user asked
    for more detail or a slower walkthrough of something already retrieved."""
    ...

@tool
def list_episodes() -> str:
    """List every episode in the catalogue with its number and title. Use this for
    inventory questions, or when a recommendation request didn't name any topic and you
    need to either ask what they're interested in or let them browse."""
    ...
```

Note each tool's own docstring already carries the "when to use this" guidance that used
to live in `intent_parser.py`'s 8-category prompt - that instruction moves from a
separate classification step into the tools themselves, read by the same model that
decides whether to call them.

---

## 4. Building the agent — `create_agent`, not a hand-rolled `StateGraph`

**Correction from an earlier draft of this document:** the obvious way to build this -
manually wiring a `StateGraph` with a hand-written `agent_node`/`tools_node` and a custom
`should_continue` conditional - is not wrong, exactly, but it's reinventing something
LangChain already provides and maintains. It's also not what a working version of this
would use as a starting point. Verified against current documentation: the correct,
current entrypoint is `create_agent` from the `langchain` package.

**Note on naming, because it's a real trap:** `langgraph.prebuilt.create_react_agent` -
the function almost every existing tutorial, StackOverflow answer, and blog post
demonstrates - is now **deprecated** in favor of `create_agent` from `langchain` itself.
Anything referencing `create_react_agent` found while implementing this should be treated
as describing the old API.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
# (a persistent checkpointer - e.g. SqliteSaver - would replace InMemorySaver for
#  anything beyond local development; see §7 for how this maps onto the project's
#  existing session_id/ConversationState mechanism)

SYSTEM_PROMPT = """You are Tracer, a physics study companion for a small podcast
catalogue. Answer only using what your tools return - never from your own knowledge.
Every claim must carry a citation in the form (Episode N, mm:ss-mm:ss), taken directly
from a tool result. If nothing you find actually answers the question, say so plainly.
..."""  # carries forward generate.py's existing SYSTEM_PROMPT rules, largely unchanged

agent = create_agent(
    model="groq:openai/gpt-oss-120b",   # same generation model as today; see note below on cost
    tools=[search_episode, search_all, search_timerange, get_neighboring_chunks, list_episodes],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": session_id}}  # session_id is already a first-class
                                                       # concept in this project - see §7
result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config=config,
)
final_answer = result["messages"][-1].content
```

`create_agent` builds a `CompiledStateGraph` under the hood - it's still LangGraph, just
with the model-call / tool-call / loop-back wiring already implemented correctly rather
than hand-rolled. If a need arises later that the built-in loop genuinely can't express
(some very specific mid-loop control-flow), dropping to a manual `StateGraph` is still
possible - `create_agent` is the correct *starting point*, not a ceiling.

**Iteration safety - verified, and importantly *not* what an earlier draft of this
document assumed.** There's no need to hand-roll an `iteration_count` state field and a
custom cutoff check - LangGraph's compiled graph already enforces a `recursion_limit`
(default 25) on any invocation, raising `GraphRecursionError` if exceeded. For this
project, that default is almost certainly too generous (25 tool-call round-trips for an
8-episode catalogue would be a clear sign something's wrong, not legitimate multi-hop
reasoning) - it should be lowered explicitly per call:

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={"configurable": {"thread_id": session_id}, "recursion_limit": 8},
)
```

**One real gap to design around, not paper over:** hitting `recursion_limit` raises an
exception - it does not let the agent gracefully wrap up with a partial answer. A
production version needs to catch `GraphRecursionError` and fall back to an honest
message (e.g. "I wasn't able to narrow this down - try being more specific about which
episode or topic you mean") rather than letting the exception surface as a raw error to
the user. This is a real implementation detail to get right, not an afterthought.

---

## 5. Preserving the citation guarantee — the part that must not regress

This is the single most important constraint on the whole redesign. Today's system
guarantees citations are never hallucinated because they're built from retrieval
metadata, never parsed from the model's free text. That guarantee must survive the move
to a multi-step loop, and it must survive more carefully than my first instinct handled
it.

**What doesn't work:** having the model self-report which chunks it "used" in a
structured field. This just moves the trust problem up one level - trusting the model to
introspect its own usage isn't more reliable than trusting it to copy a timestamp
correctly.

**What actually works, proven by a real bug found and fixed in the current codebase
during this same design conversation:** citations must be built from exactly what was
sent to the model, not from everything that was ever retrieved. Concretely, this project
already had this exact class of bug - `_fit_chunks_to_budget()` trimmed the chunk list
locally inside `generate.build_messages()`, and the caller kept building citations from
the pre-trim list. Verified with a synthetic 10-chunk, 20,000-char case: only 4 chunks
fit the budget, but citations were being built from all 10 - 6 phantom citations for
chunks the model never saw. Fixed by having every `build_messages*()` function return
`(messages, chunks_actually_sent)`, and building citations only from `chunks_actually_sent`.

**How this maps onto `create_agent`'s actual state model - and where an earlier draft of
this document overreached:** `create_agent`'s state is just `messages: list[BaseMessage]`
(verified above) - there is no separate `accumulated_chunks` field handed to you for
free. Every tool call's result becomes a `ToolMessage` appended to that same list. So the
concrete, verified-safe plan is: **after `agent.invoke()` returns, walk `result["messages"]`
for every `ToolMessage` added during *this* invocation (i.e. after the new human message,
not earlier turns pulled in from thread history), and build citations from the union of
whatever real chunk metadata those tool calls returned.** That's every chunk the model
actually had in front of it when it wrote the final answer - the same principle as
today's `chunks_sent`, generalized from "one generation call" to "every tool result
visible in context by the time the final answer was produced."

**One honest, carried-forward imprecision, not a new regression:** if the agent runs two
tool calls in one turn and only actually draws on one of them in its prose, this approach
still cites both - it can't distinguish "was in context" from "was actually quoted,"
same as today's `_build_citations()` already can't (it cites every chunk handed to the
model, not just the ones its prose references). That's a pre-existing, accepted
imprecision (see the earlier conversation on why an LLM self-reporting "chunks I used"
isn't more trustworthy than this) - not something this redesign makes worse, but also not
something it fixes for free. One additional edge case worth flagging for whoever
implements this: if a context-management middleware (e.g. `SummarizationMiddleware`, for
very long tool-call chains) ever compresses or drops older tool messages before the final
answer, the citation-building step must run *before* any such compression happens, or
account for it - otherwise a citation could point to a chunk that got summarized away
by the time the answer was written. Worth a specific test once this is built, not
something to assume away.

---

## 6. Refusal — a signal, not a hard gate

Today: `SIMILARITY_THRESHOLD` (auto-calibrated per index build against off-topic canary
queries) is a hard cutoff - below it, refuse, unconditionally.

In the agentic version, this threshold becomes **information returned in a tool's
observation** ("best match similarity: 0.31, calibrated off-topic ceiling: 0.42"),
which the agent reasons over rather than being gated by. This is a deliberate tradeoff:

- **Gains:** the agent can decide "this is below the ceiling, but the phrasing might be
  off - let me try rephrasing and searching again" (query reformulation), something the
  hard gate structurally could not do.
- **Costs:** determinism. The hard gate always refuses the same way for the same
  similarity score; an agent's decision to retry, narrow, or refuse is judgment-based and
  can vary. This should be named plainly if asked - it's the real price of moving from a
  rule to a reasoning step.

The calibration mechanism itself (`build_index.py`'s canary-query measurement) doesn't
change - only how its output gets *used* downstream changes, from a gate to a signal.

---

## 7. Follow-ups and conversation memory become a capability, not a special case

Today, `query_type == "follow_up"` triggers a hard-coded rule: reuse
`state.last_retrieved_chunks`, expand with neighbors, skip fresh retrieval
unconditionally.

In the agentic version, the agent has the last `HISTORY_TURNS` turns' plain Q+A text
(see the implementation note below - not the raw tool-call history) and reasons about
it directly:
- If that text already contains enough to answer ("explain that more simply"), it
  answers directly without calling any tool at all - verified, see below.
- If it needs more than what's in that text - including going deeper than a prior
  answer's own content - it searches again (`search_episode`/`search_all`), since
  `get_neighboring_chunks()` can only reference a chunk_id found by a tool call *within
  the current turn* (the capped history holds text, not chunk objects, so there's
  nothing from a prior turn to hand it). This is a deliberate, accepted tradeoff of the
  capped-history design over the (rejected, see below) full-checkpointer approach.
- If the question is actually a topic switch disguised as a short follow-up-looking
  phrase, the agent can recognize that and call a fresh search instead - this is exactly
  the boundary that caused the three-layered "Tiktaalik" bug in the fixed pipeline
  (history truncation hid the referent, the follow_up prompt definition was too narrow,
  and a budget-trim ordering bug dropped the relevant chunk even after correct
  classification). All three of those were bugs in a rule *simulating* judgment; an agent
  exercises the judgment directly instead - verified with a real topic-switch turn, see
  below.

**Implemented and verified - this is NOT what the paragraph above originally proposed.**
`create_agent`'s `checkpointer` mechanism does work as described (a `thread_id`
automatically restores the *entire* raw message history, including every past tool
call's full retrieved text, before each new invocation) - but that's exactly the
problem: verified empirically (dumped the actual message list after 2 turns) that this
means every turn re-sends every prior turn's full tool results forever, unbounded, for
the life of a session. That's a real, structural cost, not a hypothetical one.

**What's actually built instead:** no checkpointer at all. A small module-level dict in
`agent.py` (`_history`, keyed by `thread_id`) keeps only the last `HISTORY_TURNS = 2`
turns as plain `{query, answer}` text pairs - the exact same approach
`intent_parser.py` already used for the fixed pipeline's classifier, including the same
"don't truncate the answer text" lesson learned from the Tiktaalik bug's first root
cause. Each turn, this capped history is formatted into the *user message itself*
(`"Recent conversation:\n{...}\n\nCurrent question: {query}"`), and the system prompt
explicitly tells the agent to judge from that text whether it already has enough to
answer or needs to search again - rather than a hard-coded rule forcing either path.

Verified with a real 3-turn conversation:
- Turn 1 ("what does episode 6 say about the equivalence principle?"): 5 tool calls, 5 citations.
- Turn 2 ("explain that in short and concise"): **0 tool calls, 0 new citations** - answered directly from the capped history text, exactly as intended.
- Turn 3 (topic switch to episode 8's Wallace line): fresh, uncontaminated search - no bleed-through from the relativity history.

This also sidesteps the `session_store.py`-vs-checkpointer storage-format question
raised in the paragraph this replaced - there's no checkpointer to choose a backend
for. `session_id` still maps directly to the `_history` dict's key, same identity
mapping as originally proposed, just simpler underneath.

**Known residual gap, not yet fixed:** when the agent answers purely from history text
(turn 2 above), the *structured* `citations` returned to the API are correctly empty
(nothing new was retrieved this turn) - but the answer's own prose still repeats a real
citation from the shown prior-turn text. That's intentional and honest (see the system
prompt's citation rule), but it means a frontend relying only on the structured
`citations` array won't show a citation chip for a turn like that, even though the
prose has one. Not addressed here; worth deciding if it matters once observed in
practice.

---

## 8. What retires

- **`intent_parser.py`'s 8-category classifier** - retired. Tool selection, made fresh
  each time by the agent, replaces upfront classification. This is the direct resolution
  to the finding that 4 of the 8 categories had no distinguishing behavior.
- **`retrieval_router.py`'s fixed `route()` dispatch table** - retired. Its logic
  (scoped vs. broad vs. time-ranged vs. follow-up) becomes the tool docstrings and the
  agent's own reasoning about which tool(s) to call.
- **The hard-coded `follow_up` branch and forced neighbor-expansion** - retired as a
  forced rule, kept as an on-demand tool (`get_neighboring_chunks`).
- **`_handle_recommendation_no_topic()`'s special-cased early return** - retired as a
  special case; a topic-less recommendation becomes a natural consequence of the agent
  reasoning "I don't have enough information yet" and calling `list_episodes()` or
  asking a clarifying question, rather than a hand-written detection branch.

## 9. What does not change

- **The indexing pipeline is completely untouched**: Deepgram transcription, chunking
  (`chunk_transcripts.py`), embedding (`bge-base-en-v1.5`), and the Chroma index itself.
  This redesign only touches the retrieval *orchestration* layer, not how content gets
  indexed.
- **The manifest and episode metadata** - unchanged, still the deterministic source of
  truth for catalogue questions.
- **The chunk-metadata-based citation-building principle** - unchanged in spirit (see
  §5); what changes is exactly *which* chunk set citations get built from.
- **The calibrated similarity threshold's calculation** - unchanged; only its role
  downstream changes (gate -> signal, see §6).

---

## 10. LangSmith observability

The codebase already uses `@traceable` throughout (`parse_intent`, `retrieve*`,
`answer_conversational`, `groq_generate`), so tracing discipline already exists, and
`config.py` already sets `LANGSMITH_PROJECT` and reads `LANGSMITH_TRACING`/
`LANGSMITH_API_KEY` from `.env` - **no new environment variables are needed.** Verified:
LangGraph/LangChain auto-detect these same variables, so an agent built with
`create_agent` gets traced the same way, automatically - every node execution, every
tool call, and the full agent trajectory (reasoning text, tool calls, observations, in
order) shows up as one structured, nested trace per turn, with zero additional
instrumentation beyond what's already configured.

This is a genuine upgrade over today's `--debug` output: instead of seeing which fixed
branch fired, you see the agent's actual reasoning at each step - directly useful for
the "observe the agent's behavior and correct it" workflow this redesign is meant to
enable. Concretely, watch for in early traces:
- How many tool calls a typical query actually takes (is the iteration cap ever hit?)
- Whether the agent ever calls a tool with a redundant/near-duplicate query
- Whether refusal-worthy queries get reasoned through correctly now that the threshold
  is a signal, not a gate (this is the one area most likely to need prompt tuning after
  the initial build, precisely because it traded determinism for judgment)

---

## 11. Open questions to resolve during implementation, not before

- Exact system prompt wording - this will need iteration once real traces are visible in
  LangSmith, not something to over-specify upfront.
- What `recursion_limit` value is actually right for this catalogue's size (start low,
  e.g. 6-8, and raise only if real traces show legitimate multi-hop queries hitting it).
- Whether per-tool-call chunk-count/token limits are needed on top of `recursion_limit` -
  e.g. capping how many chunks a single `search_all` call can return, independent of how
  many tool calls happen overall.
- How `eval/cases.md`'s 15 cases get re-validated against the new architecture - the
  cases themselves (grounded/uncovered, comparison, ambiguous title, time-range,
  follow-up, chitchat, catalogue) stay valid as test scenarios; what changes is that
  `query_type`/`retrieval_mode` are no longer meaningful fields to log, and the eval
  harness will need new observability fields (number of tool calls, which tools, in what
  order) to grade against instead.
- Whether `middleware=[]` (e.g. `SummarizationMiddleware` for long tool-call chains,
  `ToolRetryMiddleware` for transient Chroma/Groq errors) is worth adopting immediately
  or only if a real need shows up in traces - these exist and are documented, but adding
  them upfront without a demonstrated need would be the same mistake as the original
  8-category classifier: complexity added ahead of evidence it's needed.

---

## Sources consulted for this verification pass (2026-09-19)

- [create_react_agent | langgraph.prebuilt | LangChain Reference](https://reference.langchain.com/python/langgraph.prebuilt/chat_agent_executor/create_react_agent) - confirms deprecation, points to `create_agent`
- [Agents - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/agents) - current `create_agent` signature, tool definition convention, checkpointer/thread_id mechanism, middleware system, structured output
- [GRAPH_RECURSION_LIMIT - Docs by LangChain](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT) - `recursion_limit` default (25) and config mechanism
- [LangGraph Tutorial: Build a Working ReAct Agent with the v1.0 API](https://agentsindex.ai/blog/langgraph-tutorial) - confirms LangGraph v1.0 stable (Oct 2025) as the version baseline for everything above
