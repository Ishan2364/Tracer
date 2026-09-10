# Product Note — Tracer

## What it is

Tracer is a conversational study companion for a small catalogue of physics podcast
episodes. A learner asks questions in plain language and gets answers grounded
*only* in what the episodes actually say — every claim carries a citation back to
the exact timestamp it came from, and the product says so plainly when a question
isn't covered instead of guessing.

## Who it's for

A physics-curious learner who has 3–4 hours of dense, long-form audio they haven't
listened to yet, and no formal course structure to lean on. They don't want to
scrub through an hour of recording to find the one explanation they need, they
don't fully trust a general chatbot to get the physics right, and they want to be
able to check any answer against the source rather than take it on faith.

This is *not* built for someone who already knows the catalogue and wants a quick
lookup, and it's not a general physics tutor — it has no opinion about anything
outside these specific episodes.

## The problem I chose to solve

Long-form audio has two failure modes as a learning medium: you can't skim it, and
if an explanation doesn't land the first time, re-listening from the same spot is
the only tool you have. I chose to solve **"let the learner talk to the audio
instead of scrubbing through it, and let them verify anything they're told"** —
not "summarize the episodes" and not "answer physics questions in general."

That distinction matters more than it sounds. A summarizer would be faster to
build and would *feel* impressive on a demo, but it fails the trust bar the brief
sets: a summary can't be checked sentence-by-sentence against the source the way a
cited, retrieval-grounded answer can. I optimized for the latter even where it
made the product feel more conservative (see the refusal behavior below) — a
wrong but confident answer is a worse outcome for a learner than an honest "this
isn't covered."

## Why this scope

The brief explicitly flags source verifiability and responsible handling of
unsupported requests as graded dimensions, not nice-to-haves. Given a fixed API
budget and two days, I chose to spend the budget on making a **narrow set of
interactions trustworthy** — grounded single-episode Q&A, cross-episode
comparison, episode recommendation, and follow-ups that build on prior context —
rather than a wider feature set with weaker grounding. Concretely:

- **Every claim is traceable.** Citations are built from retrieval metadata, not
  parsed out of the model's free text, so a citation can never be hallucinated
  independently of what was actually retrieved.
- **Refusal is a first-class behavior, not an afterthought.** The product
  distinguishes "you named a specific episode" (which we already know exists)
  from "this is a catalogue-wide question" (where "not covered" is a real,
  checkable outcome) — and calibrated *when* to refuse against measured
  similarity scores on real in-scope vs. out-of-scope queries, not a guessed
  threshold.
- **Follow-ups reuse context instead of re-searching**, so "walk me through that
  again, slower" genuinely builds on what was just discussed rather than
  returning an unrelated fresh answer.
- **Comparison questions are answered per-episode, then synthesized** — if two
  named episodes are compared and only one actually covers the topic, the answer
  says so explicitly for the one that doesn't, rather than blending both into one
  undifferentiated (and partly invented) answer.

## What I deliberately did not build

- No episode-level summarization — retrieval-based grounding and "summarize
  everything this episode covers" are different problems with different
  reliability guarantees, and I didn't want to ship the weaker one under the same
  trust promise as the rest of the product.
- No persistence of conversation state across sessions, no multi-user support, no
  authentication — out of scope per the brief, and not where the learner value is.
- No fine-tuning or custom speech/language model — the catalogue is small enough
  that off-the-shelf ASR, local embeddings, and a hosted LLM are the right tool,
  not a reason to spend the API budget on training.

## What "done" looks like

A learner can hold a real back-and-forth across all episodes in the catalogue,
every substantive claim in a response can be checked against a specific
timestamp, and asking about something the catalogue doesn't cover produces an
honest "not covered" — never an invented answer.
