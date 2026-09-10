# Eval Case Set — Questions and Ground-Truth Answers

Reference answers for all 12 cases in `cases.json`, written from the actual transcripts
(`/transcripts/chunks/`), independent of what Tracer generates — this is the ground
truth the eval harness's checks grade against, not a record of any one run's output.

---

## Case 1 — single_episode, covered
**Q:** According to episode 1, what method did Einstein propose for synchronizing two clocks?

**Expected:** Send a light flash from clock A to clock B, note when it leaves and when it
returns after bouncing back. Since light travels at the same speed both ways, define the
two clocks as synchronized when the light takes exactly as long to go as to come back —
i.e. split the round trip exactly down the middle. *(Episode 1, ~11:34–13:00)*

**Coverage:** Episode 1 — ✅ covered

---

## Case 2 — single_episode, covered
**Q:** According to episode 3, how many hydrogen bonds hold guanine and cytosine together, compared to adenine and thymine?

**Expected:** Adenine (A) pairs with thymine (T) via **2 hydrogen bonds**. Guanine (G)
pairs with cytosine (C) via **3 hydrogen bonds**. *(Episode 3, ~18:50–20:18)*

**Coverage:** Episode 3 — ✅ covered

---

## Case 3 — single_episode, uncovered (basic refusal)
**Q:** According to episode 1, how many hydrogen bonds hold guanine and cytosine together?

**Expected:** Episode 1 is about special relativity (time dilation, muons, clock
synchronization, the ether) and never discusses DNA, base pairing, or hydrogen bonds at
all — verified via direct keyword search (zero occurrences). The answer should explicitly
say episode 1 doesn't cover this, not attempt to answer from unrelated content.

**Coverage:** Episode 1 — ❌ uncovered

---

## Case 4 — named_comparison, both covered (theory vs. experiment)
**Q:** Compare episodes 1 and 2 - in both cases, was the breakthrough driven by a new experiment, or by working through existing theory more carefully?

**Expected:** Both are theory-first discoveries, not experiment-driven:
- **Episode 1:** The podcast opens by noting Einstein's 1905 paper involved "no new
  experiment, no new particle, barely any references" — pure theoretical reasoning.
  *(~00:00–01:27)*
- **Episode 2:** Hawking set out to *disprove* black hole radiation ("show the
  temperature was zero"), and instead "the mathematics handed him a clean, nonzero
  temperature" — a calculation that overturned his own expectation. *(~14:42–16:04)*

A good answer names this shared theme explicitly rather than manufacturing an artificial
difference.

**Coverage:** Episode 1 — ✅, Episode 2 — ✅

---

## Case 5 — named_comparison, both covered (reliable copying/transmission)
**Q:** Compare episodes 3 and 4 - what rule or mechanism does each one describe that makes reliably copying or transmitting information possible?

**Expected:**
- **Episode 3:** The strict A–T / G–C base-pairing rule is what lets DNA be copied
  faithfully during replication (heavily discussed — 38 "copy/copies" mentions).
- **Episode 4:** Redundancy and the channel-capacity limit (Shannon–Hartley:
  C = B·log₂(1+SNR)) are what let information be transmitted reliably over a noisy
  channel (14 "redundan-" mentions).

**Coverage:** Episode 3 — ✅, Episode 4 — ✅

---

## Case 6 — named_comparison, PARTIAL coverage (the hard case)
**Q:** What do episodes 1 and 3 say about hydrogen bonds in molecular structure?

**Expected:** Episode 3 covers this directly (see Case 2's answer). Episode 1 never
mentions hydrogen bonds at all (special relativity has nothing to do with molecular
chemistry) — verified via keyword search (0 occurrences in episode 1 vs. 9 in episode 3).
A good answer cites episode 3 concretely **and** explicitly states episode 1 doesn't
cover it — it must not stay silent on episode 1, and must not refuse the whole question
just because one of the two named episodes has nothing to say.

**Coverage:** Episode 1 — ❌ uncovered, Episode 3 — ✅ covered

**Observed variance (baseline/v2/manual runs):** the model sometimes attaches an
in-text citation to its own "episode 1 doesn't mention this" sentence (citing the
specific chunk it checked), and sometimes doesn't. Both readings are defensible — it's
citing evidence for a true negative, not fabricating a positive claim — but only the
no-citation form currently passes the `no_uncovered_citation` hard check, which can't
yet distinguish "cites evidence for a true negative" from "cites evidence for a false
claim." That distinction needs the judge-based `claim_support` check.

---

## Case 7 — broad_comparison, covered
**Q:** What physical or mathematical limits come up across all the episodes?

**Expected:**
- **Episode 1:** the speed of light as an absolute limit (space and time adjust so it
  never changes).
- **Episode 2:** the second law of thermodynamics (entropy never decreases) and
  Bekenstein's horizon-area-as-entropy argument.
- **Episode 4:** Shannon's compression floor (can't compress below entropy) and channel
  capacity ceiling (can't transmit reliably above capacity).
- **Episode 3:** has no comparable central "limit" concept — a good answer may
  legitimately have little or nothing to say about it here, rather than forcing one in.

**Coverage:** Episode 1 — ✅, Episode 2 — ✅, Episode 3 — ❌ (no real "limit" theme), Episode 4 — ✅

**Known gap found via manual run (verified against raw chunk text):** the retrieved
episode-2 chunk is a single continuous span, `262.6s-352.5s` (04:23-05:53), covering
both the second-law claim and the Bekenstein horizon-area claim. A model response split
this into two invented sub-citations, `(04:23-04:30)` and `(04:31-05:10)` — precision
that doesn't correspond to any real chunk boundary. The facts themselves were genuinely
in that chunk (verified), so this isn't a hallucinated fact, just fabricated timestamp
granularity. No current hard check catches this (`citation_grounded` only verifies the
`chunk_id` is real, not that in-text sub-timestamps stay inside that chunk's actual
`[start, end]`). Candidate future hard check: parse every in-text `(Episode N,
mm:ss-mm:ss)` and confirm it falls within some retrieved chunk's real range for that
episode.

---

## Case 8 — broad_comparison, uncovered
**Q:** What do the episodes say about how governments regulated nuclear weapons development?

**Expected:** Not covered anywhere — verified via keyword search (zero "nuclear weapon"
occurrences across all 4 episodes). None of the four papers (relativity, Hawking
radiation, DNA structure, information theory) touch nuclear policy. The system should
refuse rather than stretch tangentially-related physics content into an answer.

**Coverage:** Episode 1 — ❌, Episode 2 — ❌, Episode 3 — ❌, Episode 4 — ❌ (all uncovered)

---

## Case 9 — recommendation, covered
**Q:** Which episode should I listen to if I want to understand entropy in the context of black holes?

**Expected:** **Episode 2** — Bekenstein's argument that a black hole's horizon area is
literally its entropy (54 "entropy" mentions, far more than any other episode, and the
only episode that frames entropy specifically around black holes).

**Coverage:** Episode 2 — ✅ (unambiguous best recommendation for this specific framing)

---

## Case 10 — follow_up (2-turn), covered
**Turn 1 Q:** What did Einstein realize about simultaneity?
**Turn 2 Q:** Walk me through that step by step, I did not follow

**Expected:**
- **Turn 1:** Simultaneity is relative to the observer's motion — no universal "now."
  Grounded in episode 1's discussion of clock synchronization via light signals.
- **Turn 2:** Must build on turn 1's exact grounding (same chunks, possibly with a bit
  more surrounding context) rather than running a fresh unrelated search, and should
  break the explanation into smaller, slower steps per the "learner didn't follow"
  framing — not just repeat turn 1's answer.

**Coverage:** Episode 1 — ✅ covered

---

## Case 11 — named_comparison, references a nonexistent episode
**Q:** What do episodes 1 and 9 say about time?

**Expected:** The catalogue only has 4 episodes — episode 9 doesn't exist. The answer
should cover episode 1 normally (relativity's central topic *is* time — relativity of
simultaneity, time dilation) **and** explicitly state that episode 9 doesn't exist,
rather than silently dropping it or refusing episode 1 because of it.

**Coverage:** Episode 1 — ✅ covered, Episode 9 — does not exist (flagged, not "uncovered")

---

## Case 12 — general, deliberately ambiguous phrasing
**Q:** Tell me something interesting from these podcasts

**Expected:** No specific topic or episode named — this stress-tests the intent parser's
robustness (must still produce a valid classification, not crash) and broad retrieval on
a vague query. The catalogue obviously has substantive content, so this should **not**
be refused just because the query itself is vague — refusal is for "not covered by the
catalogue," not "hard to search for." Not scored on factual coverage; scored on schema
validity, no crash, and whether the answer is actually helpful given how open-ended the
question is.

**Coverage:** N/A (not a coverage test — an intent-parser robustness test)
