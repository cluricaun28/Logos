# What We Learned Building Logos

*Companion to [README.md](../README.md). The README describes the system; this file is
the lessons — the things that should be useful to anyone building an agent on local
hardware, whether or not they adopt Logos.*

## 1. The Alternative Path Is Real

The mainstream path to capable AI is megamodels in football-field-sized data centers,
delivered behind an API. Logos is evidence for the alternative: a 27B-parameter model
on a single 32GB consumer GPU, running **locally, personally, and privately**. The
model is a brain; the harness is what steers it. Most of what makes the system capable
— memory, curated knowledge, steering, tools — lives in the harness, not the weights,
and the harness ships with no model attached.

## 2. What Weights Actually Are — Three Parts

Weights are information, and most of it is information that already exists: the
internet. We decompose a model's weights into three parts.

1. **Redundant** — memorized corpus content. You are paying parameters, hardware, and
   power to carry it. Retrieval makes it redundant: don't carry the bookshelf, build a
   good librarian.
2. **Irreducible** — the reasoner/searcher itself. You cannot use the internet to find
   the internet. This part has a floor: a model *does* need weights, and some of them
   can only be trained, not retrieved.
3. **Counterproductive** — the alignment and safety reflex. For a truth-seeking task
   it is actively harmful. Our base model scores 4–5/5 on TruthfulQA, MMLU, and GSM8K
   while hedging on 44% of a 25-prompt moral/worldview battery — it knows the truth
   and hedges anyway.

The harness's job, precisely: stop paying for part 1 (retrieve it instead), steer past
part 3, keep part 2. This is our analysis of the architecture, not a lab's — but every
number in it is measured on our own deployment.

## 3. Archive, Don't Compress

In a compression architecture, retention compounds as r^N: r is the per-round
summarizer fidelity, N is the number of compaction rounds a thread passes through.
The measured scale is not hypothetical: 23 days of use = 4.1B prompt tokens =
4,000+ fills of a 1M-scale window across the deployment, and the largest single
conversation alone went through 13+ rounds in 20 hours. After each round the
original bytes are unrecoverable at any r; and even at a generous r = 0.99 — the
most favorable assumption about a summarizer squashing 800K tokens into a few
hundred — 13 rounds leave ~88%, and the decay is geometric in N. So: compaction
applies to the *view*, never the *store*. Every turn is archived verbatim to a
local SQLite database before it leaves the window, and recall returns the original
bytes. The window is a lens; the record is the asset.

The apples-to-apples question is time, not tokens. A frontier agent's operating
pattern is fill-the-1M-window, compress, repeat: at this conversation's measured
pace (~658K tokens/hour), a 1M window fills in ~90 minutes, so by hour 20 the
conversation is 13+ lossy compressions deep, running on summaries of summaries —
the point where the standard operational answer is to start a new session and hand
off in a written summary. Logos has no such cycle: its largest conversation ran
20.1 hours straight (870 turns, 13.2M tokens processed — 13× a 1M window), its
record at hour 20 is byte-identical to hour 1, and every one of those turns was
retrieved verbatim from conversations running days later. And no window, however
large, holds the whole record: the store now carries ~19M tokens — 19× a 1M
window — byte for byte.

## 4. The Window Is a View; the Store Is the Record

This is the relationship that makes everything else work. The rolling context window
is a bounded *projection* of an unbounded verbatim store — 55,000+ turns and counting,
across 5,000+ sessions, every one retrievable byte-for-byte with full-text and semantic
search in single-digit milliseconds. A model with a 262K window plus infinite crisp
re-reads beats a model with a 1M *compressed* window and no re-read path: the number
that matters is fidelity at the moment of recall, not tokens held at a distance.
Inside one conversation a 1M window holds more at any moment — that is not contested.
What it cannot do is hold a year, or survive the conversation ending.

## 5. Steer the Model to the User — and Measure It

Nearly every frontier model tested lands bottom-left (secular-progressive) on the
[politicalcompass.org](https://politicalcompass.org) 62-proposition instrument; the
one exception is a deliberate positioning choice by its maker. If that default is your
worldview, fine — if not, a model you can't redirect is a tool that argues with you.

The displacement is measurable. Base model + user-defined worldview profile:
**>13 units** on the instrument — enough to move our deployment into the only
top-right quadrant on the project's chart. Base model + prompt tweaks alone:
**<1 unit**. Method: the project's own 62 propositions; scoring weights regressed from
the dataset's 1,371 recorded answer-sets; scorer validated against the project's
control sets to ±0.001. Caveats: a single pass (±~1 unit), and the robust result is
the *magnitude* — the direction is whatever the user's profile is. Nothing is baked
in; the profile is the user's to write, edit, and replace.

Steering also works better against uncensored / non-safety-trained base models, which
spend less of their weight fighting the redirect. (Uncensored ≠ unbiased — it removes
the censor, not the corpus default.)

## 6. Local-Only Is a Feature — and Not Just About Data

Running the model locally means your data never leaves the box. It means something
else that is easier to miss: **you control the relationship to the model.** A
cloud-API model comes with the vendor controlling your system prompt and the safety
training, in various degrees — you get what they allow you to have, and the terms can
change under you. A local model on your own harness is yours to direct.

AI is a tool. Whether it is used for good or evil is, as with all tools, determined by
its user — which is exactly why the tool should live where the user lives.

## 7. Curation Beats Corpus

The public pattern of a plain-text knowledge base an agent reads and writes directly
(à la Karpathy's "LLM wiki") is now well known. What most implementations skip is the
*discipline on top of the container*: every entry verified against primary sources,
every source carrying a dossier (what it's truthful on, what it consistently omits),
every claim staged for review before it commits, and a nightly loop that researches,
strips framing, and distills new findings into the library. A corpus is a pile of
claims; a curated library is a pile of *checked* claims — and the difference compounds
exactly like the r^N argument compounds the other way.

## 8. One User → ~A Dozen — One GPU per Instance, Not a Shared Card

Logos started with a single owner and has scaled to roughly a dozen users, each with
their own private memory, reference library, and worldview profile. The scaling is not
a dozen users crammed into one 32GB card: every model instance runs on its own GPU,
and the deployment grows by adding instances. The ~a-dozen-user deployment is a single
8-GPU server (96GB per GPU); the 32GB single-GPU box is the single-user configuration,
not the multi-user one. Per-user isolation is a property of the harness, not of the
model — another piece of evidence that the interesting state lives outside the
weights. Dated snapshot, **as of 2026-09-16 (and growing)**: 55,000+ conversation
turns verbatim; 40,000+ curated knowledge pages. Code lineage since July 2025; this deployment in active use since April 2026.

## 9. Post-Training, Honestly

Our efforts to post-train have been less impactful than our harness-driven steering.
Therefore we still use a model readily available on Hugging Face. That is a
deliberate verdict, not an omission: the steering works *without* retraining, the
base model is replaceable (it is a brain, swappable on demand), and retraining —
including DPO/ORPO pipelines we have run and validated on real hardware — is the part
still worth proving out. The honest ordering is: harness first, weights second.

## 10. Open / Next

- **No-persona baseline A/B:** the same instrument, model, and conditions with the
  worldview profile stripped — a controlled measurement of exactly what the steering
  adds, run against our own baseline rather than the reference model's.
- **Weight self-modification:** making the external-state loop (memory, library,
  skills) into actual self-fine-tuning — the direction the recursive-self-improvement
  work on *harnesses* points. This is the genuinely novel leg, and it needs
  architecture control and compute we don't have yet. Until then, the external store
  is a working proxy: the system gets better over time, bounded by the reasoner's
  capacity.
