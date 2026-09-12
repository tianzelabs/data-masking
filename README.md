# fpe-mask

A format-preserving masking layer to sit between an agent and sensitive
data: structure-aware (a Hungarian company's `Kft.` legal-form suffix stays
readable) and, optionally, character-class-aware (an accented Hungarian
letter stays an accented letter in the same position) encryption, exposed
as an MCP server / Claude Code plugin so raw values don't have to enter the
model's context to begin with.

```
"Felmándi Kft."  --mask(mode=strict)-->  "Pnbdómbf Kft."   --unmask-->  "Felmándi Kft."
```

## Why this shape

- **Structure preservation is a dictionary problem, not a crypto one.**
  Legal-form suffixes (`Kft.`, `Zrt.`, `Bt.`, ...) are a small closed enum,
  not PII. `fpe_mask.suffixes` recognizes them (case-insensitively, dot
  optional) and passes them through untouched, so an agent reasoning about
  "this is an LLC" still can.
- **Character-class preservation is real FF1 (NIST SP 800-38G).**
  `fpe_mask/ff1.py` is a clean-room FF1 implementation, cross-checked
  round-by-round against the official NIST sample vectors (see
  `tests/test_ff1.py` — includes the published AES-128 test vector,
  key `2B7E151628AED2A6ABF7158809CF4F3C`, radix 10,
  `0123456789` → `2433477484`).
- Two modes, same engine:
  - **`standard`** (default): case + letter/digit shape preserved; plain
    and Hungarian-accented letters share one 35-symbol alphabet per case.
    Real, unqualified FF1 security for anything but very short (<4 char)
    runs.
  - **`strict`**: also keeps accented vs. plain apart, so position 5 being
    an accented vowel stays an accented vowel. See **Limitations** below —
    this mode has an honest, unavoidable weak spot.

## Known, unavoidable limitation (read this before trusting `strict` mode)

NIST SP 800-38G requires `radix**length >= 1,000,000` for FF1 to carry its
stated security guarantee, and requires `length >= 2` structurally (FF1 is
a Feistel construction — it needs two halves). The 9 Hungarian accent
letters (`áéíóöőúüű`) are a 9-symbol alphabet, so:

- A **single** accented character anywhere in a masked value cannot be
  Feistel-split at all — there is nothing to fall back to that gives real
  security, because **the entropy ceiling is the alphabet itself**
  (log2(9!) ≈ 18.5 bits total, not per-character). This isn't an
  implementation gap; no algorithm changes that.
- `strict` mode pools all occurrences of a class *across the whole input
  string* before running FF1, which mostly rescues this (e.g. "Felmándi"
  has enough accented+plain lowercase letters combined once pooled) — but
  a class that still ends up with **exactly one** occurrence after pooling
  (e.g. a lone capital initial once the legal-form suffix is stripped out,
  as happens with "Felmándi") falls back to a **keyed substitution
  cipher** (`fpe_mask/keying.py:keyed_permutation`), not FF1.

Every `mask()` call reports this honestly instead of silently
overstating its own guarantee:

```python
result = engine.mask("Felmándi Kft.", context="company_name")
result.weak_domain_classes        # classes where FF1 ran below NIST's minimum domain
result.singleton_fallback_classes # classes that used the substitution fallback
result.is_fully_compliant         # False if either list above is non-empty
```

If you don't need per-character accent-vs-plain fidelity, use `standard`
mode — real names/companies are long enough in practice that this never
happens.

Also note: FPE (this whole class of technique) is pseudonymization, not
anonymization. It's reversible by design (that's the point — the agent's
answer needs to make sense and, when authorized, be un-redacted). It does
not defend against re-identification via quasi-identifiers, and a masking
layer alone does not satisfy purpose-based access control — pair it with
real access control at the data source (see the architecture note at the
bottom).

## Marimo / agent-shield workflow

If your setup is "an agent writes code that runs in a notebook, a human
looks at the real data on a canvas, and the agent must never get raw data
back", the right primitive isn't masking a value *for the agent to read* —
it's deciding what a cell's real execution result gets turned into before
it becomes that agent's tool response, while the unfiltered result still
goes to the notebook's normal (human-facing) rendering untouched.

`fpe_mask.shield` implements that:

```python
from fpe_mask.shield import ColumnPolicy, SensitiveValueRegistry, summarize_for_agent, redact_exception

policy = ColumnPolicy().mark_safe("status", "country")  # everything else stays sensitive by default
registry = SensitiveValueRegistry()
registry.register_dataframe(df, policy)   # once, whenever a dataframe is loaded

# after the marimo bridge runs a cell and gets back `result` (or catches `exc`):
agent_tool_result = summarize_for_agent(result, context="customers_table", policy=policy, registry=registry)
# ... `result` itself, unfiltered, still goes to marimo's normal cell-output
# rendering for the human viewing the canvas -- these are two separate paths
# out of the same execution, and only one of them reaches the model.

# on an exception instead of a result:
agent_error = redact_exception(exc, registry=registry)
```

What `summarize_for_agent` gives the agent for a DataFrame: shape, per-column
dtype/kind/null-count, and then, per column:
- **sensitive (default for every column)**: distinct-count (capped/bucketed),
  a rounded histogram for numeric columns (never raw min/max — see below),
  and a `sample_rows_masked` preview run through the same FF1 engine as the
  rest of this package (dates get a deterministic day-offset instead, so
  they stay valid calendar dates rather than digit-scrambled nonsense).
- **explicitly marked safe** (`policy.mark_safe(...)`): real min/max/mean/
  quantiles, and real category counts if cardinality is low.

Two things worth being deliberate about, both driven by the same lesson —
**an aggregate over too few records IS a raw value**:

- A numeric histogram's bin edges are, by construction, order statistics of
  the real data — the outermost bin's edges are literally the column's true
  min and max. `stats.numeric_quantile_bins` mitigates this two ways: every
  bin is forced to cover at least `min_count` (default 5) records, merging
  undersized bins into a neighbor, and edges are rounded outward to a plain
  1/2/5×10^k grid instead of exact order statistics — so an edge coincides
  with a real record's value only by coincidence, not systematically. If a
  column has fewer than `min_count` non-null values at all, the histogram is
  suppressed (`"distribution_suppressed": "..."`) rather than emitted with a
  single bin whose edges are that column's literal values. This is a
  pragmatic mitigation, not a formal privacy guarantee (see the docstring
  in `fpe_mask/shield/stats.py`) — for a hard requirement, replace the
  rounded edges with fixed business-defined bucket boundaries instead of
  ones derived from the column's own min/max.
- Tracebacks and stdout are the classic accidental-leak vector for a code-
  executing agent (`ValueError: could not convert 'Jane Doe' to float`).
  `SensitiveValueRegistry` + `redact_exception`/`redact_text` do exact-match
  scrubbing of every registered sensitive value plus a small set of generic
  PII regexes (email/phone/card/IBAN-shaped strings) as a second net — call
  `registry.register_dataframe(df, policy)` (full columns, not just the
  sample) whenever real data is loaded, and route every exception through
  `redact_exception` before it reaches the agent. Never forward `str(exc)`
  or a raw traceback directly.

pandas and polars are both supported via one adapter
(`fpe_mask/shield/dataframe_adapter.py`); install whichever you use with
`pip install -e ".[pandas]"` / `.[polars]` / `.[dev]` (both, for running the
test suite).

## Layout

```
fpe_mask/          core library (ff1.py, alphabets.py, suffixes.py,
                    classifier.py, engine.py, keying.py, key_store.py)
fpe_mask/shield/    what to hand a code-executing agent instead of raw data:
                    policy.py, dataframe_adapter.py, stats.py, registry.py,
                    redact.py, summarize.py -- see "Marimo / agent-shield" above
mcp_server/         MCP server exposing mask_text / unmask_text / classify_preview
.claude-plugin/     Claude Code plugin manifest (best-effort — see below)
.mcp.json           direct MCP registration fallback
hooks/              optional PostToolUse hook sketch for automatic masking
tests/              pytest suite, including the NIST FF1 vector
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m pytest tests/ -v
```

## Using it

### As a library

```python
from fpe_mask import MaskingEngine, generate_master_key

key = generate_master_key()          # store this; losing it = losing unmask
engine = MaskingEngine(key, mode="strict")

r = engine.mask("Felmándi Kft.", context="company_name")
print(r.text)                        # e.g. "Pnbdómbf Kft."
print(engine.unmask(r.text, context="company_name").text)  # "Felmándi Kft."
```

`context` is a public (non-secret) domain-separation tag — pass the same
string to `mask()` and `unmask()` for a given field (e.g. the column
name), otherwise unmasking will "succeed" but produce garbage (FF1 doesn't
error on a wrong tweak, it just decrypts to nonsense).

### As an MCP server (the actual "protection layer around the agent")

```bash
claude mcp add fpe-mask -- /path/to/data-masking/.venv/bin/python3 -m mcp_server
```

or install `.claude-plugin/plugin.json` as a plugin (adjust
`${CLAUDE_PLUGIN_ROOT}` substitution / the `cwd` key if your Claude Code
version's plugin manifest schema differs — this was written against the
current plugin conventions but wasn't round-tripped through an actual
plugin install in this session).

The intended flow: whatever tool fetches sensitive rows/text calls
`mask_text` on each sensitive field before that text is placed in the
model's context; if the model needs to write a value back out to a real
system, the agent (or a `PostToolUse`/`PreToolUse` hook — see
`hooks/mask_tool_io.py` for a sketch) calls `unmask_text` first.

Master key: auto-generated on first run at `~/.data-masking/master.key`
(mode 600), overridable via `FPE_MASK_KEYFILE`. For anything beyond local
use, replace `fpe_mask/key_store.py`'s `load_or_create_master_key()` with a
real KMS call (AWS KMS / GCP KMS / Vault) — nothing else needs to change.

## Where this sits in a full defense-in-depth stack

This plugin is layer 3 of 5 in a typical setup (in order of how much of the
problem each one actually solves):

1. **Don't send the data at all** — code execution / handle-passing so the
   agent operates on data via a sandboxed tool and only aggregates come
   back to the model.
2. **Enforce access at the data source** — row/column-level security keyed
   to the *end user's* identity (Snowflake/BigQuery/Unity Catalog masking
   policies, Immuta/Privacera), not a shared service account.
3. **This: format-preserving masking of what does reach the model.**
4. **Trust boundary of the inference call itself** — VPC/zero-data-retention
   inference, confidential computing.
5. **Gateway-level DLP + audit logging** on top of all of the above.

Layer 3 alone is not sufficient defense — it's meant to sit inside 1+2, as
a mitigation for whatever residual sensitive text does end up needing to
pass through the model (e.g. because the agent's task genuinely requires
reasoning over the actual name/value, not just an aggregate).
