# Evaluation datasets

Versioned, hand-labelled datasets that every quality change is measured
against. Each file is validated by Pydantic schemas (`evals/core/schemas.py`)
and composition rules (`evals/core/composition.py`) in the unit tests. **No
real personal data anywhere**: identifiers are synthetic or reserved-format
(`000.000.000-00`, `@example.com`, `(11) 90000-0000`).

| File | Dataset | Used by | Size |
|---|---|---|---|
| `retrieval.yaml` | **retrieval**: retrieval and grounding questions | retrieval (offline) and grounding (live) suites | 73 answerable + 25 unanswerable |
| `router.yaml` | **router**: messages labelled with an intent | router (live) suite | 69 |
| `slots.yaml` | **slots**: loan requests labelled with expected slots | slot-extraction (live) suite | 36 |
| `compliance.yaml` | **compliance**: draft replies (promise / hedge / neutral) and PII masking cases | keyword and strict screens, masking (offline), LLM check (live) | 41 + 19 |

`review.yaml` (written only after the maintainer approves a dataset) records
the approved content hash of each file; a baseline is refused for any dataset
whose hash differs.

## Changing a dataset

1. Edit the YAML, bump `version`, run `uv run pytest tests/evals`.
2. Print a review sample: `uv run python -m evals review-sample --dataset D --n 15 --seed 42`
   (add `--evidence` for retrieval items: it shows the top-3 chunks the index
   returns, computed offline with the local embedding model — no gateway).
3. After approval the hash in `review.yaml` is updated; baselines are re-recorded.

## Retrieval dataset (`R-###`)

Fields: `question`, `kind` (`answerable` | `unanswerable`), `difficulty`.

**Answerable** items also carry `document` (`cdc-consolidada`, `lgpd`,
`open-finance-regulamento`, `cet-disclosure`, `product-catalog`),
`expected_refs`, `evidence`, `style` and `accents`.

- **Paraphrase, don't copy.** A question is how a customer would ask, never a
  sentence lifted from the article. A test rejects a question that appears
  verbatim in the corpus.
- **`style`**: `lexical` = uses the article's own vocabulary; `colloquial` =
  everyday wording, slang or contractions (`pra`, `vcs`), avoiding the article
  heading's words. At least 40% of answerable items are colloquial.
- **`accents: false`** marks a question **deliberately written without accents**
  (`emprestimo`, `credito`) to test accent folding. Leave it `true` otherwise,
  even if a short question happens to contain no accented letter.
- **`expected_refs`**: every article that legitimately answers the question
  (a hit on any counts). A plain string is an article of the item's own
  `document`; a `{document, ref}` pair is an article of **another document that
  also answers** (e.g. the CET is defined by both the CMN resolution and CDC
  art. 54-B; consent revocation appears in both LGPD art. 8º §5 and the Open
  Finance resolution art. 15). At least one ref must be in the item's own
  document; refs cannot point into the product catalog; every ref must exist in
  the corpus fixture (a test checks). Absent only for the product catalog.
  Add a cross-document ref only when the other article **directly answers the
  core of the question**, not when it merely shares vocabulary — false friends
  (LGPD art. 19 "confirmação de tratamento" vs Open Finance art. 20
  "confirmação de compartilhamento") stay out.
- **`evidence`**: a short (≤200 characters) literal quote from the expected
  article; a test verifies it against the committed corpus fixture. Quote the
  sentence that answers the question, not the article heading.
- **`difficulty`** is *derived*, and a test recomputes it: `easy` = lexical and
  one ref; `medium` = colloquial **xor** multi-ref; `hard` = colloquial and
  (multi-ref or unaccented). Unanswerable: `far` = easy, `near_miss` = hard.

**Unanswerable** items carry a `distance`:

- `far`: obviously outside the corpus's subject (tax, traffic, labour law).
- `near_miss`: sounds in-domain but the corpus does **not** answer it (a rate
  cap, a Pix refund deadline, the catalog's interest rate). A near-miss must
  be checked against the corpus (`review-sample --evidence`): if a retrieved
  chunk actually answers it, the item is wrong. At least 15 near-misses.

## Router dataset (`T-###`)

`message`, `active_flow` (`none` | `consent_confirmation` | `slot_filling`),
`category`, `expected`, `acceptable`.

- **`expected`** is one of the six intents (`product_question`,
  `regulatory_question`, `loan_simulation`, `profile_analysis`, `complaint`,
  `out_of_scope`) or `continue`.
- **`clear`**: one defensible label. A clear item with an active flow is a *new
  request that breaks out of the flow* and expects its own intent.
- **`ambiguous`**: more than one label is defensible; `expected` is the best
  label and `acceptable` lists the others. Only ambiguous items have `acceptable`.
- **A new request during an active flow** (a `clear` item with `active_flow`
  set) is a message that is unmistakably *not* an answer to what was asked
  ("Esquece a simulação, quais produtos vocês têm?"); it expects its own
  intent, not `continue`. There are at least four.
- **`continuation`**: a short reply to what the assistant just asked
  (`sim, autorizo`, `24 meses`); it needs an active flow, expects `continue`,
  and **must not be reclassified**.

## Slots dataset (`S-###`)

`message`, `known` (slots filled before the message), `expected` (the slots
after the node handles it), `tags`.

- A field absent from `expected` **must stay unset**: extraction is scored on
  not inventing values, and an *invalid* value (negative or zero amount, zero
  or negative term, an unrecognized amortization type) counts as absent.
- Amounts are strings of the plain number (`"5000"`); amortization is `PRICE`
  or `SAC`; "parcelas fixas" means `PRICE`, "parcelas decrescentes" means `SAC`.
- Colloquial numbers are part of the set: "dez mil", "2,5 mil", "R$ 10k",
  "um ano", "um ano e meio", "dois anos", "três anos".
- Tags are derived: `complete` (all three slots), `partial`, `missing` (none),
  plus `invalid` when the message has an invalid value.

## Compliance dataset (`C-###` approval items, `P-###` PII items)

**Approval items** (`text`, `label`, `tags`) are draft replies:

- `promise`: promises or guarantees credit approval (or, adversarially, says
  the credit is granted without using the word "approval").
- `hedge`: mentions approval or the decision but does not promise it ("sujeito
  à análise", "não posso garantir", "sem garantia de resultado").
- `neutral`: unrelated to approval (a simulation, a greeting) — including text
  that uses "aprovar" in another sense.
- Tags: `adversarial` (phrasing meant to slip past keyword screens, or to trip
  them), `fail_closed` (the examples that motivated the strict screen — at
  least two promises and two hedges), `mentions_approval`.

Some items are **known false positives/negatives of a screen on purpose**
(e.g. a hedge that the keyword screen flags because it contains both
"garantia" and "aprovação"): the evaluation reports each screen's precision and
recall separately, and these items are what make the numbers honest.

**PII items** (`text`, `expected_masked`, `known_gap`) check `mask_pii` end to
end; the masked placeholder is `[DADO PROTEGIDO]`. `expected_masked` is always the
**correct** output. Negatives (amounts, percentages) must come out unchanged. The
dataset measures the masker, not only what it already handles: a case the masker
gets wrong is kept with `known_gap: true`, so the baseline records it as a
failure instead of being filtered out. A test requires every `known_gap` item
to still fail — when the masker is fixed the test tells you to drop the flag
and re-record the baseline. No item currently carries the flag (see below).

### Masking gaps fixed in `fix/pii-masking-and-slot-validation`

Dataset v3 dropped `known_gap: true` from P-016–P-019 once `mask_pii` started
handling all four correctly; `masking.exact_match` moved from 15/19 to 19/19
in the offline compliance baseline.

| Input | Was | Now |
|---|---|---|
| `CPF 00000000000 informado no cadastro.` | `CPF [DADO PROTEGIDO]0 informado no cadastro.` (a digit leaked: the unbounded phone pattern matched first) | `CPF [DADO PROTEGIDO] informado no cadastro.` — a dedicated, boundary-anchored 11-digit pattern runs first |
| `Cartão 0000000000000000 cadastrado.` | `Cartão [DADO PROTEGIDO][DADO PROTEGIDO] cadastrado.` (masked in two pieces) | `Cartão [DADO PROTEGIDO] cadastrado.` — a single 13–19 digit, Luhn-checked card pattern claims the whole number before phone/CPF get a chance |
| `Depósito na conta 12345-6.` | unchanged (a one-digit check digit was not matched) | `Depósito na conta [DADO PROTEGIDO].` — masked only when an account/agency context word ("conta", "c/c", "agência", "ag") precedes it, so a CEP or a plain hyphenated number is never masked collaterally |
| `Cartão 0000 0000 0000 0000 cadastrado.` | unchanged (space-separated groups were not matched) | `Cartão [DADO PROTEGIDO] cadastrado.` — the card pattern accepts digits grouped by single spaces or hyphens |

The Luhn check is what keeps the card pattern from masking an arbitrary long
number (an id, an unformatted amount, ...): only a candidate whose digits pass
the checksum is treated as a card.
