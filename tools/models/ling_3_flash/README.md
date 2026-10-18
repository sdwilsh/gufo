# Ling-3.0-flash oracle inputs

- `oracle_text.txt`, `oracle_prompts.json`: the first oracle (public-domain prose
  the model has memorised, PPL about 1.5). Adequate for the correctness gate, not
  for quality claims.
- `oracle_novel_text.txt`, `oracle_novel_prompts.json`: a second, quality-sensitive
  oracle on novel text (12,745 tokens, 47,543 bytes). It was written from scratch
  for this purpose by Claude (no source text, no paraphrase of existing works) in
  mixed registers: narrative fiction with dialogue, a technical explanation of
  tide-gauge and levelling practice, an email and meeting notes, tables and
  numbers, original C++ and Python with comments and a unit test, a log excerpt,
  and Chinese fiction, minutes and an essay. It is only meant to be unseen by the
  model; the story, people and places are invented. Regenerate the oracle with
  `ORACLE_TEXT=... ORACLE_PROMPTS=... ORACLE_OUT=... run_oracle.sh`; outputs go
  to the git-ignored `artifacts/ling-3-flash/oracle-novel/`.
  The three greedy prompts (story, code, zh) are prefixes cut from the text.
