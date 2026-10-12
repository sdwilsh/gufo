# Ling-3.0-Flash oracle inputs

- `oracle_text.txt`, `oracle_prompts.json`: the first oracle. It is public-domain prose that the model has memorised, with a perplexity of about 1.5. It suffices for the correctness gate, not for quality claims.
- `oracle_novel_text.txt`, `oracle_novel_prompts.json`: a second, quality-sensitive oracle on novel text (12,745 tokens, 47,543 bytes). Claude wrote it from scratch for this purpose, with no source text and no paraphrase of existing works. It mixes registers: narrative fiction with dialogue, a technical explanation of tide-gauge and levelling practice, an email and meeting notes, tables and numbers, original C++ and Python with comments and a unit test, a log excerpt, and Chinese fiction, minutes, and an essay. Its only purpose is to be text the model has not seen. The story, people, and places are invented. The three greedy prompts (story, code, zh) are prefixes cut from the text.

To regenerate the oracle, run `ORACLE_TEXT=... ORACLE_PROMPTS=... ORACLE_OUT=... run_oracle.sh`. Outputs go to the git-ignored `artifacts/ling-3-flash/oracle-novel/`.
