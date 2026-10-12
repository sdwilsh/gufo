#!/usr/bin/env bash
# Builds the llama.cpp CPU oracle for Ling-3.0-Flash (PLAN.md section 4).
#
# Uses the local Homebrew llama.cpp b11146 (commit 7fe450e19), CPU only, f16 KV
# cache, greedy sampling, raw prompts (no chat template; add_bos is false).
# Produces, in $ORACLE_OUT (default <repo>/artifacts/ling-3-flash/oracle):
#   kl_base.bin   llama-perplexity --kl-divergence-base, -c 512 --chunks $CHUNKS
#   greedy.jsonl  3 prompts x 64 greedy tokens (ids, text, top-5 logprobs)
#   COMMANDS.txt  llama.cpp version string and the exact command lines
#   *.log         raw tool output
#
# Environment:
#   MODEL          shard 1 of the GGUF
#   ORACLE_OUT     output directory
#   THREADS        default 12, the most advised
#   PYTHON         python3 with numpy, only for the optional KL-base self check
#   ORACLE_TEXT    default oracle_text.txt (the novel-text oracle uses
#                  oracle_novel_text.txt)
#   ORACLE_PROMPTS default oracle_prompts.json (the novel-text oracle uses
#                  oracle_novel_prompts.json)
#   CHUNKS         default 16
#   PORT           default 8099
#   STEPS          default "kl greedy manifest". "self" adds the llama.cpp
#                  against itself KL check. "manifest" writes MANIFEST.json and
#                  copies the inputs.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
MODEL=${MODEL:?set MODEL to the first shard of the Ling-3.0-flash GGUF}
OUT=${ORACLE_OUT:-$repo/artifacts/ling-3-flash/oracle}
THREADS=${THREADS:-12}
PORT=${PORT:-8099}
PYTHON=${PYTHON:-python3}
STEPS=${STEPS:-"kl greedy manifest"}
CHUNKS=${CHUNKS:-16}
TEXT=${ORACLE_TEXT:-$here/oracle_text.txt}
PROMPTS=${ORACLE_PROMPTS:-$here/oracle_prompts.json}
mkdir -p "$OUT"

version=$(llama-cli --version 2>&1 | grep -i "version" | head -1)
case $version in
  *7fe450e19*) ;;
  *) echo "llama.cpp must be b11146 (7fe450e19), got: $version" >&2; exit 1 ;;
esac

# CPU only: no layers offloaded and no devices.
common=(-m "$MODEL" -ngl 0 -dev none -t "$THREADS" -ctk f16 -ctv f16)
run() { nice -n 10 "$@"; }  # foreground only; background jobs call nice directly

cmds=$OUT/COMMANDS.txt
{
  echo "# $(date -u +%FT%TZ) on $(hostname), steps: $STEPS"
  echo "llama.cpp: $version"
  echo "model: $MODEL"
} >> "$cmds"

if [[ " $STEPS " == *" kl "* ]]; then
  # -b/-ub 512 keeps one sequence per batch (n_seq = 1).
  cmd=(llama-perplexity "${common[@]}" -f "$TEXT" -c 512 -b 512 -ub 512 --chunks "$CHUNKS"
       --kl-divergence-base "$OUT/kl_base.bin")
  echo "kl_base: ${cmd[*]}" >> "$cmds"
  s=$(date +%s)
  run "${cmd[@]}" > "$OUT/kl_base.log" 2>&1
  echo "kl_base wall: $(( $(date +%s) - s )) s" | tee -a "$cmds"
fi

if [[ " $STEPS " == *" greedy "* ]]; then
  srv=(llama-server "${common[@]}" -c 4096 -np 1 --no-warmup --no-webui
       --host 127.0.0.1 --port "$PORT" -cram 0)
  echo "greedy server: ${srv[*]}" >> "$cmds"
  echo "greedy client: $here/oracle_greedy.py --url http://127.0.0.1:$PORT --prompts $PROMPTS --n-predict 64 --out $OUT/greedy.jsonl" >> "$cmds"
  s=$(date +%s)
  nice -n 10 "${srv[@]}" > "$OUT/greedy_server.log" 2>&1 &
  srv_pid=$!
  trap 'kill $srv_pid 2>/dev/null || true' EXIT
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    kill -0 $srv_pid 2>/dev/null || { echo "llama-server died" >&2; exit 1; }
    sleep 5
  done
  "$PYTHON" "$here/oracle_greedy.py" --url "http://127.0.0.1:$PORT" \
    --prompts "$PROMPTS" --n-predict 64 --out "$OUT/greedy.jsonl"
  kill $srv_pid; wait $srv_pid 2>/dev/null || true
  trap - EXIT
  echo "greedy wall: $(( $(date +%s) - s )) s" | tee -a "$cmds"
fi

if [[ " $STEPS " == *" self "* ]]; then
  # Run llama.cpp against its own base file. The KL must be about 0, and
  # kl_base.py must decode the same file (the caller checks it with
  # kl_base.py info and tokens).
  cmd=(llama-perplexity "${common[@]}" -f "$TEXT" -c 512 -b 512 -ub 512 --chunks "$CHUNKS"
       --kl-divergence --kl-divergence-base "$OUT/kl_base.bin")
  echo "kl self-check: ${cmd[*]}" >> "$cmds"
  run "${cmd[@]}" > "$OUT/kl_self.log" 2>&1
  grep -E "Mean +KLD|Same top" "$OUT/kl_self.log" | tee -a "$cmds"
fi
if [[ " $STEPS " == *" manifest "* ]]; then
  # Input copies, token ids, sha256 of every file (needs numpy in $PYTHON).
  (cd "$here" && "$PYTHON" make_manifest.py "$OUT")
fi
echo "done: $OUT"
