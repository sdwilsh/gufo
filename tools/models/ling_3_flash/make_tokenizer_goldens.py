#!/usr/bin/env python3
"""Generate tests/models/ling_3_flash/fixtures/tokenizer_goldens.json.

Token ids come from llama.cpp's llama-tokenize (reference: b11146) loading the
vocab-only GGUF built by make_vocab_gguf.py. llama-tokenize loads a
tensor-free GGUF without any extra flag.

usage: make_tokenizer_goldens.py VOCAB.gguf OUT.json [llama-tokenize]
"""
import json
import subprocess
import sys
import tempfile

CASES = [
    ("empty", ""),
    ("hello", "Hello, world!"),
    ("prose", "The quick brown fox jumps over the lazy dog. It's a fine day, isn't it?"),
    ("leading-space", " leading space"),
    ("trailing-space", "trailing space "),
    ("python", "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"),
    ("cpp", "#include <vector>\nint main() {\n\tstd::vector<int> v{1, 2, 3};\n\treturn v.size() > 0 ? 0 : 1;\n}\n"),
    ("json", '{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}'),
    ("digits", "1234567890 and 3.14159 and 1,000,000"),
    ("digits-glued", "abc123def456"),
    ("year", "In 2026 the price was $1,299.99 (-15%)."),
    ("contractions-lower", "I'm you're he's we've they'll she'd can't"),
    ("contractions-upper", "I'M YOU'RE HE'S WE'VE THEY'LL SHE'D CAN'T"),
    ("contractions-mixed", "It'S It'Ll It'rE It'Ve It'D It'M It'T"),
    ("apostrophe-other", "rock 'n' roll 'tis the '90s"),
    ("curly-apostrophe", "don’t stop"),
    ("cjk", "你好，世界！这是一个测试。"),
    ("japanese", "こんにちは、世界。カタカナも。"),
    ("korean", "안녕하세요 세계"),
    ("mixed-cjk-latin", "Ling-3.0 是一个 MoE model, 共 1T 参数。"),
    ("emoji", "Great job \U0001f600\U0001f680 done \U0001f44d\U0001f3fd!"),
    ("emoji-zwj", "family \U0001f468‍\U0001f469‍\U0001f467 flag \U0001f1fa\U0001f1f8"),
    ("combining", "é café ñ ǟ क्ष"),
    ("precomposed", "café naïve über Ångström"),
    ("arabic-hebrew", "مرحبا بالعالم שלום עולם"),
    ("cyrillic-greek", "Привет, мир! Γεια σου"),
    ("ws-spaces", "a  b   c    d"),
    ("ws-many-spaces", "x" + " " * 40 + "y"),
    ("ws-newlines", "line1\nline2\r\nline3\n\n\nline4"),
    ("ws-trailing-newlines", "text\n\n"),
    ("ws-tabs", "a\tb\t\tc\t \td"),
    ("ws-mixed", "  \n \t \n  indented\n\t\tcode  \n"),
    ("ws-only", " \n\t\r\n "),
    ("ws-unicode", "a b c　d e\u0085f​g"),
    ("punct", "!!! ... ??? --- *** ((())) [[]] {{}} <<>> ~~~"),
    ("punct-newline", "end.\n\nNext?!\r\n\r\nDone;\n"),
    ("symbols", "a+b=c; x*y/z; 50% #tag @user $var ^caret |pipe| &amp;"),
    ("url", "https://example.com/path?q=1&r=2#frag"),
    ("long-word", "Supercalifragilisticexpialidocious" * 4),
    ("repeat", "ha " * 50),
    ("control-bytes", "a\x01b\x1fc\x7fd"),
    ("special-role", "<role>HUMAN</role>hi<|role_end|>"),
    ("special-chat", "<role>SYSTEM</role>You are helpful.\ndetailed thinking on<|role_end|><role>HUMAN</role>Hello<|role_end|><role>ASSISTANT</role>\n<think>"),
    ("special-adjacent", "<|role_end|><|role_end|>x<|role_end|>"),
    ("special-think", "<think>ponder</think>answer"),
    ("special-tool", "<tool_call>f\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n</tool_call>"),
    ("special-near-miss", "<|role_end| <role>HUMAN< /role> <|rol"),
    ("special-eos-bos", "<|startoftext|>x<|endoftext|>y"),
]
NO_SPECIAL = ["special-role", "special-chat", "special-adjacent", "special-think",
              "special-tool", "special-eos-bos"]


def tokenize(tool, model, text, parse_special):
    with tempfile.NamedTemporaryFile() as f:
        f.write(text.encode("utf-8"))
        f.flush()
        cmd = [tool, "-m", model, "--ids", "--no-bos", "--log-disable", "-bf", f.name]
        if not parse_special:
            cmd.append("--no-parse-special")
        out = subprocess.run(cmd, check=True, capture_output=True).stdout.decode()
    return json.loads(out.strip().splitlines()[-1])


def main():
    model, out = sys.argv[1], sys.argv[2]
    tool = sys.argv[3] if len(sys.argv) > 3 else "llama-tokenize"
    cases = []
    for name, text in CASES:
        if text == "":
            continue  # llama-tokenize rejects an empty prompt; the test covers it.
        cases.append({"name": name, "text": text, "parse_special": True,
                      "ids": tokenize(tool, model, text, True)})
    for name, text in CASES:
        if name in NO_SPECIAL:
            cases.append({"name": name + "/literal", "text": text, "parse_special": False,
                          "ids": tokenize(tool, model, text, False)})
    json.dump({"reference": "llama.cpp b11146 llama-tokenize", "cases": cases},
              open(out, "w"), ensure_ascii=True, indent=1)
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
