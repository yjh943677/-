#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal expression key-value mapper
- key 형식: "<수식어구 or null>-timestamp-<relation or null>"
- 출력: input_out.jsonl 한 개
- 동작: 같은 key가 여러 번 나오면 '처음 1개'만 남기고 나머지는 스킵

사용법:
  python temporal_mapper.py input.jsonl
  # 옵션:
  #   --keep-nulls        : temporal expression 없는 줄도 출력(value='null')
  #   --progress-every N  : N줄마다 진행상황 출력
"""

import argparse, json, sys, os, pathlib

# GPU 0만 노출 (요청 사항 반영; 실제 연산은 CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _norm_or_null(x):
    if x is None:
        return "null"
    s = str(x).strip()
    if s == "" or s.lower() == "null":
        return "null"
    return " ".join(s.split())

def make_key(modifier, relation):
    # 가운데는 리터럴 'timestamp' 고정
    return f"{_norm_or_null(modifier)}-timestamp-{_norm_or_null(relation)}"

def open_utf8(path, mode):
    return open(path, mode, encoding="utf-8-sig", newline="")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="입력 JSONL 파일 경로")
    ap.add_argument("--keep-nulls", action="store_true",
                    help="temporal expression 없는 줄도 출력(value='null')")
    ap.add_argument("--progress-every", type=int, default=50000,
                    help="N줄마다 진행상황 표시")
    args = ap.parse_args()

    in_path = pathlib.Path(args.inp)
    out_jsonl = in_path.stem + "_out.jsonl"

    total = emitted = skipped = dup_skipped = 0
    seen_keys = set()

    with open_utf8(in_path, "r") as fin, open_utf8(out_jsonl, "w") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue

            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[warn] invalid JSON line {total}: {e}", file=sys.stderr)
                skipped += 1
                continue

            te = rec.get("temporal expression")
            has_te = te not in (None, "", "null")
            if not has_te and not args.keep_nulls:
                skipped += 1
                continue

            key = make_key(rec.get("수식어구"), rec.get("relation"))

            # === 키 중복 제거: 이미 본 key면 스킵 ===
            if key in seen_keys:
                dup_skipped += 1
                continue
            seen_keys.add(key)

            value = te if has_te else "null"
            sentence = rec.get("sentence")

            fout.write(json.dumps(
                {"key": key, "value": value, "sentence": sentence},
                ensure_ascii=False
            ) + "\n")
            emitted += 1

            if args.progress_every and total % args.progress_every == 0:
                print(f"[info] processed {total:,} "
                      f"(emitted {emitted:,}, skipped {skipped:,}, dup {dup_skipped:,})",
                      file=sys.stderr)

    print(f"[done] Input={total}, emitted={emitted}, skipped(no-te/blank)={skipped}, dup-skipped={dup_skipped}",
          file=sys.stderr)
    print(f"[out] {out_jsonl}")

if __name__ == "__main__":
    main()
