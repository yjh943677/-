import os
import sys
import json
import logging
import re
import random
import signal
import atexit
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers import logging as transformers_logging

# ===== Transformers 경고 억제 =====
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
transformers_logging.set_verbosity_error()

# ===== STDOUT 라인 버퍼링 (tmux/nohup에서도 즉시 출력) =====
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ===== GPU/양자화 설정 =====
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # GPU 총 4개 사용
torch.backends.cudnn.benchmark = False

def _choose_compute_dtype() -> torch.dtype:
    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return torch.bfloat16 if major >= 8 else torch.float16
    except Exception:
        pass
    return torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=_choose_compute_dtype(),
    bnb_4bit_quant_type="nf4",
)

# ===== 경로/설정 =====
BASE_DIR = Path(__file__).resolve().parents[1]   # llm/
DATA_DIR    = BASE_DIR / "data"
INPUT_FILE  = DATA_DIR / "kowiki_sents.json"
OUTPUT_FILE = Path(__file__).parent / "extraction_results.jsonl"  # 요청: 파일명 고정
LOG_FILE    = Path(__file__).parent / "run.log"   # 백그라운드 로그 파일
PID_FILE    = Path(__file__).parent / "extractor.pid"
MODEL_ID    = "Qwen/Qwen3-32B"   # Instruct가 있으면 아래에서 자동으로 채팅 템플릿 사용
LOG_LEVEL   = logging.INFO
SAMPLE_SIZE = 100000
DEBUG_DECODED = False
DIAG = False

# >>> 여러 개 모두 추출 플래그 / 상한
EXTRACT_ALL = True
MAX_ITEMS = 20

# ===== 프롬프트 =====
SYSTEM_BRIEF = (
    "당신은 문장에서 시간 제약을 JSON으로 추출하는 전문가입니다. "
    "반드시 JSON 한 줄만 출력하세요."
)

FEW_SHOT_BODY = r'''
정의
- time expression: 연/월/일/시/분/요일/절기/명절/이벤트 등 구체적 시간 표지(예: "2024년", "1924년 10월 1일", "1953년 3월", "1970년대", "2015년 현재").
- temporal relation: time expression에 붙거나 함께 쓰여 시점을 규정하는 표지(예: "에", "부터", "까지", "이후에/이후", "이전에/이전", "동안", "무렵", "경", "사이에", "에서", "의").
- 범위(기간): "2010년부터 2015년까지", "2004년 9월 28일에서 12월 21일사이에", "1930년~1932년" 등은 하나의 제약으로 본다.

출력 규칙
1) 문장에서 가장 관련 있는 시간 제약 1개만 추출.
2) JSON 한 줄만 출력. 그 외 텍스트/설명/코드블록/따옴표 금지.
3) 키는 3개:
   - "time_expression": 관계 표지 제외한 시간 표현. 범위는 "시작…끝".
   - "temporal_relation": 관계 표지. 범위면 "부터…까지" 또는 "에서…사이에" 등으로 "…" 사용. 없으면 null.
   - "constraint": 최종 결합 문자열 = time_expression + temporal_relation. 관계 없으면 time_expression과 동일.
4) 시간 제약이 전혀 없으면 모두 null.
5) JSON 끝에 "<END>"를 붙여 종료.

예시
문장: "2024년에 한국에서 가장 많이 팔린 전기차는 무엇입니까?"
{"time_expression":"2024년","temporal_relation":"에","constraint":"2024년에"}<END>
문장: "제임스 얼 카터 주니어는 1924년 10월 1일에 태어났습니다."
{"time_expression":"1924년 10월 1일","temporal_relation":"에","constraint":"1924년 10월 1일에"}<END>
문장: "어떤 사람은 2010년부터 2015년까지 일했습니다."
{"time_expression":"2010년…2015년","temporal_relation":"부터…까지","constraint":"2010년부터 2015년까지"}<END>
문장: "2004년 9월 28일에서 12월 21일사이에 사건이 있었다."
{"time_expression":"2004년 9월 28일…12월 21일","temporal_relation":"에서…사이에","constraint":"2004년 9월 28일에서 12월 21일사이에"}<END>
문장: "1953년 3월, 카터는 6개월 과정을 시작했다."
{"time_expression":"1953년 3월","temporal_relation":null,"constraint":"1953년 3월"}<END>
문장: "카터는 100세까지 산 최초의 대통령입니다."
{"time_expression":null,"temporal_relation":null,"constraint":null}<END>
문장: "2015년 현재의 교명인 국제예술대학교로 교명을 변경하고 현재에 이르고 있다."
{"time_expression":"2015년 현재","temporal_relation":"의","constraint":"2015년 현재의"}<END>
문장: "1970년대 그와 함께한 수많은 백업 뮤지션 중에서는 막 음악을 시작한 브루스 스프링스틴과 스티브 밀러가 있었다."
{"time_expression":"1970년대","temporal_relation":null,"constraint":"1970년대"}<END>
'''

def build_prompt(sentence: str, tokenizer) -> str:
    body = FEW_SHOT_BODY + f'\n아래 문장에 대해 위 JSON 형식 한 줄만 출력하세요.\n문장: "{sentence}"\n'
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_BRIEF},
            {"role": "user", "content": body},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return SYSTEM_BRIEF + "\n" + body

# ===== 로거 (콘솔 + 파일 동시 출력) =====
def _setup_logging():
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=LOG_LEVEL, format=fmt, datefmt=datefmt, handlers=handlers, force=True)

_setup_logging()
logger = logging.getLogger(__name__)

# ===== (추가) null 레코드 스킵 판별 =====
def _is_null_record(rec: dict) -> bool:
    """시간 정보가 전부 null/빈 값이면 True를 반환."""
    if not isinstance(rec, dict):
        return False
    return (
        rec.get("temporal expression") in (None, "", []) and
        rec.get("timestamp") in (None, "", []) and
        rec.get("relation") in (None, "", [])
    )

# ===== JSON 파싱 유틸 =====
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else text

def _sanitize_jsonish(text: str) -> str:
    text = _strip_code_fences(text)
    text = text.split("<END>")[0].strip()
    text = (text
            .replace("“", '"').replace("”", '"')
            .replace("„", '"').replace("‟", '"')
            .replace("’", "'").replace("‘", "'"))
    text = re.sub(r'""\s*(time_expression|temporal_relation|constraint)\s*""', r'"\1"', text)
    text = re.sub(r"'(time_expression|temporal_relation|constraint)'\s*:", r'"\1":', text)
    text = re.sub(r":\s*None\b", r": null", text)
    return text

def _balanced_json_object(text: str):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
    if end is None:
        return None
    frag = text[start:end+1]
    try:
        return json.loads(frag)
    except Exception:
        return None

def _normalize_keys(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        nk = k.strip().strip('"').strip("'")
        out[nk] = v
    return out

def _parse_model_json(decoded: str):
    txt = _sanitize_jsonish(decoded)
    data = _balanced_json_object(txt)
    if isinstance(data, dict):
        return _normalize_keys(data)
    try:
        maybe = json.loads(txt)
        if isinstance(maybe, str):
            inner = _balanced_json_object(maybe)
            if isinstance(inner, dict):
                return _normalize_keys(inner)
        if isinstance(maybe, dict):
            return _normalize_keys(maybe)
    except Exception:
        pass
    return None

# === 문장 내 등장 위치 기반 정렬용 헬퍼 ===
def _pos_key_from_sentence(s: str, sentence: str) -> int:
    """
    s(보통 constraint 또는 time_expression)가 문장(sentence)에서 처음 등장한 위치를 반환.
    못 찾으면 매우 큰 값으로 반환해 뒤로 밀리게 함.
    범위 문자열은 '…'나 '~' 기준 좌측 토큰으로도 한 번 더 시도.
    """
    if not s:
        return 10**9
    i = sentence.find(s)
    if i != -1:
        return i
    # 범위면 좌측 토큰으로 재시도
    left = re.split(r'[…~]', s)[0]
    i2 = sentence.find(left)
    return i2 if i2 != -1 else 10**9

# ===== 휴리스틱(백업) =====
_REL_WORDS = ["무렵부터", "부터", "서부터", "에서부터", "까지",
              "이후에", "이후", "이전에", "이전",
              "동안", "사이에", "에서", "무렵", "경", "에", "의", "으로",
              "후", "전", "가 되어서야", "이 되어서야"]
_REL_WORDS.sort(key=len, reverse=True)

# === 범위 패턴들 ===
RANGE_PATS = [
    re.compile(
        r'(?P<s>\d{1,2}월\s*(?:초순|중순|하순))\s*(?P<sep>부터|에서|~|−|–|—|-)\s*'
        r'(?P<e>(?:\d{1,2}월\s*)?(?:초순|중순|하순))'
        r'(?:\s*(?P<tail_base>사이에|까지|에)'
        r'(?:(?P<post>는|도|만)|(?=\s|[)\]}",.…\.!?:;]|$)))?'
    ),
    re.compile(
        r'(?P<prefix>(?:기원전|기원후|서기)\s*)?'
        r'(?P<snum>\d{1,4})(?P<sunit>년|월|일|세기)?'
        r'\s*(?P<sep>~|−|–|—|-)\s*'
        r'(?P<eprefix>(?:기원전|기원후|서기)\s*)?'
        r'(?P<enum>\d{1,4})(?P<eunit>년|월|일|세기)'
        r'(?:\s*(?P<tail_base>사이에|까지|에)'
        r'(?:(?P<post>는|도|만)|(?=\s|[)\]}",.…\.!?:;]|$)))?'
    ),
    re.compile(
        r'(?P<s>\d{3,4}년\s*\d{1,2}월(?:\s*\d{1,2}일)?)\s*(?P<sep>부터|에서|~|−|–|—|-)\s*'
        r'(?P<e>\d{1,2}월(?:\s*\d{1,2}일)?)'
        r'(?:\s*(?P<tail_base>사이에|까지|에)'
        r'(?:(?P<post>는|도|만)|(?=\s|[)\]}",.…\.!?:;]|$)))?'
    ),
    re.compile(
        r'(?P<s>\d{3,4}년(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?)\s*(?P<sep>부터|에서|~|−|–|—|-)\s*'
        r'(?P<e>\d{1,4}년(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?)'
        r'(?:\s*(?P<tail_base>사이에|까지|에)'
        r'(?:(?P<post>는|도|만)|(?=\s|[)\]}",.…\.!?:;]|$)))?'
    ),
    re.compile(
        r'(?P<s>\d{3,4}년\s*\d{1,2}월\s*\d{1,2}일)\s*(?P<sep>부터|에서|~|−|–|—|-)\s*'
        r'(?P<e>\d{1,2}일)'
        r'(?:\s*(?P<tail_base>사이에|까지|에)'
        r'(?:(?P<post>는|도|만)|(?=\s|[)\]}",.…\.!?:;]|$)))?'
    ),
    re.compile(
        r'(?P<s>\d{1,2}월(?:\s*\d{1,2}일)?)\s*(?P<sep>부터|에서|~|−|–|—|-)\s*'
        r'(?P<e>\d{1,2}월(?:\s*\d{1,2}일)?)'
        r'(?:\s*(?P<tail_base>사이에|까지|에)'
        r'(?:(?P<post>는|도|만)|(?=\s|[)\]}",.…\.!?:;]|$)))?'
    ),
    re.compile(
        r'(?P<s>\d+세기(?:\s*(?:초|중반|후반))?)\s*(?P<sep>부터|~|−|–|—|-)\s*'
        r'(?P<e>\d+세기(?:\s*(?:초|중반|후반))?)\s*(?P<tail_base>까지)?(?P<post>는|도|만)?'
    ),
]

# === 단일 표현 패턴 ===
SINGLE_PAT = re.compile(
    r'('
    r'(?:[12]\d{3})\s*[~−–—-]\s*(?:\d{2}|\d{4})'
    r'|'
    r'(?:같은\s*해|그\s*해|그해|해당\s*해|해당\s*연도|당해)\s*(?:\d{1,2}월(?:\s*\d{1,2}일)?)'
    r'|'
    r'(?:최근|요즘|현재|그때|그\s*시절|그\s*무렵|당시)'
    r'|'
    r'\d{3,4}년\s*한\s*해'
    r'|'
    r'\d{1,4}년대(?:\s*(?:초(?:반)?|중반|후반|무렵|기준|당시))?'
    r'|'
    r'\d{3,4}년\s*현재'
    r'|'
    r'(?:기원전|기원후|서기)\s*\d{1,4}년(?!대)'
    r'(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?'
    r'(?:\s*(?:초(?:반)?|중순|말|무렵|기준|당시|중반|후반))?'
    r'|'
    r'\d{3,4}년(?:도)?(?!대)(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?(?:\s*(?:초(?:반)?|중순|말|무렵|경|기준|당시|중반|후반))?'
    r'|'
    r'\d{1,2}월(?:\s*\d{1,2}일)?(?:\s*(?:초|중순|말|무렵|기준|당시))?'
    r'|'
    r'\d{1,2}일'
    r'|'
    r'(?:기원전|기원후|서기)\s*\d+세기(?:\s*(?:초(?:반)?|중반|후반|말|무렵|기준|당시))?'
    r'|'
    r'\d+세기(?:\s*(?:초(?:반)?|중반|후반|말|무렵|기준|당시))?'
    r')'
)

def _heuristic_extract_all(sentence: str):
    results = []
    range_records = []

    # 범위
    for pat in RANGE_PATS:
        for m in pat.finditer(sentence):
            gd = m.groupdict()
            if all(k in gd for k in ("snum", "sunit", "enum", "eunit", "sep")) and m.group("sep"):
                new_span = (m.start(), m.end())
                skip = False
                for (rs, re_), *_ in range_records:
                    if rs <= new_span[0] and new_span[1] <= re_:
                        skip = True
                        break
                if skip:
                    continue

                prefix    = (m.group("prefix") or "")
                eprefix   = (m.group("eprefix") or "")
                snum      = m.group("snum")
                sunit     = m.group("sunit")
                enum      = m.group("enum")
                eunit     = m.group("eunit")
                sep       = m.group("sep")
                tail_base = m.group("tail_base")
                post      = (m.group("post") or "")

                after_span = sentence[m.end():]
                if eunit in ("년", "월", "일") and re.match(r'\s*(후|전|가량|정도|남짓|쯤)\b', after_span):
                    continue
                if eunit == "년":
                    after_span = sentence[m.end():]
                    if re.match(r'\s*\d{1,2}월(?:\s*\d{1,2}일)?', after_span):
                        continue
                if sunit and eunit and (sunit != eunit):
                    continue

                sunit_full = sunit if sunit else eunit
                right_prefix = eprefix if eprefix.strip() else prefix

                s_text = f"{prefix}{snum}{sunit_full}"
                e_text = f"{right_prefix}{enum}{eunit}"

                matched_text = sentence[m.start():m.end()]
                has_space_around = re.search(r'\s' + re.escape(sep) + r'\s', matched_text) is not None
                sep_str = f" {sep} " if has_space_around else sep

                if sep in ("~", "−", "–", "—", "-"):
                    rel = "~" if sep in ("~", "−", "–", "—") else "-"
                    constraint = f"{prefix}{snum}{sunit_full}{sep_str}{right_prefix}{enum}{eunit}"
                else:
                    rel = "부터…까지"
                    constraint = f"{prefix}{snum}{sunit_full}부터 {right_prefix}{enum}{eunit}까지"

                te = f"{prefix}{snum}{sunit_full}…{right_prefix}{enum}{eunit}"
                if tail_base:
                    te = f"{s_text}…{e_text}"
                else:
                    te = f"{prefix}{snum}{sunit_full}…{right_prefix}{enum}{eunit}"

                results.append((te, rel, constraint))
                range_records.append(((m.start(), m.end()), s_text, e_text))
                continue

            # 기존 범위 처리
            new_span = (m.start(), m.end())
            skip = False
            for (rs, re_), *_ in range_records:
                if rs <= new_span[0] and new_span[1] <= re_:
                    skip = True
                    break
            if skip:
                continue

            s_text, e_text = m.group("s"), m.group("e")
            sep = m.group("sep")
            tail_base = m.group("tail_base")
            post = (m.group("post") or "")

            if tail_base:
                if "사이에" in tail_base:
                    rel = "에서…사이에" + post
                    constraint = f"{s_text}에서 {e_text} {tail_base}{post}"
                elif "까지" in tail_base:
                    if sep == "에서":
                        rel = "에서…까지" + post
                        constraint = f"{s_text}에서 {e_text}까지{post}"
                    else:
                        rel = "부터…까지" + post
                        constraint = f"{s_text}부터 {e_text}까지{post}"
                else:  # '에'
                    rel = "에" + post
                    constraint = f"{s_text}에서 {e_text} {tail_base}{post}"
            else:
                if sep in ("~", "−", "–", "—", "-"):
                    rel = "~" if sep in ("~", "−", "–", "—") else "-"
                    matched_text = sentence[m.start():m.end()]
                    tight = re.search(r'\S' + re.escape(sep) + r'\S', matched_text) is not None
                    sep_str = sep if tight else f" {sep} "
                    constraint = f"{s_text}{sep_str}{e_text}"
                elif sep == "부터":
                    rel = "부터…까지"
                    constraint = f"{s_text}부터 {e_text}까지"
                elif sep == "에서":
                    rel = "에서…까지"
                    constraint = f"{s_text}에서 {e_text}까지"
                else:
                    rel = "부터…까지"
                    constraint = f"{s_text}부터 {e_text}까지"

            te = f"{s_text}…{e_text}"
            results.append((te, rel, constraint))
            range_records.append(((m.start(), m.end()), s_text, e_text))

    # 단일
    single_items = []
    for m2 in SINGLE_PAT.finditer(sentence):
        te = m2.group(1)
        span_start, span_end = m2.start(), m2.end()
        tail_str = sentence[m2.end():]

        if re.fullmatch(r'\d{1,2}일', te):
            continue

        tail_str = re.sub(r'^\s*\([^)]*\)', '', tail_str)

        if re.fullmatch(r'[12]\d{3}년', te):
            m_merge = re.match(
                r'^\s*(?:\([^)]*\)\s*)?(?P<md>\d{1,2}월(?:\s*\d{1,2}일)?)',
                tail_str
            )
            if m_merge:
                md = m_merge.group('md')
                te = f'{te} {md}'
                advance = m_merge.end()
                tail_str = tail_str[advance:]
                span_end += advance
                tail_str = re.sub(r'^\s*\([^)]*\)', '', tail_str)

        m_tail_crit = re.match(r'(?P<ws>\s*)(?P<crit>(?:[을를]\s*)?기준으로)', tail_str)
        if m_tail_crit:
            ws = m_tail_crit.group('ws') or ''
            crit = m_tail_crit.group('crit')
            rel = crit.strip()
            constraint = te + ws + crit
            span_end = m2.end() + len(ws) + len(crit)
            single_items.append((te, rel, constraint, span_start, span_end))
            continue

        if re.fullmatch(r'(?:최근|요즘|현재|그때|그\s*시절|그\s*무렵|당시)', te):
            pass

        m_tail = re.match(
            r'(?P<ws>\s*)'
            r'(?P<rel>무렵부터|가\s*되어서야|이\s*되어서야|(?:(?:이었기|였기)\s*때문에)|'
            r'부터|서부터|에서부터|까지|이후에|이후|이전에|이전|동안|사이에|에서|무렵|경|에|엔|의|으로|후|전|이나|나)'
            r'(?:(?P<post>는|도|만|의)|(?=\s|[)\]}",.…\.!?:;]|$))',
            tail_str
        )
        if m_tail:
            ws = m_tail.group('ws') or ''
            rel_base = m_tail.group('rel')
            post = m_tail.group('post') or ''

            if rel_base in ("이후에", "이전에"):
                te = te + " " + ("이후" if rel_base.startswith("이후") else "이전")
                rel_token = "에"
                rel = rel_token + post
                constraint = te + ws + rel_token + post
                span_end = m2.end() + len(ws) + len(rel_token) + len(post)
            elif rel_base in ("이후", "이전"):
                te = te + " " + rel_base
                rel = None
                constraint = te
                span_end = m2.end() + len(ws) + len(rel_base)
            else:
                rel = rel_base + post
                constraint = te + ws + rel_base + post
                span_end = m2.end() + len(ws) + len(rel_base) + len(post)
        else:
            m_topic = re.match(r'(?P<ws>\s*)(?P<tok>는|은)(?=\s|[)\]}",.…\.!?:;]|$)', tail_str)
            if m_topic:
                ws  = m_topic.group('ws') or ''
                tok = m_topic.group('tok')
                rel = tok
                constraint = te + ws + tok
                span_end = m2.end() + len(ws) + len(tok)
            else:
                rel = None
                constraint = te

        single_items.append((te, rel, constraint, span_start, span_end))

    season_spans = [
        (s0, e0) for (te, _r, _c, s0, e0) in single_items
        if re.match(r'^\s*[12]\d{3}\s*[~−–—-]\s*(?:\d{2}|\d{4})\s*$', te)
    ]
    if season_spans:
        filtered = []
        for te, rel, cons, s0, e0 in single_items:
            is_naked_year = re.match(r'^[12]\d{3}$', te) is not None
            inside_season = any(s0 >= ss and e0 <= ee for ss, ee in season_spans)
            if is_naked_year and inside_season:
                continue
            filtered.append((te, rel, cons, s0, e0))
        single_items = filtered

    filtered_single_tmp = []
    for te, rel, cons, s0, e0 in single_items:
        drop = False
        if rel in ("부터", "서부터", "에서부터", "까지",
                   "부터는", "서부터는", "에서부터는", "까지는",
                   "부터도", "서부터도", "에서부터도", "까지도",
                   "부터만", "서부터만", "에서부터만", "까지만"):
            for (rs, re_), s_text, e_text in range_records:
                if s0 >= rs and e0 <= re_:
                    drop = True
                    break
        if not drop:
            filtered_single_tmp.append((te, rel, cons, s0, e0))

    filtered_single_tmp2 = []
    for te, rel, cons, s0, e0 in filtered_single_tmp:
        drop = False
        for (_span, s_text, e_text) in range_records:
            if te == s_text or te == e_text:
                drop = True
                break
        if not drop:
            filtered_single_tmp2.append((te, rel, cons, s0, e0))

    filtered_single = []
    dur_pat = re.compile(r'^\d{1,3}(일|주|주일|개월|달|시간|분|초|년)$')
    for te, rel, cons, s0, e0 in filtered_single_tmp2:
        drop = False
        if dur_pat.match(te):
            after = sentence[e0:]
            if re.match(r'\s*간', after):
                drop = True
        if not drop and re.match(r'^\d{1,4}(년|월|일)$', te):
            after = sentence[e0:]
            after_l = after.lstrip()
            if after_l.startswith(("후", "전")):
                drop = True
            elif re.match(r'^\s*(가량|정도|남짓|쯤|이내)\b', after):
                drop = True

        if not drop:
            filtered_single.append((te, rel, cons))

    results.extend(filtered_single)
    return results or [(None, None, None)]

# ===== 수식어구(이번/다음/지난/올) & 4필드 변환 =====
def _extract_modifier(sentence: str, start_idx: int):
    if start_idx <= 0:
        return None
    prefix = sentence[:start_idx].rstrip()
    tail = prefix[-3:]
    for cand in ["이번", "다음", "지난", "올"]:
        if tail.endswith(cand):
            return cand
    return None

def _to_four_field_output(sentence: str, te: str | None, tr: str | None, cs: str | None):
    if cs is None and te is not None:
        cs = f"{te}{tr}" if tr else te

    mod = None
    anchor = None
    if cs and (sentence.find(cs) != -1):
        anchor = sentence.find(cs)
    elif te and (sentence.find(te) != -1):
        anchor = sentence.find(te)
    if anchor is not None:
        mod = _extract_modifier(sentence, anchor)

    return {
        "sentence": sentence,
        "temporal expression": cs,
        "수식어구": mod,
        "timestamp": te,
        "relation": tr,
    }

# ===== 연도 포함 여부 필터 =====
_YEAR_PAT = re.compile(r'(?:기원전|기원후|서기)?\s*\d{3,4}\s*년(?!대)')

def _contains_year(text: str | None) -> bool:
    """'\d{3,4}년' (년대 제외) 또는 접두사+년 형태가 포함되어야 True."""
    if not text:
        return False
    return _YEAR_PAT.search(text) is not None

# ===== SIGTERM/INT 처리 (tmux/nohup에서 안전 종료) & PID 파일 =====
STOP_REQUESTED = False

def _handle_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logger.warning(f"종료 신호 수신 (signum={signum}). 현재 문장 처리 후 안전하게 중단합니다…")

def _write_pidfile():
    try:
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        logger.exception("PID 파일 기록 실패")

def _cleanup_pidfile():
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)
atexit.register(_cleanup_pidfile)

# ===== 모델 호출 =====
def extract_time_constraint(model, tokenizer, sentence: str):
    """
    후보들을 수집 → 병합/정렬 → (연도 없는 후보 제거) → (전부 or Top-1) → 4필드 반환
    """
    prompt = build_prompt(sentence, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1536,
    ).to(model.device)

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    gen_tokens = outputs[0][input_len:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # 모델 결과 1건
    data = _parse_model_json(decoded)
    model_items = []
    if isinstance(data, dict):
        data = _normalize_keys(data)
        te = data.get("time_expression")
        tr = data.get("temporal_relation")
        cs = data.get("constraint")
        if isinstance(te, str) or te is None:
            if cs is None and te is not None:
                cs = f"{te}{tr}" if tr else te
            model_items.append((te, tr, cs))

    # 휴리스틱 여러 건
    heur_items = _heuristic_extract_all(sentence)

    # 병합 + 중복 제거
    all_items = model_items + heur_items
    seen = set()
    dedup = []
    for te, tr, cs in all_items:
        key = (cs or "", tr or "", te or "")
        if key in seen:
            continue
        seen.add(key)
        dedup.append((te, tr, cs))
    
    # 문장 내 등장 순 정렬
    dedup.sort(key=lambda x: _pos_key_from_sentence(x[2] or x[0] or "", sentence))

    # === ★ 연도 없는 후보 제거 (예: "12월 20일", "20일" 등) ===
    year_kept = [(te, tr, cs) for (te, tr, cs) in dedup if _contains_year(te) or _contains_year(cs)]

    # 후보가 없으면 전부 None
    if not year_kept:
        return {
            "sentence": sentence,
            "temporal expression": None,
            "수식어구": None,
            "timestamp": None,
            "relation": None,
        }

    # 여러 개 모두 반환 or Top-1 반환
    if EXTRACT_ALL:
        items = []
        for te, tr, cs in year_kept[:MAX_ITEMS]:
            items.append(_to_four_field_output(sentence, te, tr, cs))
        return items
    else:
        te, tr, cs = year_kept[0]
        return _to_four_field_output(sentence, te, tr, cs)

# ===== 메인 =====
def main():
    _write_pidfile()

    if not INPUT_FILE.exists():
        logger.error(f"입력 파일을 찾을 수 없습니다: {INPUT_FILE}")
        return

    torch.manual_seed(0)

    logger.info(f"토크나이저 로드: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("모델 로드 및 양자화 설정 적용")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    with INPUT_FILE.open("r", encoding="utf-8") as fin:
        sentences = json.load(fin)

    valid = [s for s in sentences if isinstance(s, str) and s.strip()]
    if not valid:
        logger.error("유효한 문장이 없습니다.")
        return

    k = min(SAMPLE_SIZE, len(valid))
    sampled = random.sample(valid, k=k)
    total = len(sampled)
    logger.info(f"랜덤 샘플 {total}개 선택 (전체 {len(valid)} 중)")
    logger.info(f"로그 파일: {LOG_FILE}")
    logger.info(f"PID 파일:  {PID_FILE}")

    processed = 0
    with OUTPUT_FILE.open("w", encoding="utf-8", buffering=1) as fout:
        for i, sent in enumerate(sampled, start=1):
            if STOP_REQUESTED:
                logger.warning("중단 신호 감지됨: 현재까지의 결과를 보존하고 종료합니다.")
                break
            try:
                res = extract_time_constraint(model, tokenizer, sent)
            except Exception as e:
                if DEBUG_DECODED:
                    logging.exception("extract_time_constraint 예외")
                res = {
                    "sentence": sent,
                    "temporal expression": None,
                    "수식어구": None,
                    "timestamp": None,
                    "relation": None,
                }
                if DIAG:
                    res["debug"] = str(e)

            # 리스트(여러 레코드)도, 단일 딕셔너리도 처리 + null 레코드 스킵
            if isinstance(res, list):
                wrote_any = False
                for item in res:
                    if _is_null_record(item):
                        continue  # 빠르게 스킵: 파일에도 출력하지 않음
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    preview = item.get("temporal expression")
                    print(f"[{i}/{total}] temporal expression: {preview!r}", flush=True)
                    wrote_any = True
                # wrote_any가 False면 아무것도 쓰지 않고 조용히 넘어감
            else:
                if not _is_null_record(res):
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    preview = res.get("temporal expression")
                    print(f"[{i}/{total}] temporal expression: {preview!r}", flush=True)
                # 전부 null이면 쓰지도, 출력하지도 않음

            processed += 1

    logger.info(f"처리 완료/중단. 기록된 샘플 수(문장 수): {processed}")

if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup_pidfile()
