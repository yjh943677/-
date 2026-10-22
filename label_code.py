import os
import sys
import json
import logging
import random
import re
import signal
from typing import Tuple
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# =========================
# 실행/운영 설정
# =========================
EXAMPLE_PATH = '/home/intern25/llm/data/combined_domain_question.xlsx'
OUTPUT_JSONL = 'label_results.jsonl'   # 저장 스키마: {"label","question"}
LOG_PATH     = 'run.log'
SAVE_EVERY   = 50

FEW_SHOT = 2
BATCH_SIZE = 16
TARGET_PER_LABEL = 5000
MAX_PER_DOMAIN = 556

# =========================
# 휴리스틱 설정
# =========================
HEURISTIC_MODE = 'relabel'           # 'filter' 또는 'relabel'
HEURISTIC_MARGIN_THRESH = 3          # 재라벨 기본 강도
SALVAGE_UNCHANGED = True             # unchanged 자동 정정 후 저장

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 총 3개 사용
torch.backends.cudnn.benchmark = True

# =========================
# 로거(파일+콘솔)
# =========================
def setup_logger():
    logger = logging.getLogger("qgen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
    fh.setFormatter(fmt); fh.setLevel(logging.INFO); logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); ch.setLevel(logging.INFO); logger.addHandler(ch)
    return logger

logger = setup_logger()

# =========================
# 유사도(중복 방지)
# =========================
try:
    from rapidfuzz import fuzz
    def is_similar(a: str, b: str, thresh: int = 85) -> bool:
        return fuzz.token_sort_ratio(a, b) >= thresh
except ImportError:
    from difflib import SequenceMatcher
    def is_similar(a: str, b: str, thresh: int = 85) -> bool:
        return SequenceMatcher(None, a, b).ratio() * 100 >= thresh

# =========================
# 데이터 로드 & 그룹화
# =========================
def load_examples(path: str):
    logger.info(f"Loading style examples from {path}")
    df = pd.read_excel(path, dtype=str)

    # UNK 제거
    df = df[df['domain'].astype(str) != 'UNK']

    # 도메인 이름에서 한글 토큰만
    df['domain'] = (
        df['domain']
          .astype(str)
          .str.strip()
          .str.split()
          .apply(lambda toks: next((t for t in toks if re.search('[가-힣]', t)), toks[0]))
    )

    df['domain'] = df['domain'].astype(str).str.replace('/', '_')
    df = df.dropna(subset=['question'])

    examples = list(zip(df['domain'], df['question']))
    logger.info(f"Loaded {len(examples)} examples")

    from collections import defaultdict
    examples_by_domain = defaultdict(list)
    for dom, q in examples:
        examples_by_domain[dom].append(q)

    domains = list(examples_by_domain.keys())
    logger.info(f"Grouped into {len(domains)} domains")
    return examples_by_domain, domains

# =========================
# 텍스트 후처리 (업데이트)
# =========================
def clean_text(q: str) -> str:
    q = re.sub(r'[^가-힣a-zA-Z0-9\s\?\.,:%/\-~℃]', '', q)

    # 한글-영/숫자 사이 공백
    q = re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', q)
    q = re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', q)

    q = re.sub(r'\s+', ' ', q).strip()
    q = re.sub(r'\s+([\?\.,:%])', r'\1', q)

    # 어색한 토큰 정리
    q = re.sub(r'\b(몸|몰까요|있죠|용)(\?)', r'\2', q)
    q = re.sub(r'\b인 가\?', '인가?', q)
    q = re.sub(r'\b이 가\?', '이가?', q)
    q = re.sub(r'\b입 니까\?', '입니까?', q)
    q = re.sub(r'\b인가 요\?', '인가요?', q)

    q = re.sub(r'(\d+)\s*몸', r'\1명', q)
    q = q.replace("몸으로", "명으로")
    q = q.replace("몸을", "명을")
    q = q.replace("몸이", "명이")
    q = q.replace("몸은", "명은")

    # 숫자 + 단위 결합 확장
    units = [
        '년','월','일','시','분','초','명','건','회','대','개','개월','주','분기',
        '도','℃','%','kg','g','t','m','cm','mm','km','kmh','km/h','억원','조원'
    ]
    pattern = r'(\d+)\s+(' + '|'.join(units) + r')\b'
    q = re.sub(pattern, r'\1\2', q)

    q = re.sub(r'\b(kg|g)\s*급\b', r'\1급', q)

    # ===== 추가 클린업 =====
    q = re.sub(r'\b몆\b', '몇', q)
    q = re.sub(r'\b몃\b', '몇', q)
    q = re.sub(r'(은|는|이|가)\s*였(습니까\?)', r'였\2', q)
    q = re.sub(r'\bBTS\s+의\b', 'BTS의', q)
    q = re.sub(r'\b진\s+Jin\b', '진(Jin)', q)
    q = re.sub(r'(\d+)\s+(점슛|명|회|개|권|곡|명령|선수)', r'\1\2', q)
    q = re.sub(r'몇\s+도\s+인가요\?', '몇 도인가요?', q)
    q = q.replace("미세먼지 뿌리기", "미세먼지 농도")
    q = q.replace("녙문", "논문")
    q = q.replace("KBS2TV", "KBS2")
    q = re.sub(r'(\d{4})\s+년', r'\1년', q)
    q = re.sub(r'(\d+)\s+세대', r'\1세대', q)
    q = q.replace("몄명", "몇 명")
    q = re.sub(r'(설립|창립|창단)\s*연차', r'\1 연도', q)
    q = re.sub(r'\b연\s*도\b', '연도', q)
    q = re.sub(r'년\s*도', '년도', q)
    q = re.sub(r'세계\s*은\s*행', '세계은행', q)
    q = q.replace("방영사", "방송사")
    # 샘플 보정: '몆월 몰 일'류 오타
    q = re.sub(r'몆월\s*몰\s*일', '몇 월 며칠', q)
    q = re.sub(r'\b몰\s*일\b', '며칠', q)
    q = re.sub(r'\s+', ' ', q).strip()
    # ======================

    return q

# =========================
# 유효성 검사 · 필터링 (업데이트)
# =========================
FORMAL_ENDINGS = ('습니까?', '인가요?')
_BAD_TOKENS = [
    '대한민구','대한민궁','대한민제','대한민화','대한민통','헬라','탑현','탱자',
    '몰려','몫차','몰디','몽골 용','탑니드','녹화 전쟁','녹전',
    '몸','몸으로','몸을','몸이','몸은',
    # 추가: 명백한 오타/혼종/개체유형 오류 트리거
    '몽동인기','폴짝비','지명정호석','어느 회사였습니까?','풀잎','옌게'
]

def is_valid_question(q: str) -> bool:
    s = q.strip()
    if len(s) < 10 or not s.endswith(FORMAL_ENDINGS):
        return False

    hangul = len(re.findall(r'[가-힣]', s))
    total  = len(re.findall(r'[^\s]', s))
    if total > 0 and (hangul / total) < 0.30:
        return False

    if any(tok in s for tok in ['[질문', 'OKay', 'First', '정답', '년 현재']):
        return False
    if any(bad in s for bad in _BAD_TOKENS):
        return False
    if re.search(r'\b(몸\?|몰까요\?|있죠\?|용\?|여\?)', s):
        return False

    _bad_semantic = [
        "미세먼지 농도 뿌리기",
        "미세먼지 뿌리기",
        "가장 인기 있는 축제",
        "풀타임 11 명의 선수가 득점",
        "삼성 알레고",
        "단일화 방식은 현재 어떻게",
        "몰리브",
        "최신우",
        "국제은련",
        "대한무늬",
        "알려졌습니다",
        "알려져 있습니다",
    ]
    if any(tok in s for tok in _bad_semantic):
        return False

    # 의견/추정형 문장 컷
    if re.search(r"(왜인 것 같습니까\?|어떤 이유라고 보십니까\?|인 것 같습니까\?|같습니까\?)", s):
        return False

    # 민감 PII/범죄 피해자 신원 노출형 질문 차단
    if re.search(r'(살인|성범죄|강간|납치|유괴|폭행).*(피해자|신원).*(누구|이름)', s):
        return False

    if re.search(r"(연차|경과\s*연수)\s*는\s*\d+\s*년\s*이상", s):
        return False

    # 전제 오류 예외(보수적으로 1건만): 서울+쓰나미
    if ("서울" in s and "쓰나미" in s):
        return False

    return True

# =========================
# 프롬프트 생성
# =========================
def format_prompt_batch(examples_by_domain, domains, labels, n_shots=FEW_SHOT):
    label_desc = {
        'changed': (
            "시간이 지나면 답이 달라질 수 있는 정보를 묻는 질문을 만드세요. "
            "질문 작성 시 반드시 정중한 존댓말('습니까?' 또는 '인가요?')을 사용하세요. "
            "질문에는 반드시 구체적 상황을 명시하고, 실제 검색창에 입력할 법한 자연스러운 문장 구조를 사용하세요.\n\n"
            "예시:\n"
            "- 대한민국의 다음 공휴일은 언제인가요?\n"
            "- 미국이 운용 중인 항공모함은 총 몇 대입니까?\n"
            "- 가장 마지막으로 열린 APEC 정상회의가 몇 차입니까?\n"
            "- 원피스는 한국에 몇 권까지 나왔습니까?\n"
            "- 올해 가장 빠른 폭염은 언제인가요?\n"
        ),
        'unchanged': (
            "언제 물어봐도 답이 변하지 않는, 명확한 사실이나 정의를 묻는 질문을 만드세요. "
            "질문 작성 시 반드시 정중한 존댓말('습니까?' 또는 '인가요?')을 사용하세요.\n"
            "다음 조건을 모두 지키세요:\n"
            "1) 연도(예: 1998년/2007년) 또는 정의/법칙/용어/주소/원리/원작/설립/시행/등록/개장/최초 등장/개봉 중 하나를 포함하세요.\n"
            "2) 아래 어휘/도메인은 사용 금지: 현재, 최근, 이번, 올해, 내년, 예정, 최신, 순위, 랭킹, 매출, 구독자수, 시가총액, 날씨, 일정, 개최.\n"
            "3) 상대시점 표현(예: 지금/요즘/금일/금주/금월)과 미래 표현(예: 출시될/열릴/예정)은 사용하지 마세요.\n\n"
            "예시:\n"
            "- 대한민국의 국보 1호는 무엇입니까?\n"
            "- 토성의 위성 중 가장 큰 것은 무엇인가요?\n"
            "- 재산세는 한 해에 몇 번 납부합니까?\n"
            "- 청탁금지법이 시행된 연도는 언제인가요?\n"
        )
    }

    prompts, target_domains = [], []
    for lbl in labels:
        shot_domains = random.sample(domains, k=min(n_shots, len(domains)))
        few_shot = []
        for dom in shot_domains:
            valid_exs = [q for q in examples_by_domain[dom] if is_valid_question(q)]
            if not valid_exs:
                valid_exs = examples_by_domain[dom]
            few_shot.append((dom, random.choice(valid_exs)))

        p = f"### 유형: {lbl}\n정의: {label_desc[lbl]}\n\n"
        p += (
            "아래 예시를 참고하되, 한국어 맞춤법과 문법을 철저히 준수하고, "
            "**반드시 실제로 존재하는 장소·사람·제품 이름만 사용**하세요. "
            "가상의 이름은 절대 생성하지 마세요.\n\n"
            "다양한 주제·어미·문장 형태로 새로운 질문을 생성하세요:\n\n"
        )
        for dom, q in few_shot:
            p += f"[도메인]: {dom}\n[질문]: {q}\n\n"

        target_dom = random.choice(domains)
        p += f"[도메인]: {target_dom}\n[질문]:"
        prompts.append(p)
        target_domains.append(target_dom)

    return prompts, target_domains

# =========================
# 체크포인트/재개 (도메인 포함/미포함 모두 허용)
# =========================
def load_existing(output_path: str):
    """
    기존 JSONL을 읽어 메모리 상태 복원.
    - {"label","domain","question"} 또는 {"label","question"} 모두 처리
    """
    from collections import defaultdict
    out = {'changed': defaultdict(set), 'unchanged': defaultdict(set)}
    domain_counts = {'changed': defaultdict(int), 'unchanged': defaultdict(int)}

    if not os.path.exists(output_path):
        logger.info("No existing output found, start fresh.")
        return out, domain_counts, 0

    logger.info(f"Resuming from existing {output_path}")
    resumed = 0
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            lbl = obj.get('label')
            q = obj.get('question')
            dom = obj.get('domain', '__misc__')
            if not lbl or not q:
                continue
            out[lbl][dom].add(q)
            domain_counts[lbl][dom] += 1
            resumed += 1
    logger.info(f"Resumed {resumed} records.")
    return out, domain_counts, resumed

# =========================
# 안전 종료 핸들러
# =========================
_SHUTDOWN = False
def _signal_handler(sig, frame):
    global _SHUTDOWN
    logger.warning(f"Received signal {sig}. Will stop after current batch.")
    _SHUTDOWN = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# =========================
# 범용 휴리스틱 라벨러 (업데이트)
# =========================
_REL_TIME = [
    "올해","이번","현재","지금","최근","요즘","금일","금주","금월",
    "작년","내년","오늘","어제","방금","최신","올해의","이번 달","이번 주","이번 시즌"
]
_REL_PERIOD_RE = re.compile(r'(최근|지난)\s*\d+\s*(개월|주|분기|년)\b')

_VAR_DOMAINS = [
    "환율","시가총액","주가","코스피","코스닥","금리","물가","확진자",
    "날씨","기온","폭염","미세먼지","실업률","매출",
    "매장 수","구독자수","인구","유가","환자 수","거래량"
]

_SUPERLATIVES = ["가장","최대","최소","최다","최고","최저","역대","신기록","기록","1위","2위"]

_CURRENT_OFFICE = ["대통령","장관","사무총장","CFO","총장","대표","감독","PD","의장","국무총리","장"]
_DYNAMIC_QUANT = ["매장 수","구독자수","시가총액","생산량","순위","랭킹","이용자 수","다운로드 수","시청률","거래량"]

_PAST_SNAPSHOT_VERBS = [
    "원작","개장","도입되었","도입된","시행되었","시행된","발표되었","발표된",
    "출시되었","출시된","정해졌","결정되었","지정되었","선정되었","수상했","수상한",
    "설립되었","설립된","착공","완공","등록되었","등록된","최초 등장","첫 등장"
]
_FUTURE_CUES = ["예정","계획","상용화","출시될","나와","열리나","열리나요","개최되나","개최될","발표될","공개될"]
_DATE_PHRASES = ["언제","날짜","일시","며칠","몇 일"]
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ORDINAL_ELECTION_RE = re.compile(r"제\s*\d+\s*(?:대|회)\s*(?:대통령\s*선거|전국동시지방선거|지방선거|총선)")

# === 추가 규칙 (누적/추이/운영/스냅샷/목록·위치/선거 일반/as-of/시즌기준) ===
_EVENT_WORDS = [
    "아시안 게임","올림픽","월드컵","선거","전국동시지방선거","지방선거","총선",
    "엑스포","박람회","페스티벌","대회","리그","시즌","콘서트","공연","전시",
    "국제회의","정상회의"
]
_CUMCOUNT_RE = re.compile(r"(?:총|모두)?\s*몇\s*(?:회|차례|번)\b|누적\s*(?:회|차례|번)\b")
_TREND_WORDS = ["경향성","추이","변화","증감","상승","하락","늘었","줄었","증가","감소"]
_ORG_ACTIVITY = ["활동","운영","프로그램","사업","정책","계획","라인업","진행 중","시행 중","운영되고","방송되는","편성","출연","참여율"]
_SNAPSHOT_METRICS = ["매출액","영업이익","순이익","GDP","실업률","인구","평균기온","평균 기온","최고기온","강수량","흥행수익","흥행 수익","관객수","관객 수","누적 관객"]
_SET_MEMBERSHIP_TRIGGERS = [
    "어떤 것들이 있습니까","어떤 것이 있습니까","목록","라인업","제품군","프로그램",
    "출연진","편성","작품에는 어떤 것이","어떤 작품","작품 목록","필모","작품에는 어떤 것"
]
_LOCATION_TRIGGER = re.compile(r"(어디에 위치해 있습니까\?|어디에 위치했습니까\?|어디에 있습니까\?)")
_FEATURE_TRIGGER = re.compile(r"특징")  # 앵커 없으면 변동성

# 선거 일반 키워드 확장
_ELECTION_GENERAL = ["선거","연방 선거","의회 선거","연방의회 선거","유럽의회 선거","총선","대선","지방선거","전국동시지방선거","대통령 선거"]
# as-of 앵커 (YYYY년 [MM월]까지)
_AS_OF_RE = re.compile(r"\b(19|20)\d{2}년(?:\s*\d{1,2}월)?\s*까지\b")
_RANK_RECORD_WORDS = ["기록","최다","최고","랭킹","순위","1위","2위","3점슛","득점","우승","홈런","골"]
# 시즌 기준 앵커
_AS_OF_SEASON_RE = re.compile(r"(?:어느|어떤)?\s*시즌\s*기준")

def _contains_any(s: str, toks) -> bool:
    return any(t in s for t in toks)

def classify_changed_heuristic(q: str) -> Tuple[str, int]:
    s = q.strip()
    changed_score = 0
    static_score  = 0

    has_year   = bool(_YEAR_RE.search(s))
    has_rel    = _contains_any(s, _REL_TIME) or bool(_REL_PERIOD_RE.search(s))
    has_date_q = _contains_any(s, _DATE_PHRASES)
    has_ordinal_election = bool(_ORDINAL_ELECTION_RE.search(s))
    has_election = has_ordinal_election or _contains_any(s, _ELECTION_GENERAL)

    # --- changed 신호 ---
    if has_rel: changed_score += 3
    if _contains_any(s, _SUPERLATIVES): changed_score += 3

    # 누적 이벤트 횟수(열린/개최/이벤트 + 몇 회/차례/번/누적) → 시계열적으로 증가 가능
    if (_CUMCOUNT_RE.search(s) and (_contains_any(s, ["열린","열렸","개최","개최된"]) or _contains_any(s, _EVENT_WORDS))):
        changed_score += 3

    # 추이/경향 표현
    if _contains_any(s, _TREND_WORDS):
        changed_score += 3

    # 직함은 과거 선거 문맥(연도 또는 서수형 선거)일 땐 changed로 보지 않음
    if _contains_any(s, _CURRENT_OFFICE):
        if not (has_election and (has_year or has_ordinal_election)):
            changed_score += 2

    if _contains_any(s, _VAR_DOMAINS): changed_score += 2
    if _contains_any(s, _DYNAMIC_QUANT): changed_score += 1
    if "현재 나이" in s or re.search(r"현재\s*나이", s): changed_score += 3
    if _contains_any(s, _FUTURE_CUES): changed_score += 2
    if has_date_q and (has_rel or _contains_any(s, _FUTURE_CUES)):
        changed_score += 2

    # 조직/기관 활동·운영·라인업 류 (연도 앵커 없으면 변동성 높음)
    if _contains_any(s, _ORG_ACTIVITY) and not has_year:
        changed_score += 3

    # 목록/제품군/프로그램/필모 등 '집합 구성' 질문 (연도 앵커 없으면 변동)
    if _contains_any(s, _SET_MEMBERSHIP_TRIGGERS) and not has_year:
        changed_score += 3

    # 위치 질의(연도 앵커 없으면 변동 가능)
    if _LOCATION_TRIGGER.search(s) and not has_year:
        changed_score += 2

    # 특징(앵커 없으면 주관/변동)
    if _FEATURE_TRIGGER.search(s) and not has_year:
        changed_score += 2

    # 법/조항 (연도 앵커 없으면 개정 가능)
    if re.search(r'(법|령|조례|규칙)\s*제\s*\d+\s*조', s) and not has_year:
        changed_score += 3

    # ever 질문
    if re.search(r'적이\s*있습니까\?', s):
        changed_score += 3

    # --- unchanged 신호 ---
    if has_year and _contains_any(s, _PAST_SNAPSHOT_VERBS):
        static_score += 3
    if (has_year or has_ordinal_election) and (has_date_q or re.search(r"(였나요|이었나요|었나요|였나|이었나|었나)", s)):
        static_score += 2
    if _contains_any(s, ["정의","라틴어","임기","주소","염분","위성","법칙","공식","수식","원리"]):
        static_score += 3
    if _contains_any(s, ["매년","항상","통상","보통","정기적으로","관례적으로"]):
        static_score += 2
    if has_year and _contains_any(s, _SUPERLATIVES):
        static_score += 1

    if (not has_year and not has_rel and
        not _contains_any(s, _VAR_DOMAINS) and
        not _contains_any(s, _CURRENT_OFFICE)):
        if _contains_any(s, ["정의","원작","개장","도입","시행","라틴어","임기","주소","첫 등장","최초 등장","발명","원리"]):
            static_score += 3

    # 과거 스포츠/행사 결과
    if (has_year or has_ordinal_election) and _contains_any(s, ["올림픽","동계올림픽","월드컵","아시안게임","금메달","우승","결승","결승전","본선","예선","시상식","개최","개최지","발표식","MVP"]):
        static_score += 3
    # 날짜/연도 개념어
    if _contains_any(s, ["연도","년도","발매일","출시일","개장일","시행일","개봉","개봉일","방영","방영일","발매","발표일"]):
        static_score += 3
    # 과거 정치/선거/재임 + 후보/소속
    if has_election and (has_year or has_ordinal_election) and _contains_any(s, ["재임","재임 기간","집권","당선","후계자","입당","출마","승리","패배","당선자","후보","소속","득표율"]):
        static_score += 5
    # '처음/최초 + 돌파/개최/발표/선정/지정/개장/설립/개봉'
    if _contains_any(s, ["처음","최초"]) and _contains_any(s, ["돌파","개최","발표","선정","지정","개장","설립","개봉"]):
        static_score += 3
    # 스냅샷 계열 표현 보강
    if has_year and _contains_any(s, ["발표된","출판된","발간된","개최된","진행된","처음 발표","최초 발표"]):
        static_score += 3
    # 전통/관습 신호
    if _contains_any(s, ["전통적으로","관습","풍습","관례"]):
        static_score += 2
    # 개봉/공개/취임식(비상대) 신호
    if _contains_any(s, ["개봉","개봉일","공개일","공개된 일자","취임식"]) and not _contains_any(s, ["현직","현재"]):
        static_score += 3
    # 설계/건축가/디자인 (건축물 고정 사실)
    if _contains_any(s, ["설계","건축가","디자인"]):
        static_score += 2
    # 설립/창립/창단 연도 표현 강화
    if _contains_any(s, ["설립연도","창립연도","창단연도","설립 연도","창립 연도","창단 연도"]):
        static_score += 3
    # 자연재해 스냅샷
    if has_year and _contains_any(s, ["태풍","지진","화산","산불","홍수","호우"]):
        static_score += 2
    # 연도 + 스냅샷 지표(지표값은 과거 스냅샷으로 고정)
    if has_year and _contains_any(s, _SNAPSHOT_METRICS):
        static_score += 4
    # 스냅샷 키워드(연도 없어도 일부 인정)
    if _contains_any(s, ["발표일","발매일","개봉일","방영일","출시일","당선자","우승자"]):
        static_score += 2
    # as-of 앵커 + 기록/랭킹 계열 → 스냅샷으로 간주
    if _AS_OF_RE.search(s) and _contains_any(s, _RANK_RECORD_WORDS):
        static_score += 4
    # '시즌 기준' 앵커 → 스냅샷 취급
    if _AS_OF_SEASON_RE.search(s) and _contains_any(s, _RANK_RECORD_WORDS + ["우승","MVP","득점","어시스트","세이브"]):
        static_score += 3

    pred = "changed" if changed_score > static_score else "unchanged"
    return pred, abs(changed_score - static_score)

# =========================
# ★ 자동 정정기: unchanged로 바꾸기
# =========================
def auto_fix_to_unchanged(s: str) -> str:
    t = s

    # 상대시점/미래 표현 제거
    for tok in _REL_TIME:
        t = t.replace(tok, '')
    for tok in _FUTURE_CUES:
        t = t.replace(tok, '')

    # 현재진행 → 과거 스냅샷
    t = re.sub(r'하고\s*있습니까\?', '했습니까?', t)
    t = re.sub(r'활동하고\s*있습니까\?', '활동했습니까?', t)
    t = re.sub(r'맡고\s*있습니까\?', '맡았습니까?', t)
    t = re.sub(r'되고\s*있습니까\?', '되었습니까?', t)
    t = re.sub(r'진행되고\s*있습니까\?', '진행되었습니까?', t)
    t = re.sub(r'누가\s*맡고\s*있습니까\?', '누구였습니까?', t)
    t = re.sub(r'누가\s*되고\s*있습니까\?', '누구였습니까?', t)
    t = re.sub(r'누구입니까\?', '누구였습니까?', t)

    # 추정형 제거
    t = re.sub(r'(인 것 같습니까\?|같습니까\?)', '였습니까?', t)

    # “현재 ~은 누구?” 류의 서술형 보정
    t = re.sub(r'현재\s*', '', t)

    # 정리
    t = re.sub(r'\s+', ' ', t).strip()
    t = clean_text(t)
    return t

# =========================
# 메인
# =========================
def main():
    if not os.path.exists(EXAMPLE_PATH):
        logger.error(f"EXAMPLE_PATH not found: {EXAMPLE_PATH}")
        sys.exit(2)

    examples_by_domain, domains = load_examples(EXAMPLE_PATH)

    out_path = os.path.abspath(OUTPUT_JSONL)
    logger.info(f"Output JSONL will be written to: {out_path}")

    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
    )

    logger.info("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-32B",
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    try:
        model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    model.eval()
    logger.info("Model ready")

    from collections import defaultdict
    out, domain_counts, resumed = load_existing(out_path)

    total_target = TARGET_PER_LABEL * 2
    disable_bar = not sys.stdout.isatty()
    pbar = tqdm(total=total_target, desc="Generating", disable=disable_bar)

    done_so_far = sum(domain_counts['changed'].values()) + sum(domain_counts['unchanged'].values())
    pbar.update(min(done_so_far, total_target))

    generated_since_flush = 0
    since_last_sync = 0  # fsync 주기 제어

    while sum(domain_counts['changed'].values()) < TARGET_PER_LABEL or \
          sum(domain_counts['unchanged'].values()) < TARGET_PER_LABEL:

        if _SHUTDOWN:
            logger.info("Shutdown requested. Exiting main loop gracefully.")
            break

        rem_c = TARGET_PER_LABEL - sum(domain_counts['changed'].values())
        rem_u = TARGET_PER_LABEL - sum(domain_counts['unchanged'].values())

        # 부족한 쪽 우선 배치
        if rem_u >= rem_c:
            bu = min(BATCH_SIZE, rem_u)
            bc = max(0, BATCH_SIZE - bu)
        else:
            bc = min(BATCH_SIZE, rem_c)
            bu = max(0, BATCH_SIZE - bc)

        batch_labels = ['changed'] * bc + ['unchanged'] * bu
        if not batch_labels:
            break
        random.shuffle(batch_labels)

        prompts, target_domains = format_prompt_batch(examples_by_domain, domains, batch_labels)
        enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=320)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.50,
                top_p=0.75,
                repetition_penalty=1.15,
                no_repeat_ngram_size=2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        input_len = enc['input_ids'].shape[1]
        to_append = []

        for i, orig_lbl in enumerate(batch_labels):
            lbl = orig_lbl
            if sum(domain_counts[lbl].values()) >= TARGET_PER_LABEL:
                continue

            cand = tokenizer.decode(gen[i, input_len:], skip_special_tokens=True)
            cand = cand.split('\n', 1)[0].strip()
            cand = clean_text(cand)

            # 1) 유효성 검사
            if not is_valid_question(cand):
                continue

            # 2) 휴리스틱 라벨 검증/보정 (+ 자동 정정)
            heur_pred, margin = classify_changed_heuristic(cand)

            if HEURISTIC_MODE == 'filter':
                if heur_pred != lbl:
                    logger.info(f"Filtered by heuristic: orig={orig_lbl}, heur={heur_pred}, margin={margin}, q={cand}")
                    continue

            elif HEURISTIC_MODE == 'relabel':
                if orig_lbl == 'unchanged' and heur_pred != 'unchanged':
                    if SALVAGE_UNCHANGED:
                        # ★ 자동 정정 시도
                        fixed = auto_fix_to_unchanged(cand)
                        if is_valid_question(fixed):
                            f_pred, f_margin = classify_changed_heuristic(fixed)
                            if f_pred == 'unchanged':
                                logger.info(f"Auto-fixed to unchanged (margin={f_margin}) | {cand} -> {fixed}")
                                cand = fixed
                                lbl = 'unchanged'
                            else:
                                # 정정 실패 시: 버리지 않고 휴리스틱 라벨로 저장
                                logger.info(f"Auto-fix failed (still {f_pred}). Save as {f_pred}. | {cand} -> {fixed}")
                                cand = fixed
                                lbl = f_pred
                        else:
                            # 정정문이 유효성 탈락하면 휴리스틱 라벨로 저장
                            logger.info(f"Auto-fix invalid, save as {heur_pred}. | {cand} -> {fixed}")
                            cand = fixed
                            lbl = heur_pred
                    else:
                        # salvage 끈 경우: 기준에 따라 리라벨
                        if heur_pred != lbl and margin >= HEURISTIC_MARGIN_THRESH:
                            logger.info(f"Relabeled by heuristic: {orig_lbl} -> {heur_pred} (margin={margin}) | {cand}")
                            lbl = heur_pred

                else:
                    # changed 원본이 휴리스틱상 unchanged면 완화 기준으로 뒤집기
                    if orig_lbl == 'changed' and heur_pred == 'unchanged' and margin >= 1:
                        logger.info(f"Relabeled by heuristic (changed->unchanged): margin={margin} | {cand}")
                        lbl = 'unchanged'
                    elif heur_pred != lbl and margin >= HEURISTIC_MARGIN_THRESH:
                        logger.info(f"Relabeled by heuristic: {orig_lbl} -> {heur_pred} (margin={margin}) | {cand}")
                        lbl = heur_pred
            else:
                if heur_pred != lbl:
                    logger.info(f"[Unknown mode->filter] orig={orig_lbl}, heur={heur_pred}, margin={margin}, q={cand}")
                    continue

            # 3) 라벨 정합성 하드 게이트: unchanged 전용 (정정·재라벨 후 최종 cand/lbl 기준)
            if lbl == 'unchanged':
                # 잔여 상대시점/미래/운영성/목록/위치/특징 표현 제거 시도
                if _contains_any(cand, _REL_TIME) or _contains_any(cand, _FUTURE_CUES) \
                   or _contains_any(cand, ['순위','랭킹','일정','개최','최신']) \
                   or _contains_any(cand, _SET_MEMBERSHIP_TRIGGERS) \
                   or _LOCATION_TRIGGER.search(cand) \
                   or _FEATURE_TRIGGER.search(cand) and not _YEAR_RE.search(cand) \
                   or re.search(r"(시행\s*중|운영\s*중|진행\s*중|하고\s*있습니까|활동을\s*하고\s*있습니까|방송되는)", cand):
                    cand2 = auto_fix_to_unchanged(cand)
                    if is_valid_question(cand2):
                        c_pred, _ = classify_changed_heuristic(cand2)
                        if c_pred == 'unchanged':
                            cand = cand2
                        else:
                            lbl = c_pred
                            cand = cand2
                    else:
                        lbl = 'changed'

                # 스냅샷 앵커 요건 (엄격 게이트)
                has_year = bool(_YEAR_RE.search(cand))
                has_ordinal = bool(_ORDINAL_ELECTION_RE.search(cand))
                has_snapshot_metric = _contains_any(cand, _SNAPSHOT_METRICS)
                has_snapshot_verb = _contains_any(cand, [
                    "발표","발매","출시","개최","우승","당선","개봉","시행","설립","등록","개장","수상","선정","지정","방영","공표","발간","출판",
                    "발표된","출시된","개최된","개봉일","방영일","발매일"
                ])
                has_election_anchor = has_ordinal and _contains_any(cand, ["당선자","후보","득표율"])
                has_stable_category = _contains_any(cand, ["정의","법칙","공식","원리","라틴어","원자번호","화학식","지질시대"])

                if not ((has_year and (has_snapshot_metric or has_snapshot_verb)) or has_election_anchor or has_stable_category):
                    # 기준 미달이면 마지막으로 과거화 시도 후 실패 시 changed로 강등
                    cand2 = auto_fix_to_unchanged(cand)
                    if is_valid_question(cand2):
                        c_pred, _ = classify_changed_heuristic(cand2)
                        if c_pred == 'unchanged':
                            cand = cand2
                        else:
                            lbl = 'changed'
                            cand = cand2
                    else:
                        lbl = 'changed'

            # 4) 도메인 한도/중복 체크
            dom = target_domains[i]
            if domain_counts[lbl][dom] >= MAX_PER_DOMAIN:
                continue

            already = out[lbl].get(dom, set())
            if any(is_similar(cand, e) for e in already):
                continue
            if '__misc__' in out[lbl] and any(is_similar(cand, e) for e in out[l][ '__misc__']):
                continue

            # 5) 저장/카운트
            out[lbl][dom] = already
            out[lbl][dom].add(cand)
            domain_counts[lbl][dom] += 1

            to_append.append({"label": lbl, "question": cand})

        # 배치 결과 로깅/진척 업데이트
        if to_append:
            with open(out_path, 'a', encoding='utf-8') as f:
                for obj in to_append:
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                f.flush()
                since_last_sync += len(to_append)
                generated_since_flush += len(to_append)
                if since_last_sync >= SAVE_EVERY:
                    os.fsync(f.fileno())
                    since_last_sync = 0
            pbar.update(len(to_append))
            logger.info(f"Appended {len(to_append)} records to {out_path}")

        if generated_since_flush >= SAVE_EVERY:
            generated_since_flush = 0
            logger.info("Checkpoint: appended recent batch to disk.")

    pbar.close()
    logger.info("Done.")

# =========================
# 엔트리포인트
# =========================
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
