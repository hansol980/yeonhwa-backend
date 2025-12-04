# filters.py

import re

BAD_WORDS = [
    "씨발", "씨bal", "ㅅㅂ", "좆", "개새끼", "병신", "병신아", "son of bitch", "시발","싯발","씨발","시벌","씨벌","ㅅㅂ","끼발","띠발","띠밤","시발",
    "지랄", "썅", "염병", "미친", "ㅈ같", "씹", "마더퍼커", "motherfucker", "빗치", "bitch", "쓰발", "야발", "씌발", "씹발", "끼발", "띠발", "띠밤", "ㅅ1발",
    "시1발"
]

SEXUAL_WORDS = [
    "섹스", "야동", "포르노", "포르노", "페티쉬",
    "가슴", "젖", "엉덩이", "꼴려", "야해", "에로", "sex", "야스"
]

def has_any(patterns, text):
    t = text.lower()
    return any(p in t for p in patterns)

def classify_user_input(user_text: str):
    """
    매우 단순한 1차 필터:
    - return: {"label": "ok" | "abuse" | "sexual", "detail": "..."}
    """
    text = user_text.strip()

    if has_any(BAD_WORDS, text):
        return {"label": "abuse", "detail": "욕설/모욕 표현 금지"}

    if has_any(SEXUAL_WORDS, text):
        return {"label": "sexual", "detail": "선정적/성적 표현 금지"}

    # 필요하면 더 세분화 가능: "self_harm", "hate", etc.
    return {"label": "ok", "detail": ""}
