import json
import numpy as np
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GOOGLE_API_KEY not found. Embeddings will fail.")

# JSON 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "choices.json")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


def normalize_punctuation(text: str) -> str:
    """문장부호를 의미 기반 태그로 치환"""
    if "?" in text and "!" in text:
        tone = "[TONE_MIXED]"
    elif "?" in text:
        tone = "[TONE_QUESTION]"
    elif "!" in text:
        tone = "[TONE_FORCE]"
    else:
        tone = "[TONE_NEUTRAL]"

    # 본문에서 문장부호 제거 (의미는 tone으로 보존)
    text = re.sub(r"[?!\.]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return f"{text} {tone}"


# 🔹 전역 캐시: choice id → embedding
choice_vec_cache: dict[str, list[float]] = {}


def get_embedding(text: str) -> list[float]:
    """Gemini API를 사용하여 텍스트 임베딩 생성"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="semantic_similarity"
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding Error: {e}")
        # 에러 시 0 벡터 반환 (서버 죽는 것 방지)
        return [0.0] * 768 


def build_choice_embedding(choice):
    """
    선택지 임베딩 생성 (캐시 포함):
    - embed_text 있으면 그걸 베이스로, 없으면 text 사용
    - tags / emotion 도 문자열로 붙여서 의미 강화
    """
    cid = choice["id"]
    if cid in choice_vec_cache:
        return choice_vec_cache[cid]

    base = choice.get("embed_text") or choice["text"]
    meta_tags = " ".join(choice.get("tags", []))
    meta_emotion = " ".join(choice.get("emotion", []))

    full = base
    if meta_tags:
        full += f" [태그: {meta_tags}]"
    if meta_emotion:
        full += f" [감정: {meta_emotion}]"

    vec = get_embedding(full)
    choice_vec_cache[cid] = vec
    return vec


def find_best_choice(user_text, scene_id):
    # 1) 해당 scene 선택지만 필터
    scene = next((s for s in data if s["scene_id"] == scene_id), None)
    if not scene:
        return []

    choices = scene["choices"]

    # 2) user 텍스트 정규화 + 임베딩
    norm_text = normalize_punctuation(user_text)
    user_vec = get_embedding(norm_text)

    # 3) 씬 내부 선택지 임베딩 + 코사인 유사도
    scores = []
    
    # user_vec이 유효한지 확인 (API 에러 등)
    user_norm = np.linalg.norm(user_vec)
    
    for choice in choices:
        choice_vec = build_choice_embedding(choice)
        choice_norm = np.linalg.norm(choice_vec)

        if user_norm == 0 or choice_norm == 0:
            sim = 0.0
        else:
            sim = np.dot(user_vec, choice_vec) / (user_norm * choice_norm)

        scores.append((sim, choice))

    # 4) 유사도 높은 순 정렬
    scores.sort(key=lambda x: x[0], reverse=True)

    # 5) 결과 리턴
    return [
        {
            "choice_id": c["id"],
            "text": c["text"],
            "score": c["score"],
            "tags": c.get("tags", []),
            "emotion": c.get("emotion", []),
            "similarity": float(sim),
        }
        for sim, c in scores
    ]


# 테스트용
if __name__ == "__main__":
    result = find_best_choice(
        "조용한게 늘 좋기만 한 것은 아닌듯 합니다.",
        scene_id="chapter1_garden_jinhyo",
    )
    from pprint import pprint
    pprint(result[:5])