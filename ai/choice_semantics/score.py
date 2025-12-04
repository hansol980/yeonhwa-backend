import json
import numpy as np
from judge import find_best_choice
from judge import normalize_punctuation
from filters import classify_user_input
from difflib import SequenceMatcher  # 문자열 유사도


# ------------------------------
# 유틸 함수들
# ------------------------------
def softmax(similarities, temperature=0.3):
    """
    similarities: [s1, s2, ...]
    temperature: 작을수록 가장 높은 similarity에 더 쏠리게 됨.
    """
    sims = np.array(similarities, dtype=float)
    sims = sims / max(temperature, 1e-6)
    sims = sims - sims.max()
    exp = np.exp(sims)
    probs = exp / exp.sum()
    return probs


def calculate_lexical_similarity(user_text: str, choice_text: str) -> float:
    # judge와 동일한 규칙으로 정규화
    u_clean = normalize_punctuation(user_text)
    c_clean = normalize_punctuation(choice_text)

    # [TONE_*] 토큰은 문자열 유사도에는 안 쓰고 싶으면 제거
    u_clean = u_clean.replace("[TONE_QUESTION]", "").replace("[TONE_FORCE]", "")\
                     .replace("[TONE_MIXED]", "").replace("[TONE_NEUTRAL]", "").strip()
    c_clean = c_clean.replace("[TONE_QUESTION]", "").replace("[TONE_FORCE]", "")\
                     .replace("[TONE_MIXED]", "").replace("[TONE_NEUTRAL]", "").strip()

    # 공백 없애고 비교 (지금 스타일 유지하고 싶으면)
    u_clean = u_clean.replace(" ", "")
    c_clean = c_clean.replace(" ", "")

    if not u_clean or not c_clean:
        return 0.0

    matcher = SequenceMatcher(None, u_clean, c_clean)
    return matcher.ratio()


def get_choice_group(choice_id: str) -> str:
    """
    choice_id에서 그룹(a/b/c)을 추출.
    예: 'baekdamwoo_1_1_1_b-1' -> 'b'
    """
    try:
        tail = choice_id.split("_")[-1]  # 'b-1'
        group = tail.split("-")[0]       # 'b'
        return group
    except Exception:
        return ""


# ------------------------------
# 메인 함수
# ------------------------------
def compute_affinity_delta(
    user_text: str,
    scene_id: str,
    top_k: int = 3,
    temperature: float = 0.3,
    min_sim_ratio: float = 0.0,
    abs_min_sim: float = 0.35,
):
    """
    user_text: 플레이어가 실제로 입력한 발화
    scene_id : choices.json 안의 scene_id
    """

    # 0) 안전 필터 먼저
    # 1번째, 2번째일 때는 호감도 변화는 0, 경고 메시지 출력
    # 3번째부터는 호감도 -50
    safe = classify_user_input(user_text)
    if safe["label"] != "ok":
        if safe["label"] == "abuse":
            delta = -50.0
            print("욕설/모욕 표현은 금지됩니다.")
        elif safe["label"] == "sexual":
            delta = -50.0
            print("선정적/성적 표현은 금지됩니다.")
        else:
            delta = 0.0

        return {
            "status": "abuse",
            "mode": "blocked",
            "user_text": user_text,
            "scene_id": scene_id,
            "delta": delta,
            "matches": [],
            "probs": [],
            "moderation": safe,
            "note": f"blocked_by_safety ({safe['label']})",
        }

    # 1) 후보 검색
    candidates = find_best_choice(user_text, scene_id)
    if not candidates:
        return {
            "status": "error",
            "mode": "none",
            "user_text": user_text,
            "scene_id": scene_id,
            "delta": 0.0,
            "matches": [],
            "probs": [],
            "note": "no candidates found for this scene_id",
        }

    # 각 후보에 문자열 유사도 계산
    for c in candidates:
        c["lex_sim"] = calculate_lexical_similarity(user_text, c["text"])

    # 의미/문자 기반 최대 유사도
    max_sim = max(c["similarity"] for c in candidates)
    best_lex = max(c["lex_sim"] for c in candidates)

    # “이 말이 뭔 소린지 모르겠다” 방어막
    # (벡터도 낮고, 문자열도 별로 안비슷하면 거부)
    LEX_MIN_FOR_UNDERSTAND = 0.2
    if (max_sim < abs_min_sim) and (best_lex < LEX_MIN_FOR_UNDERSTAND):
        return {
            "status": "low_similarity",
            "mode": "none",
            "user_text": user_text,
            "scene_id": scene_id,
            "delta": 0.0,
            "matches": [],
            "probs": [],
            "note": (
                f"too low similarity: max_sim={max_sim:.3f}, best_lex={best_lex:.3f}, "
                f"abs_min_sim={abs_min_sim}, LEX_MIN_FOR_UNDERSTAND={LEX_MIN_FOR_UNDERSTAND}"
            ),
        }

    # --------------------------------------------------
    # (2) 스냅 모드: 선택지를 거의 그대로 따라쳤을 때
    # --------------------------------------------------
    # 이 값 이상이면 사실상 “그 선택지 고른 것과 같다”로 본다.
    LEX_STRICT_THRESHOLD = 0.7

    if best_lex >= LEX_STRICT_THRESHOLD:
        mode = "snap"

        best_cand = max(candidates, key=lambda x: x["lex_sim"])
        target_group = get_choice_group(best_cand["choice_id"])

        # 같은 그룹(a/b/c)만 사용 (혹시 실패하면 그냥 전체 사용)
        group_candidates = [
            c for c in candidates
            if get_choice_group(c["choice_id"]) == target_group
        ] or candidates

        sims = [c["similarity"] for c in group_candidates]
        probs = softmax(sims, temperature=temperature)

        base_scores = [c["score"] for c in group_candidates]
        delta = float(sum(p * s for p, s in zip(probs, base_scores)))

        matches = []
        for p, c in zip(probs, group_candidates):
            matches.append({
                "choice_id": c["choice_id"],
                "text": c["text"],
                "score": c["score"],
                "similarity": c["similarity"],
                "lex_sim": c["lex_sim"],
                "weight": p,
            })

        return {
            "status": "success",
            "mode": mode,
            "user_text": user_text,
            "scene_id": scene_id,
            "delta": delta,
            "matches": matches,
            "probs": probs.tolist(),
            "max_sim": max_sim,
            "best_lex": best_lex,
            "snap_group": target_group,
        }

    # --------------------------------------------------
    # (3) 연속 모드: 애드립/미묘한 변형 → 연속값
    # --------------------------------------------------
    mode = "continuous"

    # min_sim_ratio 필터: 상위 max_sim의 일정 비율 이상만 사용할 수도 있음
    if min_sim_ratio > 0.0:
        sim_threshold = max_sim * min_sim_ratio
        filtered = [c for c in candidates if c["similarity"] >= sim_threshold]
        if filtered:
            candidates = filtered

    lexical_weight = 0.2  # 벡터 vs 문자열 비중

    # final_score 계산 (벡터 + 문자열 가중합)
    for c in candidates:
        vec_sim = c["similarity"]
        lex_sim = c["lex_sim"]
        c["final_score"] = vec_sim + lexical_weight * lex_sim

    # ★ 여기서 반드시 c가 아니라 lambda c 써야 함!
    candidates.sort(key=lambda c: c["final_score"], reverse=True)

    # 3-3) ★ 연속 모드 안에서도 "약한 그룹 잠금" 추가 ★
    # top 후보와 같은 그룹만 남기되, 그 후보의 lex_sim이 어느 정도 이상일 때만
    GROUP_LOCK_LEX_THRESHOLD = 0.25  # 이 값은 네 로그 기준으로 잡은 것

    top_candidate = candidates[0]
    top_group = get_choice_group(top_candidate["choice_id"])
    top_lex = top_candidate["lex_sim"]

    if top_group and top_lex >= GROUP_LOCK_LEX_THRESHOLD:
        group_only = [
            c for c in candidates
            if get_choice_group(c["choice_id"]) == top_group
        ]
        if group_only:
            candidates = group_only
            # 그룹 필터 후 다시 final_score 기준 정렬
            candidates.sort(key=lambda c: c["final_score"], reverse=True)

    # 상위 top_k만 사용
    top = candidates[:top_k]

    sims = [c["final_score"] for c in top]
    probs = softmax(sims, temperature=temperature)

    base_scores = [c["score"] for c in top]
    delta = float(sum(p * s for p, s in zip(probs, base_scores)))

    matches = []
    for p, c in zip(probs, top):
        matches.append({
            "choice_id": c["choice_id"],
            "text": c["text"],
            "score": c["score"],
            "similarity": c["similarity"],
            "lex_sim": c["lex_sim"],
            "final_score": c["final_score"],
            "weight": p,
        })

    return {
        "status": "success",
        "mode": mode,
        "user_text": user_text,
        "scene_id": scene_id,
        "delta": delta,
        "matches": matches,
        "probs": probs.tolist(),
        "max_sim": max_sim,
        "best_lex": best_lex,
    }


# ------------------------------
# 간단 테스트
# ------------------------------
if __name__ == "__main__":
    # 테스트 문장 바꿔가면서 확인하면 됨
    # test_text = "긴말하지 않겠습니다. 길을 비키세요."
    test_text = input("지금 내가 하고 싶은 말은...\n")
    scene_id = "chapter1_market_hwayajin"

    res = compute_affinity_delta(
        test_text,
        scene_id,
        top_k=3,
        temperature=0.6,
        min_sim_ratio=0.0,
    )

    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("=== mode ===", res.get("mode"))
    print("=== delta ===", res.get("delta"))
