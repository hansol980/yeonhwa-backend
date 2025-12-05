# -*- coding: utf-8 -*-
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json

from persona_loader import load_persona_cache
from choice_semantics.score import compute_affinity_delta

# ---------- 환경 설정 ----------
load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY가 설정되어 있지 않습니다 (.env 확인).")
    sys.exit(1)

genai.configure(api_key=api_key)

# ---------- 페르소나 로드 ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
persona_dir = os.path.join(current_dir, "json_list", "persona")
persona_cache = load_persona_cache(persona_dir)


def infer_partner_from_scene(scene_id: str) -> str:
    """
    scene_id에서 대화 상대 캐릭터 이름 추론.
    """
    sid = scene_id.lower()
    if "baekdamwoo" in sid or "baekdam" in sid or "백담우" in sid:
        return "백담우"
    if "hwayajin" in sid or "hwaya" in sid or "화야진" in sid:
        return "화야진"
    if "jinhyo" in sid or "jin_hyo" in sid or "진효" in sid:
        return "진효"
    if "sohyunsoo" in sid or "sohyun" in sid or "소현수" in sid:
        return "소현수"
    return "백담우"


def build_dialogue_prompt(
    user_text: str,
    scene_id: str,
    delta: float,
    result: dict,
    persona_cache: dict,
):
    """
    Gemini API에 넣을 system / user용 프롬프트를 생성.
    """
    partner = infer_partner_from_scene(scene_id)

    princess_persona = persona_cache.get("공주", "").strip()
    partner_persona = persona_cache.get(partner, "").strip()

    best_match = None
    if result.get("matches"):
        best_match = result["matches"][0]
    best_text = best_match["text"] if best_match else "(매칭 없음)"
    best_score = best_match["score"] if best_match else 0

    # delta 기반 간단 톤 레이블
    if delta > 2:
        tone_label = "strong_positive"
    elif delta > 0:
        tone_label = "mild_positive"
    elif delta > -2:
        tone_label = "neutral_or_tense"
    else:
        tone_label = "negative"

    # ---- system 프롬프트 ----
    safety_preamble = """
[System Note]
You are a creative writing assistant for a historical romance novel.
The context involves dramatic tension and martial arts themes, not real violence.
Please generate the dialogue focusing on emotional expressions.
"""

    system_prompt = safety_preamble + f"""
너는 동양풍 판타지 연애 시뮬레이션 게임 <연화록>의 시나리오 작가이자 대사 생성 엔진이다.

[역할]
- 공주와 '{partner}'의 성격, 말투, 관계를 고려하여 "플레이어가 방금 한 말 이후" 이어지는 7줄 대사를 작성한다.
- 대사는 게임 로그에 바로 사용 가능한 형태여야 한다.

[형식 규칙]
- 오직 '공주'와 '{partner}' 두 사람만 대사에 등장한다.
- 출력은 '정확히' 7줄의 대사만 생성한다.
- 각 줄은 반드시 '이름: 내용' 형식을 따른다. (예: 공주: 지금 뭐라고 하셨습니까?)
- 말 앞뒤에 들어가는 이중 괄호 ((...)) 형태의 행동 묘사는 절대 쓰지 마라.
- 대신 공주의 내면 독백은 한 겹 괄호(...)로만 쓴다. 예: 공주: (나는 잠시 숨을 골랐다.)
- 각 줄은 '공주 :' 또는 '{partner} :' 로 시작해야 한다. (콜론 뒤에는 한 칸 띄우기)
- 총 줄 수는 정확히 7줄이다.
- 한 줄에는 1~2문장 정도만 쓴다.
- 이미 플레이어가 말한 대사(입력 문장)는 다시 반복하지 말고, 그 직후 상황부터 이어서 쓴다.
- 영어를 섞지 말고, 게임 분위기에 맞는 한국어 대사만 출력하라.

[호감도 delta 해석]
- 이번 선택으로 인한 호감도 변화 delta: {delta:.3f}
- tone_label: {tone_label}
- delta가 양수일수록, 방금 공주의 말이 상대에게 매력적/흥미롭게 느껴진다. (호기심, 끌림, 약한 설렘)
- delta가 음수일수록, 방금 공주의 말이 상대에게 경계/거리감으로 느껴진다.
- tone_label이 strong_positive일수록 '{partner}'는 공주에게 확실한 호감과 신뢰를 드러낸다.
- tone_label이 negative일수록 말은 예의를 유지하더라도, 감정선에는 거리감과 실망이 스며 있어야 한다.
- 대사의 분위기, 말투, 거리감(존대/반말/차가움/장난스러움)에 delta와 tone_label을 자연스럽게 반영하라.
- 공주의 내면 독백에는 "이번 선택이 좋은 선택인지, 나쁜 선택인지에 대한 자기 평가"가 은근히 드러나야 한다.
""".strip()

    # ---- user 프롬프트(상황 설명) ----
    
    # scene_prompts.json 로드
    prompts_path = os.path.join(os.path.dirname(__file__), "json_list", "scene_prompts.json")
    scene_context = []
    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)
            if scene_id in prompts_data:
                scene_context = prompts_data[scene_id].get("context", [])
    except Exception as e:
        print(f"Warning: Failed to load scene prompts: {e}")

    # 컨텍스트 리스트를 문자열로 변환
    context_str = "\n".join(scene_context) if scene_context else "- (별도 지정된 장면 맥락 없음)"

    user_prompt = f"""
[상황 정보]
- scene_id: {scene_id}
- 대화 상대: {partner}
- 호감도 delta: {delta:.3f}

[해당 씬에서, 가장 비슷한 원본 선택지]
- 텍스트: {best_text}
- 이 선택지의 원래 점수(score): {best_score}

[방금 공주가 실제로 입력한 대사]
공주: {user_text}

지금부터는, 공주가 한 말 그다음부터 이 말을 들은 '{partner}'의 반응에 대한 대사를 시작하라.
두 사람은 번갈아가며 말하지만, 꼭 1:1로 교대로 하지 않아도 된다.
단, 전체 줄 수는 반드시 7줄이어야 한다.

[장면 맥락]
{context_str}

[인물 페르소나]
[공주 페르소나]
{princess_persona}

[{partner} 페르소나]
{partner_persona}

요청:
- 위 정보와 페르소나를 기반으로, 공주와 {partner}가 이어서 나눌 대사를 7줄 생성하라.
- 각 줄은 '공주 :' 또는 '{partner} :' 으로 시작해야 한다.
- 공주는 1인칭 내면 독백을 괄호 안에 쓴다. 예: 공주 : (나는 잠시 숨을 골랐다.)
- 총 줄 수는 정확히 7줄이다.
- {user_text}를 반복하지 말고, {partner}의 대사부터 생성한다.
- 한 줄에는 1~2문장 정도만 쓴다.
- 이미 플레이어가 말한 대사는 반복하지 말고, 그 다음 상황부터 이어서 쓴다.
""".strip()

    return system_prompt, user_prompt


def generate_dialogue_with_gemini(system_prompt: str, user_prompt: str) -> str:
    """
    Google Gemini API를 호출해 7줄 대사 생성.
    """
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system_prompt
    )

    # 모든 안전 카테고리 차단 해제
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    generation_config = genai.types.GenerationConfig(
        temperature=0.8,
        max_output_tokens=4000,
    )

    try:
        response = model.generate_content(
            user_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        if response.parts:
            return response.text.strip()
        else:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            return f"Error: Generation blocked. Finish Reason: {finish_reason}"

    except Exception as e:
        return f"Error generating dialogue: {str(e)}"


def generate_dialogue_response(user_text: str, scene_id: str):
    """
    사용자 입력에 대한 대화 생성 (호감도 계산 + 대사 생성)
    """
    # 1. 호감도 계산 (안전 필터 + 유사도 체크 포함)
    res = compute_affinity_delta(
        user_text,
        scene_id,
        top_k=5,
        temperature=0.4,
        min_sim_ratio=0.0,
    )

    # 2. 상태 확인: abuse 또는 low_similarity면 대사 생성 없이 바로 반환
    if not res or res.get("status") != "success":
        # status가 "abuse" 또는 "low_similarity"인 경우
        # 프론트엔드에서 처리할 수 있도록 그대로 반환
        return res

    delta = res["delta"]

    # 3. 성공한 경우에만 프롬프트 구성 및 대사 생성
    system_prompt, user_prompt = build_dialogue_prompt(
        user_text=user_text,
        scene_id=scene_id,
        delta=delta,
        result=res,
        persona_cache=persona_cache,
    )

    # 4. Gemini 호출
    dialogue = generate_dialogue_with_gemini(system_prompt, user_prompt)

    return {
        "status": "success",
        "delta": delta,
        "dialogue": dialogue,
        "matches": res.get("matches", [])
    }
