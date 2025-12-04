import os
import json
from collections import defaultdict

PERSONA_CATEGORIES = {
    "profile",
    "personality",
    "manner_out",
    "manner_in",
    "inner_rule",
    "dialogue_rule",
}

def load_persona_cache(persona_dir: str = "./json_list/persona") -> dict:
    """
    persona_dir 안의 모든 .json 파일을 읽어서
    캐릭터별 페르소나 텍스트를 하나의 문자열로 합쳐 반환.

    반환 형식:
      {
        "화야진": "[profile]\\n...\\n\\n[personality]\\n...\\n\\n[dialogue_rule]\\n...",
        "진효": "...",
        ...
      }
    """
    by_char: dict[str, list[str]] = defaultdict(list)

    for filename in os.listdir(persona_dir):
        if not filename.lower().endswith(".json"):
            continue

        file_path = os.path.join(persona_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[페르소나 로드 에러] {file_path}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[경고] {filename} 최상단이 list가 아님. 건너뜀.")
            continue

        for item in data:
            if not isinstance(item, dict):
                continue

            if item.get("type") != "character_profile":
                continue

            character = item.get("character")
            category = item.get("category")
            content = (item.get("content") or "").strip()

            if not character or not content:
                continue

            if category not in PERSONA_CATEGORIES:
                continue

            by_char[character].append(f"[{category}]\n{content}")

    persona_cache = {c: "\n\n".join(chunks) for c, chunks in by_char.items()}
    print(f"[정보] 페르소나 캐시 로드 완료: {len(persona_cache)}명")
    return persona_cache