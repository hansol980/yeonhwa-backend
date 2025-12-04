import json
import os
from tqdm import tqdm
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------------
# 1. 환경 설정 및 모델 준비
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

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
        return [0.0] * 768 

# -----------------------------
# 2. JSON 로드
# -----------------------------
JSON_PATH = "./choices.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

all_choices = []   # 모든 선택지 데이터
all_vectors = []   # 임베딩 벡터 저장
id_map = []        # (scene_id, choice_id) 저장용


# -----------------------------
# 3. 임베딩용 문장 생성 함수
# -----------------------------
def build_combined_text(choice):
    base = choice.get("embed_text") or choice["text"]
    text = choice["text"]
    tags = ", ".join(choice.get("tags", []))
    emotion = ", ".join(choice.get("emotion", []))

    combined = f"{text} {base} [tags: {tags}] [emotion: {emotion}]"
    return combined


# -----------------------------
# 4. 모든 선택지 임베딩 생성
# -----------------------------
print("\n🔍 Generating embeddings with Gemini API...")

combined_texts = []
# API 호출 비용/속도 고려하여 배치 처리가 좋지만, 여기서는 단순하게 순차 처리하거나
# genai.embed_content가 배치를 지원하는지 확인 필요. 
# 현재 SDK는 단일 호출 위주이므로 루프 돌림.

for scene in tqdm(data, desc="Processing Scenes"):
    scene_id = scene["scene_id"]
    character = scene["character"]
    chapter = scene["chapter"]
    step = scene["step"]

    for choice in scene["choices"]:
        combined_text = build_combined_text(choice)
        
        # API 호출
        vector = get_embedding(combined_text)
        
        # 리스트를 numpy 배열로 변환
        vector_np = np.array(vector, dtype="float32")

        all_vectors.append(vector_np)
        
        id_map.append({
            "scene_id": scene_id,
            "character": character,
            "chapter": chapter,
            "step": step,
            "choice_id": choice["id"],
            "text": choice["text"],
            "score": choice["score"],
            "tags": choice.get("tags", []),        
            "emotion": choice.get("emotion", [])  
        })

print(f"✔ Total choices embedded: {len(all_vectors)}")

if not all_vectors:
    print("❌ No vectors generated.")
    exit(1)

# 리스트 -> 2D 배열 변환
all_vectors = np.vstack(all_vectors)

# -----------------------------
# 5. FAISS Index 생성 및 저장
# -----------------------------
d = all_vectors.shape[1]  # vector dimension (Gemini usually 768)
print(f"Vector dimension: {d}")

# Gemini 임베딩은 이미 정규화되어 있을 수 있으나, 코사인 유사도를 위해 Inner Product(IP) 사용 시 정규화 확인 필요.
# faiss.normalize_L2(all_vectors) # 필요시 주석 해제

index = faiss.IndexFlatIP(d)
index.add(all_vectors)

print("✔ FAISS index built")

# 저장
faiss.write_index(index, "./choice_index.faiss")

with open("./choice_id_map.json", "w", encoding="utf-8") as f:
    json.dump(id_map, f, ensure_ascii=False, indent=2)

print("🎉 All Done! Embeddings + Index Saved.")
