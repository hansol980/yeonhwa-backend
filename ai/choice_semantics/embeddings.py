import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------
# 1. 모델 로드
# -----------------------------
MODEL_NAME = "jhgan/ko-sroberta-multitask"
model = SentenceTransformer(MODEL_NAME)

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
# def build_combined_text(choice):
    # text = choice["text"]
    # tags = ", ".join(choice.get("tags", []))
    # emotion = ", ".join(choice.get("emotion", []))
    # combined = (
    #     f"{text} "
    #     f"[tags: {tags}] "
    #     f"[emotion: {emotion}]"
    # )
    # return combined
def build_combined_text(choice):
    base = choice.get("embed_text") or choice["text"]
    text = choice["text"]
    tags = ", ".join(choice.get("tags", []))
    emotion = ", ".join(choice.get("emotion", []))

    # return f"{base} [tags: {tags}] [emotion: {emotion}]"
    combined = f"{text} {base} [tags: {tags}] [emotion: {emotion}]"
    return combined



# -----------------------------
# 4. 모든 선택지 임베딩 생성
# -----------------------------
print("\n🔍 Generating embeddings...")

# for scene in data:
#     scene_id = scene["scene_id"]
#     character = scene["character"]
#     chapter = scene["chapter"]
#     step = scene["step"]

#     for choice in scene["choices"]:
#         combined_text = build_combined_text(choice)

#         vector = model.encode(combined_text)
#         vector = vector.astype("float32")

#         all_vectors.append(vector)
#         id_map.append({
#             "scene_id": scene_id,
#             "character": character,
#             "chapter": chapter,
#             "step": step,
#             "choice_id": choice["id"],
#            "tags": choice.get("tags", []),        
#            "emotion": choice.get("emotion", [])  
#             "text": choice["text"],
#             "score": choice["score"]
#         })

# print(f"✔ Total choices embedded: {len(all_vectors)}")

# all_vectors = np.vstack(all_vectors)

combined_texts = []
for scene in data:
    scene_id = scene["scene_id"]
    character = scene["character"]
    chapter = scene["chapter"]
    step = scene["step"]

    for choice in scene["choices"]:
        combined_text = build_combined_text(choice)
        combined_texts.append(combined_text)

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

# 🔑 한 번에 인코딩 + 정규화 (코사인 유사도용)
embeddings = model.encode(
    combined_texts,
    normalize_embeddings=True  # ★ 코사인 유사도용 정규화
).astype("float32")

all_vectors = embeddings

print(f"✔ Total choices embedded: {len(all_vectors)}")

# -----------------------------
# 5. FAISS Index 생성 및 저장
# -----------------------------
d = all_vectors.shape[1]  # vector dimension

# index = faiss.IndexFlatL2(d)
index = faiss.IndexFlatIP(d)
index.add(all_vectors)

print("✔ FAISS index built")

# 저장
faiss.write_index(index, "./choice_index.faiss")

with open("./choice_id_map.json", "w", encoding="utf-8") as f:
    json.dump(id_map, f, ensure_ascii=False, indent=2)

print("🎉 All Done! Embeddings + Index Saved.")
