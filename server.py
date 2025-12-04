from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Add ai/choice_semantics to Python path
sys.path.insert(0, os.path.join(current_dir, 'ai/choice_semantics'))
# Add ai to Python path for dialogue_generator_gemini
sys.path.insert(0, os.path.join(current_dir, 'ai'))

from score import compute_affinity_delta
from dialogue_generator_gemini import generate_dialogue_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScoreRequest(BaseModel):
    user_text: str
    scene_id: str

@app.post("/calculate_score")
async def calculate_score(request: ScoreRequest):
    try:
        result = compute_affinity_delta(
            request.user_text,
            request.scene_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_input")
async def process_input(request: ScoreRequest):
    try:
        result = generate_dialogue_response(
            request.user_text,
            request.scene_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
