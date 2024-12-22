from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from multilingualmc.server.model import Translator, Args, ModelMT, ModelRefine, Mode
from fastapi.middleware.cors import CORSMiddleware

# Define the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific domains if needed
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

args = Args()
print(args)
args.model_mt = ModelMT.seamless
# args.cache_dir = "/data/user_data/jiaruil5/.cache/"
# args.openai_key_path = "/home/jiaruil5/openai_key.txt"
# args.term_path = "/home/jiaruil5/multilingual/multilingual-model-card/src/dictionary_collection/growing_dict/mturk.json"

# src_lang = "English"
# tgt_lang = "Chinese"


translator = Translator(args)

# Define the request body structure
class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str 
    mode: str

# Define the response structure
class TranslationResponse(BaseModel):
    translated_text: str

# API endpoint for translation
@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    try:
        # Perform the translation
        translated_text = translator.translate(request.text, request.src_lang, request.tgt_lang, request.mode)
        return TranslationResponse(translated_text=translated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)