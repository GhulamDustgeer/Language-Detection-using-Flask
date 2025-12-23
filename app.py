from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from utils.openai_client import detect_language_openai

app = FastAPI(title="Language Detector API (OpenAI)")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Language Detector (OpenAI)"}

@app.post("/detect", response_class=HTMLResponse)
async def detect_html(request: Request, text: str = Form(...)):
    if not text.strip():
        result = "Please enter some text"
    else:
        result = detect_language_openai(text)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "input_text": text},
    )

# Optional JSON endpoint (useful for demos/tests)
@app.post("/detect.json")
async def detect_json(text: str = Form(...)):
    if not text.strip():
        return JSONResponse({"language": "Unknown", "error": "Empty input"}, status_code=400)
    return {"language": detect_language_openai(text)}
