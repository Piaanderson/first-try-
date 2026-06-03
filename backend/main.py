from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from xenonite import generate_xenonite

app = FastAPI(title="Effect Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/xenonite")
async def xenonite_endpoint(
    file: UploadFile = File(...),
    seed_density: float = Form(0.0012),
    wire_width: float = Form(0.22),
):
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file")

    try:
        result = generate_xenonite(
            data,
            seed_density=seed_density,
            wire_width=wire_width,
        )
    except Exception as e:
        raise HTTPException(422, str(e))

    return Response(
        content=result,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=xenonite.stl"},
    )
