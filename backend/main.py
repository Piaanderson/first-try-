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
    seed_ratio: float = Form(0.04),
    tube_radius: float = Form(0.18),
    node_radius: float = Form(0.32),
):
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file")

    try:
        result = generate_xenonite(
            data,
            seed_ratio=seed_ratio,
            tube_radius=tube_radius,
            node_radius=node_radius,
        )
    except Exception as e:
        raise HTTPException(422, str(e))

    return Response(
        content=result,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=xenonite.stl"},
    )
