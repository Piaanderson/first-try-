from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware


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


from xenonite import generate_xenonite, RIDGE_HEIGHT, VALLEY_DEPTH, SEED_DENSITY

@app.post("/xenonite")
async def xenonite_endpoint(
    file: UploadFile = File(...),
    seed_density: float = Form(SEED_DENSITY),
    ridge_height: float = Form(RIDGE_HEIGHT),
    valley_depth: float = Form(VALLEY_DEPTH),
):
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file")

    try:
        result = generate_xenonite(
            data,
            seed_density=seed_density,
            ridge_height=ridge_height,
            valley_depth=valley_depth,
        )
    except Exception as e:
        raise HTTPException(422, str(e))

    return Response(
        content=result,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=xenonite.stl"},
    )
