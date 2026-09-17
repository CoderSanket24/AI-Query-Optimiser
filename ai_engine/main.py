from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import queryOptimizer, feedback, anomaly_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(queryOptimizer.router)
app.include_router(feedback.router)
app.include_router(anomaly_router.router)


@app.get("/")
async def health():
    return {"message": "server is running successfully."}