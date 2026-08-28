from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import queryOptimizer

app = FastAPI()

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(queryOptimizer.router)

@app.get("/")
async def health():
    return {"message":"server is running successfully."}