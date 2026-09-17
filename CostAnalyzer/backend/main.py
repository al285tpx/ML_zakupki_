import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult

from .api.client import ClearSpendingClient
from .schemas.search import SearchRequest, TaskStatus, AnalysisResponse
from .worker import celery_app

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Client lifecycle management
class APIClientManager:
    def __init__(self):
        self.client: ClearSpendingClient = None

    async def init_client(self):
        api_key = os.getenv("CLEARSPENDING_API_KEY")
        self.client = ClearSpendingClient(api_key=api_key)
        logger.info("ClearSpendingClient initialized.")

    async def close_client(self):
        if self.client:
            await self.client.close()
            logger.info("ClearSpendingClient closed.")

client_manager = APIClientManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await client_manager.init_client()
    yield
    # Shutdown
    await client_manager.close_client()

app = FastAPI(
    title="ClearSpending Analysis API",
    description="FastAPI backend for government contract analysis",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "ClearSpending Analysis API is running."}

@app.post("/api/search", response_model=AnalysisResponse)
async def start_search(request: SearchRequest):
    """
    Initiates a contract search and analysis task.
    Returns a task_id to poll for results.
    """
    try:
        # Convert Pydantic model to dict for Celery serialization
        search_params = request.model_dump()
        # Convert datetimes to ISO strings
        search_params['date_start'] = request.date_start.isoformat()
        search_params['date_end'] = request.date_end.isoformat()

        # Trigger Celery task
        task = celery_app.send_task("tasks.analyze_contracts", args=[search_params])

        return AnalysisResponse(task_id=task.id, status="PENDING")
    except Exception as e:
        logger.error(f"Error starting search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Polls the status of an analysis task.
    """
    res = AsyncResult(task_id, app=celery_app)

    status_map = {
        "PENDING": "PENDING",
        "STARTED": "PROCESSING",
        "SUCCESS": "COMPLETED",
        "FAILURE": "FAILED",
        "RETRY": "PROCESSING"
    }

    current_status = status_map.get(res.status, "PENDING")

    return TaskStatus(
        task_id=task_id,
        status=current_status,
        result_url=None if current_status != "COMPLETED" else f"/api/results/{task_id}",
        error=str(res.result) if res.status == "FAILURE" else None
    )

@app.get("/api/results/{task_id}")
async def get_task_results(task_id: str):
    """
    Returns the final analyzed data for a completed task.
    """
    res = AsyncResult(task_id, app=celery_app)

    if res.status != "SUCCESS":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed. Current status: {res.status}"
        )

    # The result is the return value of analyze_contracts_task
    result_data = res.result

    if isinstance(result_data, dict) and result_data.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result_data.get("error", "Unknown error"))

    return result_data.get("data", [])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
