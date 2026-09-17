import os
import logging
from celery import Celery
from .api.client import ClearSpendingClient
from .services.contract_service import ContractService
from .services.analytics_service import AnalyticsService
from .schemas.search import SearchRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# API Key from env
API_KEY = os.getenv("CLEARSPENDING_API_KEY")
OKPD2_SPRAV_PATH = os.getenv("OKPD2_SPRAV_PATH", "data/okpd2_sprav.xlsx")

@celery_app.task(bind=True, name="tasks.analyze_contracts")
def analyze_contracts_task(self, search_params: dict):
    """
    Celery task to perform the full analysis pipeline:
    Fetch -> Detail -> Process -> Analyze -> Classify -> Quartiles
    """
    try:
        # 1. Setup Clients and Services
        # Since this runs in a separate process, we create a new client
        import asyncio

        async def run_pipeline():
            client = ClearSpendingClient(api_key=API_KEY)
            contract_service = ContractService(client)
            analytics_service = AnalyticsService()

            # Parse parameters
            # Use a local dummy for date_start/end as they come as strings from JSON
            from datetime import datetime
            date_start = datetime.fromisoformat(search_params['date_start'])
            date_end = datetime.fromisoformat(search_params['date_end'])

            # Step 1: Fetch and Process Contracts
            df, regnums = await contract_service.fetch_contracts(
                product_search=search_params['product_search'],
                product_attribute=search_params.get('product_attribute'),
                region_code=search_params['region_code'],
                date_start=date_start,
                date_end=date_end,
                fz=search_params.get('fz'),
                price_min=search_params.get('price_min'),
                price_max=search_params.get('price_max'),
                okdp=search_params.get('okdp')
            )

            if df is None or df.empty:
                return {"status": "completed", "data": [], "message": "No contracts found"}

            # Step 2: Analyze OKPD2
            df_analyzed, list_nan2 = analytics_service.analyze_okpd2(
                df=df,
                product_search=search_params['product_search'],
                okpd2_sprav_path=OKPD2_SPRAV_PATH
            )

            # Step 3: Classify remaining OKPD2 using ML
            df_classified = analytics_service.classify_okpd2(
                df=df_analyzed,
                list_nan2=list_nan2,
                product_search=search_params['product_search']
            )

            # Convert DataFrame to JSON records
            results = df_classified.to_dict(orient='records')

            await client.close()
            return {"status": "completed", "data": results}

        # Run the async pipeline in the synchronous Celery task
        result = asyncio.run(run_pipeline())
        return result

    except Exception as e:
        logger.exception(f"Task failed: {e}")
        return {"status": "failed", "error": str(e)}
