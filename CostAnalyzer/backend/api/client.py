import httpx
import logging
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

class ClearSpendingClient:
    """
    Asynchronous client for the ClearSpending API.
    """

    DEFAULT_BASE_URL = "https://newapi.clearspending.ru/csinternalapi/v1"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        # We use an AsyncClient for all requests.
        # In a real FastAPI app, this would be managed via a lifespan event or a dependency.
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        """Close the underlying HTTP client."""
        await self.client.aclose()

    async def _request_with_status(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Internal request handler that returns both the JSON response and the HTTP status code.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}
        params['format'] = 'json'
        if self.api_key:
            params['apikey'] = self.api_key

        logger.info(f"Requesting API: {method} {url} with params {params}")

        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=data
            )
            if response.is_success:
                return response.json(), response.status_code
            return None, response.status_code
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None, None

    async def _request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Generic request handler for the API.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}
        params['format'] = 'json'
        if self.api_key:
            params['apikey'] = self.api_key

        logger.info(f"Requesting API: {method} {url} with params {params}")

        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=data
            )
            response.raise_for_status()
            json_data = response.json()

            if isinstance(json_data, dict):
                regnum = json_data.get('regnum')
                if regnum:
                    logger.info(f"API Response contains regnum: {regnum}")
                elif 'contracts' in json_data and 'data' in json_data['contracts']:
                    data = json_data['contracts']['data']
                    if data and isinstance(data, list) and isinstance(data[0], dict):
                        first_reg = data[0].get('regnum') or data[0].get('regNum')
                        if first_reg:
                            logger.info(f"API Search Response found {len(data)} contracts. First regnum: {first_reg}")

            return json_data
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    async def search_contracts(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Search for contracts based on product search, region, date range, etc.
        """
        return await self._request("GET", "filtered-contracts", params=params)

    async def get_contract_details(self, reg_num: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a specific contract by its registration number.
        If request to contracts44 returns 404, retry with contracts223.
        """
        # Try contracts44 first
        data, status = await self._request_with_status("GET", f"contracts44/{reg_num}")

        if data:
            return data

        if status == 404:
            logger.info(f"Contract {reg_num} not found in contracts44 (404). Retrying with contracts223...")
            data, status = await self._request_with_status("GET", f"contracts223/{reg_num}")
            return data

        return None

    async def get_db_info(self) -> Optional[Dict[str, Any]]:
        """
        Get general information about the API database.
        """
        params = {'info': 'all'}
        return await self._request("GET", "dbinfo/", params=params)
