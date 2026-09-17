import requests
import logging
from typing import Any, Dict, Optional, List
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout

logger = logging.getLogger(__name__)

class ClearSpendingClient:
    """
    Professional client for the ClearSpending API.

    Provides a high-level interface for interacting with the contract search
    and retrieval endpoints of the ClearSpending service.
    """

    # Default base URL. This can be overridden in __init__.
    DEFAULT_BASE_URL = "https://newapi.clearspending.ru/csinternalapi/v1"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 30):
        """
        Initialize the API client.

        Args:
            api_key: API key for authentication.
            base_url: Custom base URL for the API. Defaults to DEFAULT_BASE_URL.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self.session = requests.Session()

    def _request_with_status(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Internal request handler that returns both the JSON response and the HTTP status code.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Initialize params and add mandatory format
        params = params or {}
        params['format'] = 'json'

        if self.api_key:
            params['apikey'] = self.api_key

        from requests import Request
        prepared_req = Request(method, url, params=params).prepare()
        logger.info(f"Requesting API: {prepared_req.url}")

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=self.timeout
            )
            if response.ok:
                return response.json(), response.status_code
            return None, response.status_code
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None, None

    def _request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Generic request handler for the API.
        Maintains backward compatibility by returning None on error.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}
        params['format'] = 'json'
        if self.api_key:
            params['apikey'] = self.api_key

        from requests import Request
        prepared_req = Request(method, url, params=params).prepare()
        logger.info(f"Requesting API: {prepared_req.url}")

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=self.timeout
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

    def search_contracts(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Search for contracts based on product search, region, date range, etc.

        Args:
            params: Dictionary of search parameters.
                    Expected keys: 'search', 'region_code', 'sign_date_gte',
                    'sign_date_lte', 'page', 'page_size', 'sort'.

        Returns:
            JSON response containing contract data if successful, None otherwise.
        """
        return self._request("GET", "filtered-contracts", params=params)

    def get_contract_details(self, reg_num: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a specific contract by its registration number.
        If request to contracts44 returns 404, retry with contracts223.

        Args:
            reg_num: The contract registration number (regNum).

        Returns:
            JSON response with contract details if successful, None otherwise.
        """
        # Try contracts44 first
        data, status = self._request_with_status("GET", f"contracts44/{reg_num}")

        if data:
            return data

        if status == 404:
            logger.info(f"Contract {reg_num} not found in contracts44 (404). Retrying with contracts223...")
            data, status = self._request_with_status("GET", f"contracts223/{reg_num}")
            return data

        return None

    def get_db_info(self) -> Optional[Dict[str, Any]]:
        """
        Get general information about the API database.

        Returns:
            JSON response with database info if successful, None otherwise.
        """
        params = {'info': 'all'}
        return self._request("GET", "dbinfo/", params=params)
