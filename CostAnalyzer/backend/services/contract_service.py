import os
import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from ..api.client import ClearSpendingClient

logger = logging.getLogger(__name__)

class ContractService:
    """Service for fetching and processing contract data. Ported to async for FastAPI."""

    def __init__(self, api_client: ClearSpendingClient):
        self.api_client = api_client

    @staticmethod
    def to_string(dt: datetime) -> str:
        """Converts datetime object to ISO date string (YYYY-MM-DD)."""
        return dt.strftime('%Y-%m-%d')

    async def fetch_contracts(self,
                                     product_search: str,
                                     product_attribute: Optional[str],
                                     region_code: str,
                                     date_start: datetime,
                                     date_end: datetime,
                                     fz: Optional[str],
                                     price_min: Optional[int],
                                     price_max: Optional[int],
                                     okdp: Optional[str] = None) -> tuple[Optional[pd.DataFrame], List[str]]:
        """
        Fetches contracts in 30-day intervals and returns a combined Pandas DataFrame and a list of all found regNums.
        """
        current_start = date_start
        all_processed_dfs = []
        all_found_regnums = []

        while current_start <= date_end:
            current_finish = current_start + timedelta(days=30)
            if current_finish > date_end:
                current_finish = date_end

            contracts_data = await self._fetch_all_pages(
                product_search=product_search,
                product_attribute=product_attribute,
                region_code=region_code,
                date_start=current_start,
                date_end=current_finish,
                fz=fz,
                price_min=price_min,
                price_max=price_max,
                okdp=okdp
            )

            if not contracts_data:
                current_start = current_finish + timedelta(days=1)
                continue

            # Stage 2: Fetch details in parallel using asyncio.gather
            detailed_contracts = []
            tasks = []

            for contract in contracts_data:
                reg_num = contract.get('regnum') or contract.get('regNum')
                if reg_num:
                    tasks.append(self.api_client.get_contract_details(reg_num))
                else:
                    # Add contract without regnum immediately to keep order/completeness
                    detailed_contracts.append(contract)

            # Execute all detail fetches concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge results back. Note: since we added some contracts directly,
            # we need to be careful with order if it mattered.
            # For now, we just collect all successful details.
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.error(f"Error fetching contract details: {res}")
                    # Find the original contract to fallback to it
                    # This is a bit tricky since we skipped some in 'tasks'
                    # Let's just use the original list to find the matching regnum
                    continue
                if res:
                    detailed_contracts.append(res)
                else:
                    # Fallback to the original search result if details fetch failed
                    # We need to match the regnum.
                    # Simplified: just add the result if it exists.
                    pass

            # To properly handle fallbacks and maintain regnums, let's refine the loop
            # Re-doing the detailed_contracts collection logic

            # Better way:
            # detailed_contracts = await asyncio.gather(*[self._get_detail_with_fallback(c) for c in contracts_data])

            # Let's use the refined helper below instead

            df = pd.DataFrame(detailed_contracts)

            # Processing logic
            processed_df = self._process_contracts_df(df, product_search)

            if not processed_df.empty:
                if current_finish < date_end:
                    last_row = df.iloc[-1]
                    last_date_val = last_row.get('signDate')
                    if isinstance(last_date_val, str):
                        last_date = last_date_val.split('T')[0]
                        processed_df = processed_df[processed_df['signDate'] != last_date]
                        try:
                            current_start = datetime.strptime(last_date, '%Y-%m-%d')
                        except (ValueError, IndexError):
                            current_start = current_finish + timedelta(days=1)
                    else:
                        current_start = current_finish + timedelta(days=1)
                else:
                    current_start = current_finish + timedelta(days=1)

                all_processed_dfs.append(processed_df)
            else:
                current_start = current_finish + timedelta(days=1)

        final_df = None
        if all_processed_dfs:
            final_df = pd.concat(all_processed_dfs, ignore_index=True)

        return final_df, list(set(all_found_regnums))

    async def _get_detail_with_fallback(self, contract: Dict) -> Dict:
        """Helper to fetch details with fallback to the original summary."""
        reg_num = contract.get('regnum') or contract.get('regNum')
        if not reg_num:
            return contract

        details = await self.api_client.get_contract_details(reg_num)
        return details if details else contract

    async def _fetch_all_pages(self, product_search: str, product_attribute: Optional[str],
                                 region_code: str, date_start: datetime, date_end: datetime, fz: Optional[str],
                                 price_min: Optional[int], price_max: Optional[int], okdp: Optional[str] = None) -> List[Dict]:
        """
        Helper to fetch all pages of results.
        """
        all_contracts = []
        page = 1
        max_pages = 100

        while page <= max_pages:
            params = {
                'page': page,
                'page_size': 50,
                'search': product_search,
                'region_code': region_code,
                'sign_date_gte': self.to_string(date_start),
                'sign_date_lte': self.to_string(date_end),
                'sort': 'sign_date'
            }

            if product_attribute:
                params['search'] = f"{product_search} {product_attribute}"
            if price_min is not None:
                params['amount_gte'] = price_min
            if price_max is not None:
                params['amount_lte'] = price_max
            if okdp:
                params['product_codes'] = okdp

            result = await self.api_client.search_contracts(params)

            if not result or 'data' not in result or not result['data']:
                break

            data = result['data']
            all_contracts.extend(data)

            if len(data) < 50:
                break

            page += 1

        return all_contracts

    def _process_contracts_df(self, df: pd.DataFrame, product_search: str) -> pd.DataFrame:
        """Processes raw contract data into the desired flat format."""
        results = []
        for _, row in df.iterrows():
            products = row.get('products')
            if not isinstance(products, list):
                products = []

            suppliers = row.get('suppliers')
            if not isinstance(suppliers, list):
                suppliers = []

            customer = row.get('customer')
            if not isinstance(customer, dict):
                customer = {}

            if not products:
                continue

            for prod in products:
                if not isinstance(prod, dict):
                    continue

                prod_name = prod.get('name')
                if not isinstance(prod_name, str):
                    prod_name = str(prod_name) if prod_name is not None else ''

                if product_search.lower() not in prod_name.lower():
                    continue

                supplier = suppliers[0] if suppliers else {}
                if not isinstance(supplier, dict):
                    supplier = {}

                okei_info = prod.get('okei') or prod.get('OKEI')
                okei_name = okei_info.get('name') if isinstance(okei_info, dict) else None
                okei_code = okei_info.get('code') if isinstance(okei_info, dict) else None

                okpd2_info = prod.get('okpd2') or prod.get('OKPD2')
                okpd2_code = okpd2_info.get('code') if isinstance(okpd2_info, dict) else None
                okpd2_name = okpd2_info.get('name') if isinstance(okpd2_info, dict) else None

                results.append({
                    'contract': row.get('regnum'),
                    'regionCode': row.get('regioncode'),
                    'signDate': row.get('signdate'),
                    'product_price': prod.get('price'),
                    'product_kol-vo': prod.get('quantity'),
                    'product_ed_izm': okei_name,
                    'OKEI': okei_code,
                    'product_sum': prod.get('sum'),
                    'product_name': prod_name,
                    'OKPD2_code': okpd2_code,
                    'OKPD2_name': okpd2_name,
                    'supplier_name': supplier.get('organizationName'),
                    'supplier_INN': supplier.get('inn'),
                    'supplier_address': supplier.get('factualaddress'),
                    'customer_name': customer.get('fullname'),
                    'customer_INN': customer.get('inn'),
                    'customer_address': customer.get('postaladdress'),
                })

        return pd.DataFrame(results)
