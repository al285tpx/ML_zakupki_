import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .api_client import ClearSpendingClient

logger = logging.getLogger(__name__)

class ContractService:
    """Service for fetching and processing contract data."""

    def __init__(self, api_client: ClearSpendingClient):
        self.api_client = api_client

    @staticmethod
    def to_string(dt: datetime) -> str:
        """Converts datetime object to ISO date string (YYYY-MM-DD)."""
        return dt.strftime('%Y-%m-%d')

    def fetch_contracts(self,
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
        print("\n!!! [FORCED DEBUG] STARTING fetch_contracts !!!")
        current_start = date_start
        all_processed_dfs = []
        all_found_regnums = []

        while current_start <= date_end:
            current_finish = current_start + timedelta(days=30)
            if current_finish > date_end:
                current_finish = date_end

            print(f"!!! [FORCED DEBUG] Window: {current_start} to {current_finish}")
            contracts_data = self._fetch_all_pages(
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
                print(f"!!! [FORCED DEBUG] No contracts found for window {current_start} to {current_finish}")
                current_start = current_finish + timedelta(days=1)
                continue

            print(f"!!! [FORCED DEBUG] Stage 1 complete. Found {len(contracts_data)} contracts. Starting Stage 2")

            if contracts_data:
                first_item = contracts_data[0]
                print(f"!!! [FORCED DEBUG] First contract type: {type(first_item)}")
                if isinstance(first_item, dict):
                    print(f"!!! [FORCED DEBUG] First contract keys: {list(first_item.keys())}")
                    print(f"!!! [FORCED DEBUG] First contract regnum value: {first_item.get('regnum')} / {first_item.get('regNum')}")

            detailed_contracts = []
            # Use ThreadPoolExecutor to fetch contract details in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Map futures to the original contract to handle results in order or identify failures
                future_to_contract = {
                    executor.submit(self.api_client.get_contract_details,
                                   (contract.get('regnum') or contract.get('regNum'))): contract
                    for contract in contracts_data if (contract.get('regnum') or contract.get('regNum'))
                }

                # For contracts without regnum, add them directly
                for contract in contracts_data:
                    if not (contract.get('regnum') or contract.get('regNum')):
                        detailed_contracts.append(contract)

                for future in as_completed(future_to_contract):
                    contract = future_to_contract[future]
                    try:
                        details = future.result()
                        if details:
                            detailed_contracts.append(details)
                        else:
                            detailed_contracts.append(contract)
                    except Exception as e:
                        logger.error(f"Error fetching details for contract {contract.get('regnum')}: {e}")
                        detailed_contracts.append(contract)

            # Update all_found_regnums
            for contract in contracts_data:
                reg_num = contract.get('regnum') or contract.get('regNum')
                if reg_num:
                    all_found_regnums.append(reg_num)

            df = pd.DataFrame(detailed_contracts)

            # Processing logic
            processed_df = self._process_contracts_df(df, product_search)

            if not processed_df.empty:
                # To avoid overlapping, remove last date if not at the end
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

        print(f"!!! [FORCED DEBUG] fetch_contracts finished. Total regnums found: {len(all_found_regnums)} !!!")
        return final_df, list(set(all_found_regnums))

    def _fetch_all_pages(self, product_search: str, product_attribute: Optional[str],
                         region_code: str, date_start: datetime, date_end: datetime, fz: Optional[str],
                         price_min: Optional[int], price_max: Optional[int], okdp: Optional[str] = None) -> List[Dict]:
        """
        Helper to fetch all pages of results.
        Iterates through pages by incrementing the 'page' parameter in a loop.
        """
        all_contracts = []
        page = 1
        max_pages = 100  # Safety limit to prevent infinite loops

        while page <= max_pages:
            # Construct parameters for the current page
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
            # Temporary disable fz filter to match working browser request
            # if fz and fz != 'All':
            #     params['fz'] = fz
            if price_min is not None:
                params['amount_gte'] = price_min
            if price_max is not None:
                params['amount_lte'] = price_max
            if okdp:
                params['product_codes'] = okdp

            logger.info(f"Fetching page {page} for window {self.to_string(date_start)} to {self.to_string(date_end)}")

            result = self.api_client.search_contracts(params)

            # If no result or no data, we've reached the end of the available pages
            if not result or 'data' not in result or not result['data']:
                logger.info(f"No more data found at page {page}. Ending pagination.")
                break

            data = result['data']
            num_records = len(data)
            logger.info(f"Successfully retrieved {num_records} records from page {page}")

            all_contracts.extend(data)

            # If we received fewer records than the page size, it's the last page
            if num_records < 50:
                logger.info(f"Page {page} was the last page (received {num_records}/50).")
                break

            # Increment page for the next iteration
            page += 1

        if page > max_pages:
            logger.warning(f"Reached max_pages limit ({max_pages}). Some data might have been missed.")

        return all_contracts

    def _process_contracts_df(self, df: pd.DataFrame, product_search: str) -> pd.DataFrame:
        """Processes raw contract data into the desired flat format."""
        results = []
        initial_count = len(df)
        logger.info(f"Processing {initial_count} contracts from API...")

        for _, row in df.iterrows():
            # Use .get() and provide defaults, then verify types to avoid 'float' object is not iterable
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

                # Extract OKEI (unit of measure) info - check both cases
                okei_info = prod.get('okei') or prod.get('OKEI')
                okei_name = okei_info.get('name') if isinstance(okei_info, dict) else None
                okei_code = okei_info.get('code') if isinstance(okei_info, dict) else None

                # Extract OKPD2 info - check both cases
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

        final_df = pd.DataFrame(results)
        logger.info(f"Processed {len(final_df)} items after filtering by search term '{product_search}'")
        return final_df
