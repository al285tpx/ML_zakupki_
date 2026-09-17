import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
from django.test import RequestFactory, TestCase
from loader.api_client import ClearSpendingClient
from loader.services import ContractService
from loader.analytics import AnalyticsService
from loader.views import index
import requests

class TestClearSpendingClient(unittest.TestCase):
    def setUp(self):
        self.client = ClearSpendingClient(api_key="test_key")

    @patch('requests.Session.request')
    def test_search_contracts_success(self, mock_request):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': [{'regnum': '123'}]}
        mock_request.return_value = mock_response

        result = self.client.search_contracts({'search': 'test'})

        self.assertEqual(result['data'][0]['regnum'], '123')
        # Verify API key was added
        args, kwargs = mock_request.call_args
        self.assertEqual(kwargs['params']['apikey'], 'test_key')

    @patch('requests.Session.request')
    def test_search_contracts_http_error(self, mock_request):
        # Mock 404 error using actual requests exception
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_request.return_value = mock_response

        result = self.client.search_contracts({'search': 'test'})
        self.assertIsNone(result)

class TestContractService(unittest.TestCase):
    def setUp(self):
        self.mock_api = MagicMock(spec=ClearSpendingClient)
        self.service = ContractService(self.mock_api)

    def test_process_contracts_df(self):
        # Sample raw data from API
        raw_data = pd.DataFrame([{
            'regnum': 'REG1',
            'regionCode': '77',
            'signDate': '2023-01-01T00:00:00',
            'products': [{'name': 'Бумага офисная А4', 'price': 100, 'quantity': 10, 'OKEI': {'name': 'лист', 'code': '1'}, 'sum': 1000, 'OKPD2': {'code': '123', 'name': 'Paper'}}],
            'suppliers': [{'organizationName': 'Supp1', 'inn': '123', 'factualAddress': 'Addr1'}],
            'customer': {'fullName': 'Cust1', 'inn': '456', 'postalAddress': 'Addr2'}
        }])

        processed = self.service._process_contracts_df(raw_data, 'Бумага')

        self.assertEqual(len(processed), 1)
        self.assertEqual(processed.iloc[0]['product_name'], 'Бумага офисная А4')
        self.assertEqual(processed.iloc[0]['contract'], 'REG1')

    def test_fetch_contracts_pagination_limit(self):
        # Mock API to return data for 2 pages then stop
        # Each page must have 50 records to continue pagination
        page1 = [{'regnum': str(i), 'signDate': '2023-01-01T00:00:00', 'products': [{'name': 'test'}]} for i in range(50)]
        page2 = [{'regnum': str(i), 'signDate': '2023-01-02T00:00:00', 'products': [{'name': 'test'}]} for i in range(50, 100)]

        self.mock_api.search_contracts.side_effect = [
            {'data': page1},
            {'data': page2},
            None # stop pagination
        ]



        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 10)

        df = self.service.fetch_contracts(
            product_search='test', product_attribute=None, region_code='77',
            date_start=start, date_end=end, fz=None, price_min=None, price_max=None
        )

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(self.mock_api.search_contracts.call_count, 3)

class TestAnalyticsService(unittest.TestCase):
    def setUp(self):
        self.service = AnalyticsService()

    def test_classify_okpd2_quartiles(self):
        # Sample data with realistic product names to avoid empty vocabulary error
        data = {
            'product_name': ['Бумага белая А4', 'Бумага белая А4', 'Бумага белая А4', 'Бумага белая А4', 'Бумага цветная'],
            'product_price': [10, 20, 30, 40, 50],
            'product_ed_izm': ['pcs', 'pcs', 'pcs', 'pcs', 'pcs'],
            'OKPD2_code_res': ['1', '1', '1', '1', '1'],
            'OKPD2_name_res': ['Бумага', 'Бумага', 'Бумага', 'Бумага', 'Бумага'],
        }
        df = pd.DataFrame(data)

        # Mock missing list
        list_nan2 = []

        result = self.service.classify_okpd2(df, list_nan2, 'Бумага')
        self.assertIsNotNone(result)
        self.assertIn('Quartile', result.columns)

class ContractViewIntegrationTest(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_index_get(self):
        request = self.factory.get('/')
        response = index(request)
        self.assertEqual(response.status_code, 200)

    @patch('loader.views.ContractService.fetch_contracts')
    @patch('loader.views.AnalyticsService.analyze_okpd2')
    @patch('loader.views.AnalyticsService.classify_okpd2')
    def test_index_post_success(self, mock_classify, mock_analyze, mock_fetch):
        # Setup mocks
        mock_df = pd.DataFrame({
            'product_name': ['Test'], 'product_price': [100], 'product_ed_izm': ['pcs'],
            'OKPD2_code': ['1'], 'OKPD2_name': ['Name'], 'signDate': ['2023-01-01']
        })
        mock_fetch.return_value = mock_df
        mock_analyze.return_value = (mock_df, [])
        mock_classify.return_value = mock_df

        # Create POST request
        request = self.factory.post('/', {
            'product_search': 'test',
            'product_attribute': '',
            'kod_regiona': '77',
            'date_start': '2023-01-01',
            'date_end': '2023-01-31',
            'fz': 'All',
            'price_min': '',
            'price_max': ''
        })

        response = index(request)
        self.assertEqual(response.status_code, 200)
