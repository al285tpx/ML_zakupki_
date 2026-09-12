from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.paginator import Paginator
from .forms import UserForm
import os
import pandas as pd
from .api_client import ClearSpendingClient
from .services import ContractService
from .analytics import AnalyticsService

# Base directory for saving files
BASE_DIR = os.getcwd()

def index(request):
    # Support both POST and GET for search to enable pagination
    if request.method == "POST":
        form = UserForm(request.POST)
    elif request.method == "GET" and any(k in request.GET for k in ['product_search', 'kod_regiona']):
        form = UserForm(request.GET)
    else:
        form = UserForm()

    if form.is_valid():
        # Extract cleaned data
        data = form.cleaned_data
        product_search = data.get("product_search")
        product_attribute = data.get("product_attribute")
        kod_regiona = data.get("kod_regiona")
        date_start = data.get("date_start")
        date_end = data.get("date_end")
        fz = data.get("fz")
        price_min = data.get("price_min")
        price_max = data.get("price_max")
        okdp = data.get("okdp")
        # Get page number from request.GET instead of form.cleaned_data
        page_number = request.GET.get("page", 1)
        try:
            page_number = int(page_number)
        except (ValueError, TypeError):
            page_number = 1

        # Initialize services
        api_key = getattr(settings, 'CLEARSPENDING_API_KEY', None)
        api_client = ClearSpendingClient(api_key=api_key)
        contract_service = ContractService(api_client)
        analytics_service = AnalyticsService()

        # 1. Fetch contracts
        df, found_regnums = contract_service.fetch_contracts(
            product_search=product_search,
            product_attribute=product_attribute,
            region_code=kod_regiona,
            date_start=date_start,
            date_end=date_end,
            fz=fz,
            price_min=price_min,
            price_max=price_max,
            okdp=okdp
        )

        if df is None or df.empty:
            return HttpResponse("No data found for the given parameters.")

        # 3. Analyze OKPD2 and classify
        sprav_path = os.path.join(BASE_DIR, 'ОКПД2_2017-01-01.xlsx')

        df_analyzed, list_nan2 = analytics_service.analyze_okpd2(
            df=df,
            product_search=product_search,
            okpd2_sprav_path=sprav_path
        )

        ce = analytics_service.classify_okpd2(
            df=df_analyzed,
            list_nan2=list_nan2,
            product_search=product_search
        )

        # Pagination logic
        items_per_page = 50
        paginator = Paginator(ce.index, items_per_page)

        try:
            page_obj = paginator.page(page_number)
            # Get the slice of the DataFrame for the current page
            current_page_indices = page_obj.object_list
            ce_page = ce.loc[current_page_indices]
        except:
            # Fallback to first page or empty if page is out of range
            page_obj = paginator.get_page(page_number)
            current_page_indices = page_obj.object_list
            ce_page = ce.loc[current_page_indices]

        # Format date range for display
        date_range_str = f"{date_start.strftime('%d.%m.%Y')}-{date_end.strftime('%d.%m.%Y')}"

        dictd = {
            'ce': ce_page.to_html(classes='table table-striped'),
            'product_search': product_search,
            'kod_regiona': kod_regiona,
            'dates_contracts_baza': date_range_str,
            'my_directory': BASE_DIR,
            'page_obj': page_obj,
            'form': form,
            'found_regnums': found_regnums
        }

        return render(request, 'result_table.html', dictd)
    else:
        return render(request, "index.html", {"form": form})

def analise(request):
    return HttpResponse("<h2>Анализ выгрузки</h2>")

def contact(request):
    return HttpResponse("<h2>Контакты</h2>")
