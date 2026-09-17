from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class SearchRequest(BaseModel):
    product_search: str = Field(..., description="Product name to search for")
    product_attribute: Optional[str] = Field(None, description="Additional product attribute")
    region_code: str = Field(..., description="Region code (e.g., '77')")
    date_start: datetime = Field(..., description="Start date for contract signing")
    date_end: datetime = Field(..., description="End date for contract signing")
    fz: Optional[str] = Field(None, description="Law number (e.g., '44' or '223')")
    price_min: Optional[int] = Field(None, description="Minimum product price")
    price_max: Optional[int] = Field(None, description="Maximum product price")
    okdp: Optional[str] = Field(None, description="OKPD2 code")

class ProductSchema(BaseModel):
    contract: Optional[str]
    regionCode: Optional[str]
    signDate: Optional[str]
    product_price: Optional[float]
    product_kol_vo: Optional[float] = Field(None, alias="product_kol-vo")
    product_ed_izm: Optional[str]
    OKEI: Optional[str]
    product_sum: Optional[float]
    product_name: Optional[str]
    OKPD2_code: Optional[str]
    OKPD2_name: Optional[str]
    supplier_name: Optional[str]
    supplier_INN: Optional[str]
    supplier_address: Optional[str]
    customer_name: Optional[str]
    customer_INN: Optional[str]
    customer_address: Optional[str]
    Quartile: Optional[str]

    class Config:
        populate_by_name = True

class TaskStatus(BaseModel):
    task_id: str
    status: str  # 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'
    result_url: Optional[str] = None
    error: Optional[str] = None

class AnalysisResponse(BaseModel):
    task_id: str
    status: str
