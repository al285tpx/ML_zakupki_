from django.db import models
from django_pandas.managers import DataFrameManager


class Product(models.Model):
    """Модель продукт в контракте"""
    contract = models.TextField()
    product_price = models.FloatField()
    product_kol_vo = models.FloatField()
    product_ed_izm = models.TextField()
    OKEI = models.TextField()
    product_sum = models.FloatField()
    product_name = models.TextField()
    OKPD2_code = models.TextField()
    OKPD2_name = models.TextField()
    regionCode = models.IntegerField()
    supplier_name = models.TextField()
    supplier_INN = models.TextField()
    supplier_address = models.TextField()
    customer_name = models.TextField()
    customer_INN = models.TextField()
    customer_address = models.TextField()
    signDate = models.DateField()

    def __str__ (self):
        return self.product_name

    # objects = models.Manager()
    # pdobjects = DataFrameManager()  # Pandas-Enabled Manager
