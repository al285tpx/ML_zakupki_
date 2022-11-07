from django.urls import resolve
from django.test import TestCase
from . import views

class HomePageTest(TestCase):
    """Тест домашней страницы"""

    def test_root_url_resolves_to_home_page(self):
        """тест: коневой url преобразуется в предствлени домашней страницы"""
        found = resolve('/')
        self.assertEqual(found.func, home_page)

