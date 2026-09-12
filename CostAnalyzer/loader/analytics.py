import pandas as pd
import numpy as np
import re
import logging
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AnalyticsService:
    """Service for analyzing contract data and classifying OKPD2 codes."""

    def __init__(self):
        self._okpd2_sprav_cache = None
        self._last_sprav_path = None

    def _get_okpd2_sprav(self, path: str) -> Optional[pd.DataFrame]:
        """Loads and caches the OKPD2 reference spreadsheet."""
        if self._okpd2_sprav_cache is not None and self._last_sprav_path == path:
            return self._okpd2_sprav_cache

        try:
            logger.info(f"Loading OKPD2 spreadsheet from {path}...")
            sprav = pd.read_excel(path)
            # Standardize spreadsheet columns
            sprav['len_kod'] = sprav.iloc[:, 0].astype(str).str.len()
            sprav_12 = sprav[sprav['len_kod'] == 12].drop(['len_kod'], axis=1)

            # Map column names for consistency
            sprav_12 = sprav_12.rename(columns={
                sprav_12.columns[0]: 'code',
                sprav_12.columns[1]: 'OKPD2_name'
            })
            self._okpd2_sprav_cache = sprav_12
            self._last_sprav_path = path
            return self._okpd2_sprav_cache
        except Exception as e:
            logger.error(f"Error loading OKPD2 spreadsheet: {e}")
            return None

    def analyze_okpd2(self, df: pd.DataFrame, product_search: str, okpd2_sprav_path: str) -> tuple[pd.DataFrame, list]:
        """
        Fills missing OKPD2 codes using existing data and a reference spreadsheet.
        """
        df = df.copy()
        # Remove rows with missing unit prices
        df = df[df['product_price'].notna()]

        # Identify products missing OKPD2 names
        ce_nan = df[df['OKPD2_name'].isna()]
        list_nan = ce_nan['product_name'].unique()

        # Result columns
        df['OKPD2_code_res'] = df['OKPD2_code']
        df['OKPD2_name_res'] = df['OKPD2_name']

        # Strategy: fill missing codes using the most frequent OKPD2 for similar product names
        for product in list_nan:
            # Match products starting with the same name (before the first dot)
            mask = df['product_name'].str.split('.', expand=True)[0].str.rstrip(' ') == product
            ce_1 = df[mask]

            # Find non-null OKPD2 names for this product
            ce_2 = ce_1[ce_1['OKPD2_name'].notna()]
            if ce_2.empty:
                continue

            # Find the most frequent OKPD2 name
            counts = ce_2['OKPD2_name'].value_counts()
            if counts.empty:
                continue

            most_frequent_name = counts.idxmax()
            # Get the corresponding code
            most_frequent_code = ce_2[ce_2['OKPD2_name'] == most_frequent_name]['OKPD2_code'].values[0]

            df.loc[mask, 'OKPD2_name_res'] = most_frequent_name
            df.loc[mask, 'OKPD2_code_res'] = most_frequent_code

        # Fill remaining missing values using the reference spreadsheet
        ce_nan2 = df[df['OKPD2_name_res'].isna()]
        list_nan2 = ce_nan2['product_name'].unique()

        okpd2_sprav_12 = self._get_okpd2_sprav(okpd2_sprav_path)
        if okpd2_sprav_12 is not None:
            list_okpd2_names = okpd2_sprav_12['OKPD2_name'].values
            for product in list_nan2:
                for sprav_name in list_okpd2_names:
                    if product == sprav_name:
                        code = okpd2_sprav_12[okpd2_sprav_12['OKPD2_name'] == sprav_name]['code'].values[0]
                        df.loc[(df['product_name'] == product) & (df['OKPD2_name'].isna()), 'OKPD2_name_res'] = sprav_name
                        df.loc[(df['product_name'] == product) & (df['OKPD2_code'].isna()), 'OKPD2_code_res'] = code
                        break

        return df, list_nan2

    @staticmethod
    def classify_okpd2(df: pd.DataFrame, list_nan2: list, product_search: str) -> pd.DataFrame:
        """
        Uses ML (Naive Bayes) to predict missing OKPD2 codes.
        """
        df = df.copy()
        col = ['product_name', 'OKPD2_code_res', 'OKPD2_name_res']
        df_train = df[col].dropna(subset=['OKPD2_name_res']).copy()

        if df_train.empty:
            return df

        def preprocess_text(text):
            if not isinstance(text, str):
                return ''
            text = text.lower()
            # Split on Latin letters and some special characters
            text = re.split('[a-zA-z№=.+"«]', text)[0]
            # Clean up whitespace and trailing punctuation
            text = text.replace(r'\n', ' ').replace(r'\t', ' ').strip().rstrip('[ (-:,]')
            return text

        # Preprocessing training data
        df_train['product_name_res'] = df_train['product_name'].apply(preprocess_text)

        # TF-IDF setup
        sw = stopwords.words("russian")
        sw.extend(['прочий', 'прочая', 'прочее', 'прочие', 'прочих'])

        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words=sw)

        # Train model
        X = tfidf.fit_transform(df_train['product_name_res'])
        y = df_train['OKPD2_name_res']

        clf = MultinomialNB().fit(X, y)

        # Predict for missing values
        for product in list_nan2:
            # Prepare product name for prediction
            prod_res = preprocess_text(product)

            pred_name = clf.predict(tfidf.transform([prod_res]))[0]
            pred_name_cap = pred_name.capitalize()

            mask = df['product_name'] == product
            df.loc[mask, 'OKPD2_name_res'] = pred_name_cap

            # Find most frequent corresponding code from training data to avoid inconsistency
            matching_codes = df_train[df_train['OKPD2_name_res'].str.lower() == pred_name.lower()]['OKPD2_code_res']
            if not matching_codes.empty:
                most_frequent_code = matching_codes.value_counts().idxmax()
                df.loc[mask, 'OKPD2_code_res'] = most_frequent_code

        # Calculate quartiles and filter results
        return AnalyticsService.calculate_price_quartiles(df, product_search)

    @staticmethod
    def calculate_price_quartiles(df: pd.DataFrame, product_search: str) -> pd.DataFrame:
        """
        Filters data by product search in OKPD2 name and calculates price quartiles per unit of measure.
        """
        df = df.copy()
        # Remove rows with missing unit prices
        df = df[df['product_price'].notna()]
        # Filter by product search in OKPD2 name
        ce = df[df['OKPD2_name_res'].str.contains(product_search, case=False, na=False)].copy()

        if ce.empty:
            return ce

        okei_list = [x for x in ce['product_ed_izm'].unique() if pd.notna(x)]
        ce['Quartile'] = 'unique'

        for okei in okei_list:
            mask = ce['product_ed_izm'] == okei
            prices = ce.loc[mask, 'product_price'].dropna()

            if prices.empty:
                continue

            q1 = prices.quantile(0.25)
            q2 = prices.quantile(0.50)
            q3 = prices.quantile(0.75)
            p_min = prices.min()
            p_max = prices.max()

            def assign_quartile(p):
                if p_min <= p < q1: return 'Q1'
                if q1 <= p < q2: return 'Q2'
                if q2 <= p < q3: return 'Q3'
                if q3 <= p <= p_max: return 'Q4'
                return 'unique'

            ce.loc[mask, 'Quartile'] = prices.apply(assign_quartile)

        return ce.sort_values("Quartile")
