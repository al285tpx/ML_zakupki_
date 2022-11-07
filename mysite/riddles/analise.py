import requests
import pandas as pd
import numpy as np
from datetime import timedelta
# from .forms import UserForm

import os
import glob


def okpd2analize (my_directory, product_search, kod_regiona, dates_contracts_baza):
    # на всякий случай вернемся в исходную директорию:
    os.chdir(my_directory)
    print(my_directory)
    # загрузим нужный файл:
    combined_csv_e = pd.read_csv(
        my_directory + '/общий_' + product_search + '_регион-' + kod_regiona + '_' + dates_contracts_baza + '.csv')
    # уберем строки с пустыми значениями цены за единицу продукции:
    combined_csv_e = combined_csv_e[combined_csv_e.product_price.notna()]

    # посмотрим, какие позиции остались без кодов ОКПД2:
    ce_nan = combined_csv_e[combined_csv_e['OKPD2_name'].isna() == True]
    # составим список этих позиций
    list_nan = ce_nan['product_name'].unique()

    # заполним пропущенные значение кода ОКПД2
    # для этого сначала скопируем имеющуюся информация по кодам в новые результирующие столбцы:
    combined_csv_e['OKPD2_code_res'] = combined_csv_e['OKPD2_code']
    combined_csv_e['OKPD2_name_res'] = combined_csv_e['OKPD2_name']

    # ВАРИАНТ (с заменой, подходит для дальнейшей многоклассовой классификации)
    # так как нам каждому уникальному названию надо будет поставить в соответствие только один из кодов ОКПД2, то при заполнении
    # пропущенных кодов ОКПД2 для одинаковой продукции нужно выставить наиболее часто встречающейся для нее код ОКПД2
    # заполним редкие значения в результирующих столбцах кодах ОКПД2 наиболее часто встречающимися значениями для
    # соответствующего продукта, при этом в работу возьмем название продукта до точки или до круглой скобки:
    for j in list_nan:
        # выделим все строки с конкретным продуктов, для которых могут быть пропущены коды ОКПД2
        # возьмем название продукта до точки:
        ce_1 = combined_csv_e[
            combined_csv_e['product_name'].str.split(pat='.', expand=True)[0].str.rstrip(' ') == j]
        # выделим для этих строк все НЕНУЛЕВЫЕ приведенные названия кода ОКПД2:
        ce_2 = ce_1[ce_1['OKPD2_name'].isna() == False]
        # создадим список из отобранных ненулевых названий кодов ОКПД2
        list_3 = ce_2['OKPD2_name'].unique()
        # выясним максимальное количество возможного упоминания одного из кодов ОКПД2 для этого продукта:
        max_4 = ce_2['OKPD2_name'].value_counts().max()
        # для каждого кода ОКПД2 из списка выясним частоту упоминания:
        for i in list_3:
            if ce_2[ce_2['OKPD2_name'] == i]['OKPD2_name'].value_counts().values == max_4:
                x_code = ce_2[ce_2['OKPD2_name'] == i]['OKPD2_code'].values[0]
                combined_csv_e.loc[combined_csv_e.product_name.str.split(pat='.', expand=True)[0].str.rstrip(
                    ' ') == j, 'OKPD2_name_res'] = i
                combined_csv_e.loc[combined_csv_e.product_name.str.split(pat='.', expand=True)[0].str.rstrip(
                    ' ') == j, 'OKPD2_code_res'] = x_code

    # посмотрим, какие позиции оказались незаполненными:
    ce_nan2 = combined_csv_e[combined_csv_e['OKPD2_name_res'].isna() == True]
    list_nan2 = ce_nan2['product_name'].unique()

    # при желании часть из них можно заполнить с помощью справочника ОКПД2:
    # сначала загрузим справочник кодов ОКПД2 и преобразуем его в удобный формат:
    OKPD2_sprav = pd.read_excel('ОКПД2_2017-01-01.xlsx')
    OKPD2_sprav['len_kod'] = OKPD2_sprav['01'].str.len()
    OKPD2_sprav_12 = OKPD2_sprav[OKPD2_sprav['len_kod'] == 12].drop(['len_kod'], axis=1)
    OKPD2_sprav_12 = OKPD2_sprav_12.rename(
        columns={'01': 'code', 'Продукция и услуги сельского хозяйства и охоты': 'OKPD2_name'})

    # можно посмотреть часть справочника ОКПД2, соответствующую искомому продукту:
    # OKPD2_sprav_12[OKPD2_sprav_12['OKPD2_name'].str.contains(pat=product_search, case=False)]

    # заполним еще не внесенные коды ОКПД2 полностью совпадающими значениями из справочника ОКПД2 (если есть):
    list_OKPD2 = OKPD2_sprav_12['OKPD2_name'].values
    for i in list_nan2:
        for j in list_OKPD2:
            if i == j:
                x_code = OKPD2_sprav_12[OKPD2_sprav_12['OKPD2_name'] == list_OKPD2]['code']
                combined_csv_e.loc[
                    (combined_csv_e.product_name == i) & (combined_csv_e.OKPD2_name.isna()), 'OKPD2_name_res'] = i
                combined_csv_e.loc[
                    (combined_csv_e.product_name == i) & (
                        combined_csv_e.OKPD2_code.isna()), 'OKPD2_code_res'] = x_code
                print('найдены коды ОКПД2 из справочника', i, x_code)

                # сохраним результат, полученный без ML:
    variant_without_ML = combined_csv_e

    # посмотрим, для каких продуктов код ОКПД2 оказался незаполненным:
    # list_nan2

    return combined_csv_e, list_nan2


# combined_csv_e, list_nan2 = okpd2analise(my_directory, product_search, kod_regiona, dates_contracts_baza)

def multiclassifire (combined_csv_e, list_nan2):
    # ВАРИАНТ 1. Присвоение кодов ОКПД2, исходя из примеров продукции
    # НЕДОСТАТКИ: присваивает коды ОКПД2 исходя из уже присвоенных кодов. То есть, если есть продукция без кода ОКПД2 и код,
    # который было бы правильно присвоить продукции, еще не использовался в этом файле, то он не будет присвоен, а будет присвоен
    # наиболее подходящий из имеющихся. Этот способ хорошо определит коды для позиций, которые часто встречаются и отличающихся
    # деталями в названии, например, различные виды туалетной бумаги. Но при таком подходе будет возникать "засорение" хорошо
    # сгруппированных позиций продуктами, которые по факту не имеют отношения к данной группе, потому что система будет вынуждена
    # подбирать коды из имеющихся.

    # Мультиклассовая классификация текста: Классификатор делает предположение, что каждая новая жалоба относится к одной и
    # только одной категории. Это проблема классификации классов по нескольким классам
    # Желательно иметь классификатор, который дает высокую точность прогнозирования по сравнению с классом большинства,
    # сохраняя при этом разумную точность для классов меньшинства

    # создадим названия колонок, куда внесем данных из известных нам кодов ОКПД2, их названий и названий продуктов
    # попробуем обойтись уже имеющимися значениями кодов ОКПД2, не переводя их в числа
    col = ['product_name', 'OKPD2_code_res', 'OKPD2_name_res']
    # создадим пустой датафрейм с названными колонками
    df3 = combined_csv_e[col]
    # мы удалим пустые значения OKPD2_name_res
    df3 = df3[pd.notnull(df3['OKPD2_name_res'])]
    # избавимся от дублирующих строк и отсортируем данные по коду ОКПД2:
    category_id_df = df3[['OKPD2_name_res', 'OKPD2_code_res']].drop_duplicates().sort_values('OKPD2_code_res')
    # создадим словарь перевода наименований кодов ОКПД2 в сами коды ОКПД2:
    category_to_id = dict(category_id_df.values)
    # и создадим словарь переводов из кодов ОКПД2 в наименования кодов ОКПД2:
    id_to_category = dict(category_id_df[['OKPD2_code_res', 'OKPD2_name_res']].values)

    # количество позиций не сбалансировано: каких-то позиций меньше, каких-то больше.  Обычные алгоритмы часто смещены в сторону
    # мажоритарного класса, не принимая во внимание распределение данных.В худшем случае классы меньшинства рассматриваются как
    # выбросы и игнорируются. В нашем случае изучения несбалансированных данных, большинство классов могут представлять для нас
    # большой интерес. Желательно иметь классификатор, который дает высокую точность прогнозирования по сравнению с классом
    # большинства, сохраняя при этом разумную точность для классов меньшинства. Поэтому мы оставим все как есть.

    # посчитаем частоту слов

    # прежде чем выявлять признаки в названиях продуктов, "почистим" названия: удалим знаки препинания, латинские буквы, заменим
    # все символы на строчные (цифры удалять не будем, так как они встречаются в кодах ОКПД2):

    import re

    df3['product_name_res'] = df3['product_name'].apply(lambda x: x.lower())
    df3['product_name_res'] = df3['product_name_res'].str.split('[a-zA-z№=.+"«]', expand=True)[0].str.rstrip(
        '[ (-:,]')
    df3['product_name_res'] = df3['product_name_res'].str.replace(r'\n', ' ').str.rstrip('[ (-:,]')
    df3['product_name_res'] = df3['product_name_res'].str.replace(r'\t', ' ').str.rstrip('[ (-:,]')
    df3['product_name_res'].unique()

    # немного изменим параметры под нашу задачу:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from nltk.corpus import stopwords

    sw = stopwords.words("russian")
    # добавим в словарь слово "прочий":
    lst_prochie = ['прочий', 'прочая', 'прочее', 'прочие', 'прочих']
    sw = sw.append(lst_prochie)

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='utf-8', ngram_range=(1, 2),
                            stop_words=sw)
    # sublinear_df установлен на True - использование логарифмической формы для частоты.
    # min_df - с плавающей точкой в диапазоне [0.0, 1.0] или int, по умолчанию = 1:
    # минимальное количество документов, в которых должно храниться слово (в наших настройках сначла было 2, изменим на 1)
    # norm устанавливается на l2, чтобы гарантировать, что все наши векторы функций имеют евклидову норму 1.
    # ngram_range устанавливается (1, 2), чтобы указать, что мы хотим рассмотреть как униграммы, так и биграммы (сочетания из 2х)
    # stop_words устанавливаются "russian", чтобы удалить неинформативные слова (сделано выше)

    # определим признаки кода ОКПД2, исходя из примеров продукции
    features = tfidf.fit_transform(df3.product_name_res).toarray()
    labels = df3.OKPD2_code_res
    # features.shape

    # Теперь каждый из продуктов описан признаками, представляющими оценку tf-idf для разных униграмм и биграмм
    # Мы можем использовать sklearn.feature_selection.chi2, чтобы найти термины, которые наиболее соответствуют каждому из кодов:
    from sklearn.feature_selection import chi2
    # импорт модуля для вычисления хи-квадрат характеристики между каждым неотрицательным признаком и классом
    import numpy as np

    N = 2
    for OKPD2_name_res, OKPD2_code_res in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == OKPD2_code_res)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        # print("# '{}':".format(OKPD2_name_res))
        # print("  . Наиболее коррелирующие униграммы:\n. {}".format('\n. '.join(unigrams[-N:])))
        # print("  . Наиболее коррелирующие биграммы:\n. {}".format('\n. '.join(bigrams[-N:])))

    # обучим нашу модель с помощью наивного байесовского классификатора:

    # наиболее подходящим для подсчета слов является полиномиальный вариант
    # (полиномы - слова, получаемые из слова перестановкой букв)

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    # TfidfTransformer используется для нормализации частоты слов (то есть чтобы большое количество слова в длинном документе не
    # перевешивало небольшое количество того же слова в коротком)

    from sklearn.naive_bayes import MultinomialNB

    # загрузим полиномиальный наивный баесовский классификатор

    # разобъем данные на обучающую и тестовую выборку
    # перед этим на всякий случай сначала сделаем все буквы в названии кодов ОКПД2 строчными
    df3['OKPD2_name_res'] = df3['OKPD2_name_res'].apply(lambda x: x.lower())

    X_train, X_test, y_train, y_test = train_test_split(df3['product_name_res'], df3['OKPD2_name_res'],
                                                        random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    # для каждой позции, для которых все еще не определн код ОКПД2, распечатаем предлагаемый вариант кода:
    # for i in list_nan2:
    #   print(i, '\n', clf.predict(count_vect.transform([i])), '\n', '\n')

    # присвоим продуктам с все еще отсутствующим кодом ОКПД2 подобранные с помощью ML из уже использованных коды:
    # для всех продуктов с отсутствующим кодом ОКПД2...
    for i in list_nan2:
        # ...запишем для каждого продукта предсказанное название кода ОКПД2:
        OKPD2_name_predicted = clf.predict(count_vect.transform([i]))
        # ...составим список индексов для каждого из продуктов
        x_index = combined_csv_e[combined_csv_e['product_name'] == i].index
        # затем внутри каждой группы продуктов...
        for j in x_index:
            # ...заполним отсутствующее назнвание кода ОКПД2 предсказанным
            combined_csv_e.loc[j, 'OKPD2_name_res'] = OKPD2_name_predicted
            # и запишем его с заглавной буквы
            combined_csv_e.loc[j, 'OKPD2_name_res'] = combined_csv_e.loc[j, 'OKPD2_name_res'].capitalize()
            # отберем все продукты с таким же названием кода ОКПД2:
            x_list = combined_csv_e[combined_csv_e['OKPD2_name_res'].str.lower().values == OKPD2_name_predicted]
            # запомним, каким кодом заполненые непустые коды ОКПД2:
            x_code = x_list['OKPD2_code_res'][x_list['OKPD2_code_res'].notna()].unique()
            # внесем этот код для таких же продуктов с отсутствующим кодом:
            combined_csv_e.loc[combined_csv_e.product_name == i, ['OKPD2_code_res']] = x_code
            combined_csv_e.OKPD2_code_res[combined_csv_e.OKPD2_code_res == 'i'] = x_code[0]

            # for index in combined_csv_e['OKPD2_code_res'][combined_csv_e['product_name'] == i].index:
            #    combined_csv_e['OKPD2_code_res'][index] = x_code[0]

    # на всякий случай вернемся в исходную директорию:
    os.chdir(my_directory)
    combined_csv_a = combined_csv_e
    # уберем строки с пустыми значениями цены за единицу продукции:
    combined_csv_a = combined_csv_a[combined_csv_a.product_price.notna()]

    # отберем строчки только с теми кодами OKPD2, в которых встречается строка поиска product_search
    # (в нашем случае коды со словом "бумага"):
    ce = combined_csv_a[combined_csv_a['OKPD2_name_res'].str.contains(pat=product_search, case=False) == True]
    # составим список таких кодов
    list_e = ce['product_ed_izm'].unique()

    # чтобы данные отображались красиво, ограничим вывод названия кода ОКПД2:
    pd.set_option('max_colwidth', 31)

    # для каждой единицы измерения из списка выведем, разбивая по кодам ОКПД2, минимиальное, медиану, среднее и максимальной
    # значения:
    # for i in list_e:
    #    print('\n', i.upper(), '\n', round(ce[ce['product_ed_izm'] == i].groupby(['OKPD2_name_res']).agg(
    #        {'product_price': ['min', 'median', 'mean', 'max']}), 2))
    # вернеем стандартную ширину вывода строки

    pd.set_option('max_colwidth', 50)  # вывести

    # сначала попробуем сгрупировать по кодам ОКПД2 и посмотреть разброс цен:
    round(ce.groupby(['OKPD2_name_res']).agg({'product_price': ['min', 'median', 'mean', 'max']}, 3), 2)
    # мы видим: 1) много позиций, не имеющего прямого отношения к бумаге; 2) большой разброс цен

    # создадим список непустых значений индекса OKEI:
    OKEI = ce['product_ed_izm'].sort_values().unique()
    OKEI = [x for x in OKEI if str(x) != 'nan']
    # Создадим пустой столбец в df для квартилей
    ce['Quartile'] = None

    for okei in OKEI:
        a = pd.DataFrame()
        b = pd.DataFrame()
        # запоминаем код OKEI
        c = ce[ce['product_ed_izm'] == okei]['product_ed_izm'].head(1).values  # задает начальное значение
        # отбираем позиции с ненулевымми ценами для выбранного кода OKEI
        a = ce[ce['product_ed_izm'] == okei]['product_price'][
            ce[ce['product_ed_izm'] == okei]['product_price'].notna()]
        # отбираем позиции с ненулевым количеством для выбранного кода OKEI
        q = ce[ce['product_ed_izm'] == okei]['product_kol-vo'][
            ce[ce['product_ed_izm'] == okei]['product_kol-vo'].notna()]
        # запоминаем название единицы измерения
        b = ce[ce['product_ed_izm'] == okei]['product_ed_izm'].head(1).values[0]
        m = round(ce[ce['product_ed_izm'] == okei]['product_price'].mode()[0], 2)
        s = round(ce[ce['product_ed_izm'] == okei]['product_price'].mean(), 2)
        print('\n', 'единица измерения', '\t', b.upper())
        print('РАЗБИЕНИЕ НА КВАРТИЛИ:')
        try:
            a_min = round(a.min(), 4)
            q1 = round(a.quantile(0.25), 4)
            q2 = round(a.quantile(0.5), 4)
            q3 = round(a.quantile(0.75), 4)
            a_max = round(a.max(), 4)
            q_q1 = round(q.quantile(0.25), 4)
            q_q3 = round(q.quantile(0.75), 4)

            # назначим каждой выбранной записи свой квартиль, запишем в фрейм
            for index in a.index:
                # print(a.index)
                if a_min <= a[index] and a[index] < q1:
                    # print('Q1' , a[index])
                    ce['Quartile'][index] = 'Q1'
                elif q1 <= a[index] and a[index] < q2:
                    ce['Quartile'][index] = 'Q2'
                    # print('Q2' , a[index])
                elif q2 <= a[index] and a[index] < q3:
                    ce['Quartile'][index] = 'Q3'
                    # print('Q3' , a[index])
                elif q3 <= a[index] and a[index] <= a_max:
                    ce['Quartile'][index] = 'Q4'
                    # print('Q4' , a[index])
                else:
                    ce['Quartile'][index] = 'unique'
        except:
            pass
        if np.isnan(a_min) == True:
            print('не удалось разбить на интервалы, так как значений нет')
        else:
            print(a_min, '\t', q1, '\t', q2, '\t', q3, '\t', a_max)
        print('медиана - ', q2, '\t', 'среднее - ', s)

        os.chdir(my_directory)
        ce.to_csv(
            'общий_' + product_search + '_регион-' + kod_regiona + '_' + dates_contracts_baza + '_квартили' + '.csv',
            encoding='Windows-1251', index=False)
    return ce

# ce = multiclassifire(combined_csv_e, list_nan2)
