import holidays
import locale
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

locale.setlocale(locale.LC_TIME, 'pt_BR')
date_descriptions = {}
month_names = {
    1: 'Janeiro',
    2: 'Fevereiro',
    3: 'Março',
    4: 'Abril',
    5: 'Maio',
    6: 'Junho',
    7: 'Julho',
    8: 'Agosto',
    9: 'Setembro',
    10: 'Outubro',
    11: 'Novembro',
    12: 'Dezembro'
}

def get_data(input_seq, output_seq, padding_step, id_max_size, date_max_size):
    while True:
        try:
            ts_df = load_data()
            break
        except:
            print("failed")
            pass

    id_corpus = ts_df["ID_STRING"].unique()
    date_corpus = ts_df["DATE_STRING"].unique()
    id_vectorizer = get_vectorizer(id_corpus, id_max_size)
    date_vectorizer = get_vectorizer(date_corpus, date_max_size)

    grouped_data = ts_df.groupby('ID_STRING')
    sorted_data = {id: data for id, data in grouped_data}

    padded_sequences = {}
    padded_dates = {}
    for id, data in sorted_data.items():
        sequences = [data['QTD_ITEM'].values[i: i + input_seq + output_seq] for i in range(0, len(data) - input_seq - output_seq + 1, padding_step)]
        dates = [data['DATE_STRING'].values[i: i + input_seq + output_seq] for i in range(0, len(data) - input_seq - output_seq + 1, padding_step)]
        padded_sequences[id] = sequences
        padded_dates[id] = dates

    train_ratio = 0.8
    train_sequences, val_sequences = {}, {}
    train_dates, val_dates = {}, {}
    for id, padded_seq in padded_sequences.items():
        train_seq, val_seq = train_test_split(padded_seq, train_size=train_ratio, shuffle=False)
        train_sequences[id] = train_seq
        val_sequences[id] = val_seq
    for id, padded_date in padded_dates.items():
        train_date, val_date = train_test_split(padded_date, train_size=train_ratio, shuffle=False)
        train_dates[id] = train_date
        val_dates[id] = val_date

    X_train, Y_train, X_val, Y_val = [], [], [], []
    for id in padded_sequences.keys():
        X_train.extend(np.array(train_sequences[id])[:, :input_seq])
        Y_train.extend(np.array(train_sequences[id])[:, input_seq:])
        X_val.extend(np.array(val_sequences[id])[:, :input_seq])
        Y_val.extend(np.array(val_sequences[id])[:, input_seq:])
    
    X_dates_train, X_dates_val = [], []
    for id in padded_dates.keys():
        X_dates_train.extend(np.array(train_dates[id])[:, :input_seq])
        X_dates_val.extend(np.array(val_dates[id])[:, :input_seq])
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    X_dates_train = np.array(X_dates_train)
    X_dates_val = np.array(X_dates_val)

    return X_train, Y_train, X_val, Y_val, X_dates_train, X_dates_val, id_vectorizer, date_vectorizer

def get_vectorizer(corpus, max_size):
    vectorizer = tf.keras.layers.TextVectorization(
        standardize = "lower_and_strip_punctuation",
        split = "whitespace",
        ngrams = 5,
        output_mode = "int",
        output_sequence_length = max_size)
    vectorizer.adapt(corpus)
    return vectorizer

def generate_date_description(date):
    if date in date_descriptions:
        return date_descriptions[date]

    month = date.month
    year = date.year
    month_name = month_names[month]
    num_days = pd.Period(date, freq='M').days_in_month
    start_date = pd.Timestamp(year, month, 1)
    end_date = start_date + pd.tseries.offsets.MonthEnd(0)
    date_range = pd.bdate_range(start=start_date, end=end_date)
    num_weekdays = len(date_range)
    num_weekends = num_days - num_weekdays
    br_holidays = holidays.Brazil(years=year)
    month_holidays = [
        holiday for holiday in br_holidays.items()
        if holiday[0].month == month and holiday[0].year == year
    ]

    description = f"{month_name} de {year}, contém {num_days} dias dos quais {num_weekdays} foram dias úteis, {num_weekends} foram finais de semana."
    if month_holidays:
        holidays_description = ' '.join([f"{date.strftime('%d/%m/%Y')} - {name}" for date, name in month_holidays])
        description += f" {month_name} teve {len(month_holidays)} feriado(s): {holidays_description}"

    date_descriptions[date] = description
    return description

def load_data():
    try:
        ts_df = pd.read_pickle('ts_df.pkl')
    except:
        from utils import setup_connection
        from sqlalchemy import text
        import database_config as cfg
        _, con = setup_connection(cfg.dw_path)
        query = text("""
            SELECT
                TRIM(FILIAL.NOM_FILIAL) AS NOM_FILIAL,
                FILIAL.DES_REGIONAL,
                FILIAL.NOM_REGIONAL,
                CAD.DES_AGRUPAMENTO_LINHA,
                CAD.DES_CATEGORIA_TIPO_EQUIPAMENTO,
                CAD.DES_SUBGRUPO_PRODUTO,
                CAD.COD_PRODUTO_MASTER,
                CAD.DES_GRUPO_MARCA,
                CAD.DES_MARCA,
                CAD.COD_PRODUTO,
                FT.DAT_FATO,
                SUM(FT.QTD_ITEM) AS QTD_ITEM,
                SUM(FT.VAL_ITEM) AS VAL_ITEM
            FROM
                COMERCIAL.FATO_FATURAMENTO FT
            LEFT JOIN
                DM.DIM_PRODUTO CAD ON CAD.COD_PRODUTO = FT.COD_PRODUTO
            LEFT JOIN
                DM.DIM_CLIENTE CLIENTE ON CLIENTE.UID_CLIENTE = FT.UID_CLIENTE
            LEFT JOIN
                DM.DIM_EMPRESA_FILIAL FILIAL ON FILIAL.UID_EMPRESA_FILIAL = FT.UID_EMPRESA_FILIAL
            WHERE
                CAD.TIP_REGISTRO = 'P' AND
                CLIENTE.DES_PERFIL_CLIENTE = 'Cliente Comum' AND
                FT.DES_REGISTRO = 'S' AND
                (FILIAL.COD_EMPRESA NOT IN ('001', '003', '004', '010', '011', '012', '014', '015', '016', '020', '022', '037', '042') OR  FILIAL.NOM_FILIAL = 'V/NAVEGANTES')
            GROUP BY
                FILIAL.NOM_FILIAL,
                FILIAL.DES_REGIONAL,
                FILIAL.NOM_REGIONAL,
                CAD.DES_AGRUPAMENTO_LINHA,
                CAD.DES_CATEGORIA_TIPO_EQUIPAMENTO,
                CAD.DES_SUBGRUPO_PRODUTO,
                CAD.COD_PRODUTO_MASTER,
                CAD.DES_GRUPO_MARCA,
                CAD.DES_MARCA,
                CAD.COD_PRODUTO,
                FT.DAT_FATO
            ORDER BY
                FILIAL.NOM_FILIAL,
                CAD.COD_PRODUTO,
                FT.DAT_FATO
        """)
        df = pd.read_sql_query(query, con)
        df.fillna("null", inplace=True)
        df['ID_STRING'] = df['NOM_FILIAL']
        df['ID_STRING'] += " " + df['DES_REGIONAL']
        df['ID_STRING'] += " " + df['NOM_REGIONAL']
        df['ID_STRING'] += " " + df['DES_AGRUPAMENTO_LINHA']
        df['ID_STRING'] += " " + df['DES_CATEGORIA_TIPO_EQUIPAMENTO']
        df['ID_STRING'] += " " + df['DES_SUBGRUPO_PRODUTO']
        df['ID_STRING'] += " " + df['COD_PRODUTO_MASTER']
        df['ID_STRING'] += " " + df['DES_GRUPO_MARCA']
        df['ID_STRING'] += " " + df['DES_MARCA']
        df['ID_STRING'] += " " + df['COD_PRODUTO']
        df.drop([
            'NOM_FILIAL', 
            'DES_REGIONAL', 
            'NOM_REGIONAL', 
            'DES_AGRUPAMENTO_LINHA', 
            'DES_CATEGORIA_TIPO_EQUIPAMENTO', 
            'DES_SUBGRUPO_PRODUTO', 
            'COD_PRODUTO_MASTER', 
            'DES_GRUPO_MARCA', 
            'DES_MARCA', 
            'COD_PRODUTO'], axis=1, inplace=True)
        df['DAT_FATO'] = pd.to_datetime(df['DAT_FATO'])
        df = df.groupby(['ID_STRING', pd.Grouper(key = 'DAT_FATO', freq = "MS")]).sum()
        idx = pd.MultiIndex.from_product([
            df.index.levels[0], 
            pd.date_range(df.index.levels[1].min(), df.index.levels[1].max(), freq="MS")], 
            names=['ID_STRING', 'DAT_FATO'])
        new_df = pd.DataFrame(index=idx).reset_index()
        ts_df = pd.merge(new_df, df, how='left', left_on=['ID_STRING', 'DAT_FATO'], right_index=True)
        ts_df.sort_values(['ID_STRING', 'DAT_FATO'], inplace=True)
        ts_df['QTD_ITEM'].fillna(0.0, inplace=True)
        ts_df.reset_index(drop=True, inplace=True)
        ts_df['DATE_STRING'] = ts_df['DAT_FATO'].apply(generate_date_description)
        ts_df.to_pickle('ts_df.pkl')
        con.close()
    return ts_df

load_data()