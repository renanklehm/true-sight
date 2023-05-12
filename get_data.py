import pandas as pd
import database_config as cfg
from utils import setup_connection
from sqlalchemy import text

def get_data():
    _, con = setup_connection(cfg.dw_path)
    query = text("""
            SELECT
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
    df['DAT_FATO'] = pd.to_datetime(df['DAT_FATO'])
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
    ts_df = df.pivot(index='ID_STRING', columns='DAT_FATO', values='QTD_ITEM')
    ts_df = ts_df.reset_index()
    ts_df = ts_df.melt(id_vars='ID_STRING', var_name='DAT_FATO', value_name='QTD_ITEM')
    ts_df.fillna(0.0, inplace=True)
    corpus = ts_df["ID_STRING"].unique()
    return ts_df, corpus