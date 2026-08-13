'''

Tokenizing CSV full in memory using Polars.

This is the second step in processing, we are finally getting a Parquet file
wich is an column-oriente data format designed for efficient data storage
and retrieval wich gives compression and encoding schemes with enhanced 
performance.

'''

import polars as pl
import torch


df = pl.read_parquet("./preprocesing/csic_cleaned.parquet")


'''
    Crear un tensor con las columnas en el dataset.
    Se define ademas la cantidad maxima de caracteres por cada 
    una de las columnas.
'''

y = torch.tensor(df["Label"].to_list(), dtype=torch.float32)


cols_texto = ['Method', 'User-Agent', 'Pragma', 
              'Cache-Control', 'Accept', 'Accept-encoding', 'Accept-charset', 
              'language', 'host', 'cookie', 'content-type', 'connection', 'content',
                'classification', 'URL', 'URL-encoded' ]


max_lens = {
    'Method' :6 ,
    'User-Agent':  100, 
    'Pragma' : 50,
    'Cache-Control': 50 ,
    'Accept': 200,
    'Accept-encoding': 200,
    'Accept-charset': 100, 
    'language': 5, 
    'host': 120, 
    'cookie': 250, 
    'content-type': 225,
    'connection': 20 ,
    'content': 250 ,
    'URL': 250 , 
    'URL-encoded': 250 
}

