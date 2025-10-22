import io
import os
import csv
import chardet
import pandas as pd

class Preprocessing:

    def _detect_encoding(self):
        # 1) We read the first 1kb (as bytes) to detect the encoding of the csv file
        bytes_data = self.csv_file.getvalue()
        chardet_result = chardet.detect(bytes_data[:1000])
        encoding = chardet_result['encoding']
        if encoding == 'ascii':
            encoding = 'utf-8'

        # 2) We detect the separator
        text = bytes_data.decode(encoding, errors = 'replace')
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text.splitlines()[0] + "\n" + text.splitlines()[1])
        delimeter = dialect.delimiter
        
        # This is done in order to bring back the file pointer to the beginning of the csv file
        # otherwise pandas will return an empty dataframe
        self.csv_file.seek(0)
        self.df = pd.read_csv(self.csv_file, encoding=encoding, sep=delimeter)
    
    def _translate_column_name(self):
        translations = {
            'data_inversa': 'date',
            'dia_semana': 'week_day',
            'horario': 'hour',
            'uf': 'state',
            'br': 'road_id',
            'municipio': 'city',
            'causa_acidente': 'cause_of_accident',
            'tipo_acidente': 'type_of_accident',
            'tipo_veiculo': 'veichle_type',
            'marca': 'veichle_brand',
            'ano_fabricacao_veiculo': 'veichle_manufacturing_year',
            'idade': 'person_age'
        }
        # We iterate all the columns of the dataframe. If a mapping exists, 
        # we get the value associated to the key in the dictionary (so the translation)
        # If the key doesn't exists, so a translation is not there, we stay with the original column name
        self.df.columns = [translations.get(col, col) for col in self.df.columns]

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self._detect_encoding()
        self._translate_column_name()
    
    def get_df(self):
        return self.df