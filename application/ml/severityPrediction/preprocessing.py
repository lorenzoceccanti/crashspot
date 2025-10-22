import numpy as np
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
    
    def _translate_week_day_instances(self):
        # week_day
        self.df["week_day"] = self.df["week_day"].str.rstrip()
        week_day_map = {
            "Domingo": "sunday",
            "Sábado": "saturday",
            "Sexta": "friday",
            "Quinta": "thursday",
            "Quarta": "wednesday",
            "Terça": "tuesday",
            "Segunda": "monday",
            'domingo': "sunday",
            'sábado': 'saturday',
            'sexta-feira': 'friday',
            'segunda-feira': 'monday',
            'quinta-feira': 'thursday',
            'quarta-feira': 'wednesday',
            'terça-feira': 'tuesday',
        }
        self.df["week_day"] = self.df["week_day"].replace(week_day_map)

    def _translate_cause_of_accident(self):
        # cause_of_accident
        cause_of_accident_map = {
            "Reação tardia ou ineficiente do condutor": "Driver's lack of reaction",
            "Falta de atenção": "Driver's lack of reaction",
            "Acessar a via sem observar a presença dos outros veículos": "Acessing the road without seeing the presence of other vehicles",
            "Condutor deixou de manter distância do veículo da frente": "Driver failed to keep distance from the vehicle in front",
            "Manobra de mudança de faixa": "Driver changed the lane illegally",
            "Velocidade Incompatível": "Incompatible velocity",
            "Transitar na contramão": "Driver was in the opposite direction",
            "Ingestão de álcool pelo condutor": "Alcohol ingestion by the driver",
            "Demais falhas mecânicas ou elétricas": "Electrical or mechanical flaws",
            "Ultrapassagem Indevida": "Driver changed the lane illegally",
            "Conversão proibida": "Prohibited conversion",
            "Avarias e/ou desgaste excessivo no pneu": "Excessive use of the car's tire",
            "Condutor Dormindo": "Driver was sleeping",
            "Desrespeitar a preferência no cruzamento": "Driver broke the laws of transit",
            "Trafegar com motocicleta (ou similar) entre as faixas":"Traffic with a motorcycle (or similar) between lanes",
            "Ausência de reação do condutor": "Driver's lack of reaction",
            "Outras": "Other",
            "Acesso irregular": "Irregular access",
            "Entrada inopinada do pedestre": "Unexpected pedestrian entry",
            "Pedestre andava na pista":"Pedestrian was walking in the road",
            "Chuva": "Rain",
            "Não guardar distância de segurança": "Driver failed to keep distance from the vehicle in front",
            "Velocidade incompatível": "Incompatible velocity",
            "Defeito mecânico em veículo": "Mechanical loss/defect of vehicle",
            "Desobediência à sinalização": "Driver broke the laws of transit",
            "Ultrapassagem indevida":  "Driver changed the lane illegally",
            "Ingestão de álcool":  "Alcohol ingestion by the driver",
            "Animais na Pista":  "Animals on the road",
            "Dormindo":  "Driver was sleeping",
            "Pista Escorregadia": "Slippery track",
            "Pedestre cruzava a pista fora da faixa": "Pedestrian was crossing the road outside of the crosswalk",
            "Defeito na via":  "Road's defect",
            "Acumulo de água sobre o pavimento": "Accumulation of water on the road",
            "Mal súbito do condutor": "Driver had a cardiac attack",
            "Transitar no Acostamento": "Driving on the breakdown lane",
            "Retorno proibido": "Prohibited conversion",
            "Frear bruscamente": "Abrupt use of the car's brake",
            "Objeto estático sobre o leito carroçável":"Static object on the drainage gate",
            "Problema com o freio": "Car's brake problem",
            "Condutor desrespeitou a iluminação vermelha do semáforo": "Driver disrespected the red traffic light",
            "Carga excessiva e/ou mal acondicionada": "Excessive load/cargo",
            "Estacionar ou parar em local proibido": "Stopping at a prohibited place",
            "Ausência de sinalização": "Absence of sinalization",
            "Suicídio (presumido)": "suicide (presumed)",
            "Pista esburacada": "Unlevel track",
            "Acumulo de óleo sobre o pavimento": "Oil accumulation on the road",
            "Deficiência do Sistema de Iluminação/Sinalização":"Deficiency of vehicle's sinalization/ilumination system",
            "Curva acentuada": "Curvy road",
            "Acumulo de areia ou detritos sobre o pavimento": "Road had lots of sand/wreckage",
            "Pedestre - Ingestão de álcool/ substâncias psicoativas": "Alcohol and/or drug ingestion by the pedestrian",
            "Acostamento em desnível":"Stopping at a prohibited place",
            "Afundamento ou ondulação no pavimento": "Sinking or ondulation in the pavement",
            "Iluminação deficiente": "Poor ilumination (of the road)",
            "Condutor usando celular": "Driver using cellphone",
            "Neblina": "Fog",
            "Demais Fenômenos da natureza": "Natural phenomena",
            "Ingestão de substâncias psicoativas pelo condutor": "Driver was using drugs",
            "Fumaça": "Road condition",
            "Falta de acostamento": "Sinking or ondulation in the pavement",
            "Área urbana sem a presença de local apropriado para a travessia de pedestres": "Urban area without appropriate pedestrian walking",
            "Sinalização mal posicionada":"Inadequate sinalization of the road",
            "Transtornos Mentais (exceto suicidio)":"mental disorder (except suicide)",
            "Falta de elemento de contenção que evite a saída do leito carroçável": "Road defect",
            "Problema na suspensão":"Car's suspension system with problems",
            "Restrição de visibilidade em curvas horizontais": "Visibility restriction",
            "Desvio temporário": "Road works (in maintenance)",
            "Participar de racha": "Major traffic offense",
            "Declive acentuado":"Unlevel track",
            "Faixas de trânsito com largura insuficiente": "Road defect",
            "Deixar de acionar o farol da motocicleta (ou similar)": "Minor traffic offense",
            "Modificação proibida": "Veichle human fault",
            "Restrição de visibilidade em curvas verticais": "Visibility restriction",
            "Semáforo com defeito": "Road condition",
            "Transitar na calçada": "Pedestrian involved",
            "Faróis desregulados": "Veichle human fault",
            "Sistema de drenagem ineficiente": "Accumulation of water on the road",
            "Sinalização encoberta": "Road defect",
            "Redutor de velocidade em desacordo": "High speed"
        }
        self.df["cause_of_accident"] = self.df["cause_of_accident"].replace(cause_of_accident_map)

    def _translate_type_of_accident(self):
        type_of_accident_map = {
            "Colisão traseira": "Rear-end collision",
            "Colisão lateral": "Broadside collision",
            "Saída de Pista": "Run-off-road",
            "Colisão Transversal": "Side impact collision",
            "Colisão transversal": "Side impact collision",
            "Colisão lateral mesmo sentido": "Side collision (same direction)",
            "Saída de leito carroçável": "Run-off-road",
            "Colisão frontal": "Head-on collision",
            "Capotamento": "Rollover",
            "Colisão com objeto": "Collision with object",
            "Colisão com objeto fixo": "Collision with fixed object",
            "Colisão lateral sentido oposto": "Side collision (opposite direction)",
            "Atropelamento de Pedestre": "Pedestrian collision",
            "Atropelamento de pessoa": "Pedestrian collision",
            "Engavetamento": "Chain reaction crash (pile-up)",
            "Tombamento": "Overturn",
            "Colisão com bicicleta": "Collision with moving object",
            "Atropelamento de animal": "Animal collision",
            "Atropelamento de Animal": "Animal collision",
            "Queda de motocicleta / bicicleta / veículo": "Fall of veichle occupant",
            "Queda de ocupante de veículo": "Fall of veichle occupant",
            "Colisão com objeto móvel": "Collision with object",
            "Danos Eventuais": "Minor incidental damage",
            "Derramamento de Carga": "Cargo spill",
            "Derramamento de carga": "Cargo spill",
            "Incêndio": "Veichle fire",
            "Eventos atípicos": "Unusual event",
            "Sinistro pessoal de trânsito": "Personal traffic accident"
        }
        self.df["type_of_accident"] = self.df["type_of_accident"].replace(type_of_accident_map)

    def _translate_veichle_type(self):
        vehicle_type_map = {
            "Automóvel": "Car",
            "Motocicleta": "Motorcycle",
            "Motocicletas": "Motorcycle",
            "Semireboque": "Semi-trailer",
            "Caminhonete": "Pickup truck",
            "Caminhão-trator": "Tractor-trailer truck",
            "Caminhão-Trator": "Tractor-trailer truck",
            "Caminhão": "Truck",
            "Caminhão-Tanque": "Truck",
            "Ônibus": "Bus",
            "Bonde / Trem": "Tram / Train",
            "Camioneta": "Van",
            "Motoneta": "Scooter",
            "Utilitário": "Utility vehicle",
            "Bicicleta": "Bicycle",
            "Micro-ônibus": "Minibus",
            "Microônibus": "Minibus",
            "Reboque": "Trailer",
            "Outros": "Others",
            "Ciclomotor": "Moped",
            "Carroça-charrete": "Cart-wagon",
            "Carroça": "Cart-wagon",
            "Trator de rodas": "Wheeled tractor",
            "Motor-casa": "Motorhome",
            "Triciclo": "Tricycle",
            "Trem-bonde": "Tram",
            "Trator de esteira": "Crawler tractor",
            "Trator de esteiras": "Crawler tractor",
            "Trator misto": "Backhoe loader",
            "Carro de mão": "Wheelbarrow",
            "Carro-de-mao": "Wheelbarrow",
            "Chassi-plataforma": "Chassis platform",
            "Quadriciclo": "Quadricycle",
            "Não identificado": pd.NA,
            "(null)": pd.NA
        }
        self.df["veichle_type"] = self.df["veichle_type"].replace(vehicle_type_map)

    def _preprocess_veichle_brand(self):
        self.df['veichle_brand'] = self.df["veichle_brand"].replace({
            "Não Informado/Não Informado": pd.NA,
            "Não Informado/Não Informado/Não Informado": pd.NA,
            "NA/NA": pd.NA,
            "(null)": pd.NA
        })    
    
    def _preprocess_veichle_manufacturing_year(self):
        self.df['veichle_manufacturing_year'] = self.df["veichle_manufacturing_year"].replace(0,pd.NA)
        self.df['veichle_manufacturing_year'] = self.df["veichle_manufacturing_year"].replace("    ",pd.NA)
        self.df['veichle_manufacturing_year'] = self.df["veichle_manufacturing_year"].replace("(null)", pd.NA)

        self.df["veichle_manufacturing_year"] = self.df["veichle_manufacturing_year"].astype("Float64")
    
    def _preprocess_person_age(self):
        self.df.loc[self.df["person_age"] > 125.0, "person_age"] = pd.NA
        self.df["person_age"] = self.df["person_age"].replace(-1.0, pd.NA)
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self._detect_encoding()
        self._translate_column_name()

        self._translate_week_day_instances()
        self._translate_type_of_accident()
        self._preprocess_veichle_manufacturing_year()
        self._preprocess_person_age()
        self._translate_veichle_type()

        # If we have passed a test-set compliant dataset, we don't require
        # any translation/preprocessing
        if not 'general_cause_of_accident' in self.df.columns:
            self._translate_cause_of_accident()
            # Creating the general_cause_of_accident attribute

        if not 'general_veichle_brand' in self.df.columns:
            self._preprocess_veichle_brand()
        
    def get_df(self):
        return self.df