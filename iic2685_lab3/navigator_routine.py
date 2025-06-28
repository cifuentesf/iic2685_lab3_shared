
# Fns útiles para localization pkg
import os
from ament_index_python.packages import get_package_share_directory
import yaml
import numpy as np
import time

class MapLoader:
    """
    Clase para cargar el mapa desde un archivo YAML.
    """
    
    def __init__(self, map_file):
        self.map_file = map_file
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.load_map()
        if self.map_data is None:
            raise ValueError(f"El mapa {self.map_file} no se pudo cargar. Verifica el archivo YAML.")
        else:
            self.save_probability_map(self.map_data, 'probability_map.yaml')


    def load_map(self):
        """
        Carga el mapa desde el archivo YAML.
        """
        map_path = os.path.join(get_package_share_directory('iic2685_lab3'), 'maps', self.map_file)
        with open(map_path, 'r') as file:
            map_yaml = yaml.safe_load(file)
            self.map_data = map_yaml['data']
            self.map_resolution = map_yaml['resolution']
            self.map_origin = map_yaml['origin']

    def get_map_info(self):
        """
        Retorna la información del mapa.
        """
        if self.map_data is None:
            raise ValueError("El mapa no ha sido cargado. Llama a load_map() primero.")
        
        return {
            'data': self.map_data,
            'resolution': self.map_resolution,
            'origin': self.map_origin
        }
    
    def save_probability_map(self, probability_map, output_file):
        """
        Guarda el mapa de probabilidades en un archivo YAML.
        Args:
            probability_map: Mapa de probabilidades a guardar.
            output_file: Nombre del archivo de salida.
        """
        x,y = np.shape(probability_map)
        for i in range(x):
            for j in range(y):
                if probability_map[i][j] < 0.01:
                    probability_map[i][j] = 0.01
                elif probability_map[i][j] > 0.99:
                    probability_map[i][j] = 0.99
                else:
                    probability_map[i][j] = round(probability_map[i][j], 2)
        # Guardar el mapa de probabilidades en un archivo YAML
        output_path = os.path.join(get_package_share_directory('iic2685_lab3'), 'maps', output_file)
        with open(output_path, 'w') as file:
            yaml.dump(probability_map, file)


def bayes_probability(p_hit, p_miss, z_hit, z_miss, z):
    """
    Calcula la probabilidad bayesiana de una medición.
    Usando la fórmula de la clase 4.1:
    P(x_state| z_sense_state) = P(z_sense_state| x_state) * P(x_state) / P(z_sense_state)
    Args:
        p_hit: Probabilidad de que la medición sea correcta.
        p_miss: Probabilidad de que la medición sea incorrecta.
        z_hit: Probabilidad de la medición correcta.
        z_miss: Probabilidad de la medición incorrecta.
        z: Medición del sensor.
    Ademas:
        Se actualiza la probabilidad de la medición correcta y la incorrecta. 
        Según cuantas veces se ha medido el sensor.
    Retorna la probabilidad total de la medición.
    """
    