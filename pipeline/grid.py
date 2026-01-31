import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def _load_config(self,path):
        BASE_DIR = Path(__file__).resolve().parent.parent
        CONFIG_PATH = BASE_DIR/ path

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

        return(config)

    def __init__(self,config_path='data/config.json'):
        self.config =self._load_config(config_path)

        self.hx=self.config['config']['hx']
        self.hy=self.config['config']['hy']
        self.ht=self.config['config']['ht']
        self.width_m=self.config['config']['width_m']
        self.height_m=self.config['config']['height_m']

        self.nx = int(self.width_m/self.hx)+1
        self.ny = int(self.height_m/self.hy)+1

        self.material_grid = np.zeros((self.ny,self.nx))

        self._build_grid()

    def _build_grid(self):
        geometry = self.config['geometry']

        for item in geometry:
            mat_id = item['type']
            x_range = item['x']
            y_range = item['y']


            col_start = int(round(x_range[0] / self.hx))
            col_end = int(round(x_range[1] / self.hx))
            row_start = int(round(y_range[0] / self.hx))
            row_end = int(round(y_range[1] / self.hx))


            if col_end == self.nx - 1:
                col_end = self.nx

            if row_end == self.ny - 1:
                row_end = self.ny

            self.material_grid[row_start:row_end, col_start:col_end] = mat_id


    def get_material_matrix(self):
        return self.material_grid



