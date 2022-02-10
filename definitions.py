import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is Project Root
CT_IMAGES_DIR = os.path.join(ROOT_DIR, 'CT scans from Eric')
DENSITY = os.path.join(ROOT_DIR, 'Density.csv')
AVERAGE_HU = os.path.join(ROOT_DIR, 'Average HU.csv')
MATERIAL_ATT_COEFF = os.path.join(ROOT_DIR, 'Materials attenuation coefficient raw')
