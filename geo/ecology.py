from . import np
from . import math

def infer_biomes(elevations_m: np.array, temperatures_c: np.array, precipitation_mm: np.array, saturation: np.array, biome_mapping: dict):
    '''
    Ocean = 0
    '''
    biomes = np.full(len(elevations_m), [-1], dtype=int)
    means = np.array([list(biome['features']['mean'].values()) for id, biome in biome_mapping.items() if id != 0])
    std_devs = np.array([list(biome['features']['std_dev'].values()) for id, biome in biome_mapping.items() if id != 0])

    for idx, elev in enumerate(elevations_m):
        if elev <= 0:
            biomes[idx] = 0
            continue

        vals = [temperatures_c[idx], precipitation_mm[idx], saturation[idx]]
        i = math.gaussian_classificator(means, std_devs, vals)
        biomes[idx] = i[0][0] + 1

    return biomes


sample_biome_colors = {
    0: [55, 157, 204, 1],
    1: [225, 180, 114, 1],
    2: [203, 182, 107, 1],
    3: [144, 120, 84, 1],
    4: [165, 186, 94, 1],
    5: [126, 189, 81, 1],
    6: [40, 164, 46, 1],
    7: [48, 196, 55, 1],
    8: [31, 132, 36, 1],
    9: [121, 167, 70, 1],
    10: [51, 96, 53, 1],
    11: [184, 217, 185, 1],
    12: [216, 228, 224, 1],
    13: [179, 186, 147, 1],
    14: [123, 142, 106, 1],
    15: [80, 103, 60, 1]
}

def normalize_biome_rgba(biome_colors):
    rgba = {}
    for id, color in biome_colors.items():
        r, g, b, a = color
        rgba[id] = (r/255, g/255, b/255, a)

    return rgba

_sample_biome_mapping = {
    0: {"name": "ocean", "features" : {}},
    1: {"name": "desert", "features": {"t": 32, "p": 100, "t_spread": 7, "p_spread": 200, "s": .1, "s_spread": .1}},
    2: {"name": "savannah", "features": {"t": 28, "p": 600, "t_spread": 6, "p_spread": 300, "s": .15, "s_spread": .1}},
    3: {"name": "cold desert", "features": {"t": 5, "p": 200, "t_spread": 5, "p_spread": 200, "s": .1, "s_spread": .1}},
    4: {"name": "steppe", "features": {"t": 8, "p": 500, "t_spread": 5, "p_spread": 200, "s": .15, "s_spread": .1}},
    5: {"name": "grassland", "features": {"t": 18, "p": 700, "t_spread": 6, "p_spread": 200, "s": .25, "s_spread": .1}},
    6: {"name": "rainforest", "features": {"t": 27, "p": 3000, "t_spread": 5, "p_spread": 600, "s": .45, "s_spread": .15}},
    7: {"name": "temperate rainforest", "features": {"t": 15, "p": 3000, "t_spread": 5, "p_spread": 600, "s": .3, "s_spread": .1}},
    8: {"name": "temperate deciduous forest", "features": {"t": 12, "p": 1500, "t_spread": 6, "p_spread": 400, "s": .25, "s_spread": .15}},
    9: {"name": "temperate forest", "features": {"t": 18, "p": 600, "t_spread": 6, "p_spread": 250, "s": .25, "s_spread": .15}},
    10: {"name": "taiga", "features": {"t": 3, "p": 600, "t_spread": 5, "p_spread": 250, "s": .3, "s_spread": .15}},
    11: {"name": "tundra", "features": {"t": -5, "p": 400, "t_spread": 5, "p_spread": 200, "s": .4, "s_spread": .15}},
    12: {"name": "glacier", "features": {"t": -10, "p": 300, "t_spread": 4, "p_spread": 400, "s": .2, "s_spread": .15}},
    13: {"name": "bog", "features" : {"t": -5, "p": 400, "t_spread": 5, "p_spread": 200, "s": .95, "s_spread": .3}},
    14: {"name": "marsh", "features" : {"t": 18, "p": 700, "t_spread": 6, "p_spread": 200, "s": .95, "s_spread": .3}},
    15: {"name": "wetland", "features" : {"t": 27, "p": 3000, "t_spread": 5, "p_spread": 600, "s": .95, "s_spread": .3}}
}

sample_biome_mapping ={0: {'name': 'ocean', 'features': {}},
 1: {'name': 'desert',
  'features': {'mean': {'t': 32, 'p': 100, 's': 0.1},
   'std_dev': {'t_spread': 7, 'p_spread': 200, 's_spread': 0.1}}},
 2: {'name': 'savannah',
  'features': {'mean': {'t': 28, 'p': 600, 's': 0.15},
   'std_dev': {'t_spread': 6, 'p_spread': 300, 's_spread': 0.1}}},
 3: {'name': 'cold desert',
  'features': {'mean': {'t': 5, 'p': 100, 's': 0.1},
   'std_dev': {'t_spread': 5, 'p_spread': 200, 's_spread': 0.1}}},
 4: {'name': 'steppe',
  'features': {'mean': {'t': 8, 'p': 500, 's': 0.15},
   'std_dev': {'t_spread': 5, 'p_spread': 200, 's_spread': 0.1}}},
 5: {'name': 'grassland',
  'features': {'mean': {'t': 20, 'p': 600, 's': 0.25},
   'std_dev': {'t_spread': 8, 'p_spread': 200, 's_spread': 0.15}}},
 6: {'name': 'rainforest',
  'features': {'mean': {'t': 27, 'p': 2500, 's': 0.65},
   'std_dev': {'t_spread': 5, 'p_spread': 400, 's_spread': 0.3}}},
 7: {'name': 'temperate rainforest',
  'features': {'mean': {'t': 15, 'p': 2500, 's': 0.3},
   'std_dev': {'t_spread': 5, 'p_spread': 200, 's_spread': 0.1}}},
 8: {'name': 'temperate deciduous forest',
  'features': {'mean': {'t': 12, 'p': 1125, 's': 0.3},
   'std_dev': {'t_spread': 6, 'p_spread': 400, 's_spread': 0.15}}},
 9: {'name': 'temperate forest',
  'features': {'mean': {'t': 18, 'p': 1125, 's': 0.3},
   'std_dev': {'t_spread': 6, 'p_spread': 250, 's_spread': 0.15}}},
 10: {'name': 'taiga',
  'features': {'mean': {'t': 3, 'p': 600, 's': 0.3},
   'std_dev': {'t_spread': 5, 'p_spread': 300, 's_spread': 0.15}}},
 11: {'name': 'tundra',
  'features': {'mean': {'t': -5, 'p': 400, 's': 0.2},
   'std_dev': {'t_spread': 5, 'p_spread': 200, 's_spread': 0.15}}},
 12: {'name': 'glacier',
  'features': {'mean': {'t': -10, 'p': 400, 's': 0.2},
   'std_dev': {'t_spread': 4, 'p_spread': 400, 's_spread': 0.15}}},
 13: {'name': 'bog',
  'features': {'mean': {'t': -5, 'p': 1900, 's': 1.15},
   'std_dev': {'t_spread': 5, 'p_spread': 150, 's_spread': 0.16}}},
 14: {'name': 'marsh',
  'features': {'mean': {'t': 18, 'p': 1900, 's': 1.15},
   'std_dev': {'t_spread': 6, 'p_spread': 150, 's_spread': 0.16}}},
 15: {'name': 'wetland',
  'features': {'mean': {'t': 27, 'p': 2500, 's': 1.15},
   'std_dev': {'t_spread': 5, 'p_spread': 150, 's_spread': 0.16}}}}