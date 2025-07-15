import numpy as np

SMPL_TO_SMPLX_MAPPING = './data/smpl_to_smplx.pkl'

AFFORD_LIST_PIAD = np.array(['grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support', 'wrapgrasp', 'pour', 
                             'move', 'display', 'push', 'listen', 'wear', 'press', 'cut', 'stab'])

AFFORD_LIST_LEMON = np.array(['grasp', 'lift', 'open', 'lay', 'sit', 'support', 'wrapgrasp', 'pour', 
                              'move', 'pull', 'listen', 'press', 'cut', 'stab', 'ride', 'play', 'carry'])
                       
AFFORD_PROB_PIAD = {
    'Bag': {'open': 0.2, 'lift': 0.3, 'grasp': 0.15, 'contain': 0.2},
    'Bed': {'lay': 0.3, 'sit': 0.5},
    'Bottle': {'wrapgrasp': 0.2, 'open': 0.2, 'pour': 0.2, 'contain': 0.2},
    'Bowl': {'wrapgrasp': 0.2, 'pour': 0.2, 'contain': 0.2},
    'Chair': {'sit': 0.2, 'move': 0.3},
    'Clock': {'display': 0.2},
    'Dishwasher': {'open': 0.3, 'contain': 0.2},
    'Display': {'display': 0.2},
    'Door': {'open': 0.2, 'push': 0.2},
    'Earphone': {'grasp': 0.3, 'listen': 0.3},
    'Faucet': {'open': 0.2, 'grasp': 0.2},
    'Hat': {'wear': 0.1, 'grasp': 0.3},
    'Keyboard': {'press': 0.1},
    'Knife': {'grasp': 0.1, 'stab': 0.1, 'cut': 0.1},
    'Laptop': {'press': 0.2, 'display': 0.2},
    'Microwave': {'open': 0.1, 'contain': 0.2},
    'Mug': {'wrapgrasp': 0.2, 'grasp': 0.2, 'pour': 0.2, 'contain': 0.2},
    'Refrigerator': {'open': 0.2, 'contain': 0.2}, 
    'Scissors': {'grasp': 0.1, 'cut': 0.1, 'stab': 0.1},
    'StorageFurniture': {'open': 0.2, 'contain': 0.2},
    'Table': {'move': 0.2, 'support': 0.2},
    'TrashCan': {'open': 0.2, 'contain': 0.2, 'pour': 0.2},
    'Vase': {'wrapgrasp': 0.25, 'contain': 0.25},
}

AFFORD_PROB_LEMON = {
    'Backpack': {'carry': 0.1},
    'Bag': {'lift': 0.2, 'grasp': 0.2},
    'Baseballbat': {'grasp': 0.2},
    'Bed': {'lay': 0.3, 'sit': 0.5},
    'Bicycle': {'ride': 0.2},
    'Bottle': {'wrapgrasp': 0.2, 'open': 0.2, 'pour': 0.2},
    'Bowl': {'wrapgrasp': 0.3},
    'Chair': {'sit': 0.2, 'move': 0.3},
    'Earphone': {'listen': 0.3},
    'Guitar': {'play': 0.2},
    'Keyboard': {'press': 0.2},
    'Knife': {'grasp': 0.1, 'cut': 0.1, 'stab': 0.1},
    'Motorcycle': {'ride': 0.2},
    'Mug': {'wrapgrasp': 0.2, 'grasp': 0.3},
    'Scissors': {'grasp': 0.1, 'cut': 0.1},
    'Skateboard': {'support': 0.2},
    'Suitcase': {'pull': 0.1},
    'Surfboard': {'carry': 0.2, 'support': 0.1},
    'Tennisracket': {'grasp': 0.1},
    'Umbrella': {'grasp': 0.1},
    'Vase': {'wrapgrasp': 0.25},
}

# PIAD----
# Bag  Bottle  Chair  Dishwasher  Door      Faucet  Keyboard  Laptop     Mug           Scissors          Table     Vase
# Bed  Bowl    Clock  Display     Earphone  Hat     Knife     Microwave  Refrigerator  StorageFurniture  TrashCan

# LEMON ---
# Backpack/  Baseballbat/  Bicycle/  Bowl/   Earphone/  Keyboard/  Motorcycle/  Scissors/    Suitcase/   Tennisracket/  Vase/
# Bag/       Bed/          Bottle/   Chair/  Guitar/    Knife/     Mug/         Skateboard/  Surfboard/  Umbrella/

# PICO ---
# apple/           bed/      bowl/        chair/         fork/        knife/       remote/      spoon/        tennis_racket/
# backpack/        bench/    broccoli/    couch/         frisbee/     laptop/      sandwich/    sports_ball/  toothbrush/
# banana/          bicycle/  cake/        cup/           hair_drier/  motorcycle/  scissors/    suitcase/     wine_glass/
# baseball_bat/    book/     carrot/      dining_table/  handbag/     mouse/       skateboard/  surfboard/
# baseball_glove/  bottle/   cell_phone/  donut/         keyboard/    pizza/       snowboard/   teddy_bear/


# 1.  `apple` (PICO)
# 2.  `backpack` (LEMON, PICO)
# 3.  `bag` (PIAD, LEMON)
# 4.  `banana` (PICO)
# 5.  `baseballbat` (LEMON, PICO - `baseball_bat`)
# 6.  `baseballglove` (PICO - `baseball_glove`)
# 7.  `bed` (PIAD, LEMON, PICO)
# 8.  `bench` (PICO)
# 9.  `bicycle` (LEMON, PICO)
# 10. `book` (PICO)
# 11. `bottle` (PIAD, LEMON, PICO)
# 12. `bowl` (PIAD, LEMON, PICO)
# 13. `broccoli` (PICO)
# 14. `cake` (PICO)
# 15. `carrot` (PICO)
# 16. `cellphone` (PICO - `cell_phone`)
# 17. `chair` (PIAD, LEMON, PICO)
# 18. `clock` (PIAD)
# 19. `couch` (PICO)
# 20. `cup` (PICO)
# 21. `diningtable` (PICO - `dining_table`)
# 22. `dishwasher` (PIAD)
# 23. `display` (PIAD)
# 24. `donut` (PICO)
# 25. `door` (PIAD)
# 26. `earphone` (PIAD, LEMON)
# 27. `faucet` (PIAD)
# 28. `fork` (PICO)
# 29. `frisbee` (PICO)
# 30. `guitar` (LEMON)
# 31. `hairdrier` (PICO - `hair_drier`)
# 32. `handbag` (PICO)
# 33. `hat` (PIAD)
# 34. `keyboard` (PIAD, LEMON, PICO)
# 35. `knife` (PIAD, LEMON, PICO)
# 36. `laptop` (PIAD, PICO)
# 37. `microwave` (PIAD)
# 38. `motorcycle` (LEMON, PICO)
# 39. `mouse` (PICO)
# 40. `mug` (PIAD, LEMON)
# 41. `pizza` (PICO)
# 42. `refrigerator` (PIAD)
# 43. `remote` (PICO)
# 44. `sandwich` (PICO)
# 45. `scissors` (PIAD, LEMON, PICO)
# 46. `skateboard` (LEMON, PICO)
# 47. `snowboard` (PICO)
# 48. `spoon` (PICO)
# 49. `sportsball` (PICO - `sports_ball`)
# 50. `storagefurniture` (PIAD - `StorageFurniture`)
# 51. `suitcase` (LEMON, PICO)
# 52. `surfboard` (LEMON, PICO)
# 53. `table` (PIAD)
# 54. `teddybear` (PICO - `teddy_bear`)
# 55. `tennisracket` (LEMON - `Tennisracket`, PICO - `tennis_racket`)
# 56. `toothbrush` (PICO)
# 57. `trashcan` (PIAD - `TrashCan`)
# 58. `umbrella` (LEMON)
# 59. `vase` (PIAD, LEMON)
# 60. `wineglass` (PICO - `wine_glass`)

OBJS_VIEW_DICT = {
    '4MV-XY_Rand': {'order': 'rand',
                  'grid_size': np.array([1, 2, 2]),
                  'mask_size': 512,
                  'folder': 'rendered_points_4viewsOld',
                  'names': np.array([[['frontleft', 'frontright'],
                                     ['backleft', 'backright']]]),
                  'ignore_keywords': ['pour'],
                  'cam_params': {'frontleft': None, 'frontright': None, 'backleft': None, 'backright': None},
            },
    '4MV-XY_Fix': {'order': 'fix',
                  'grid_size': np.array([1, 2, 2]),
                  'mask_size': 512,
                  'folder': 'rendered_points_0917',
                  'names': np.array([[['frontleft', 'frontright'],
                                     ['backleft', 'backright']]]),
                  'ignore_keywords': ['Refrigerator', 'Baseballbat'],
                  'cam_params': {'frontleft': None, 'frontright': None, 'backleft': None, 'backright': None},
                },
    '4MV-XY_Rand': {'order': 'rand',
                   'grid_size': np.array([1, 2, 2]),
                   'mask_size': 512,
                   'folder': 'rendered_points_0917',
                   'names': np.array([[['frontleft', 'frontright'],
                                      ['backleft', 'backright']]]),
                   'ignore_keywords': ['Refrigerator', 'Baseballbat'],
                   'cam_params': {'frontleft': None, 'frontright': None, 'backleft': None, 'backright': None},
                },
    '4MV-Z_Fix': {'order': 'fix',
                   'grid_size': np.array([4, 1, 1]),
                   'mask_size': 512,
                   'folder': 'rendered_points_0917',
                   'names': np.array([[['frontleft']], 
                                      [['frontright']],
                                      [['backleft',]],
                                      [[ 'backright']]]),
                   'ignore_keywords': ['Refrigerator', 'Baseballbat'],
                   'cam_params': {
                       'frontleft':  [2., 45., 315., 0., 0.],
                       'frontright': [2., 45., 45., 0., 0.],
                       'backleft':   [2., 330., 135., 0., 0.],
                       'backright':  [2., 330., 225., 0., 0.]
                   }
                },
    '4MV-Z_HM': {'order': 'fix',
                 'grid_size': np.array([4, 1, 1]),
                 'mask_size': 1024,
                 'folder': 'rendered_points_heatmap_1025',
                 'names': np.array([[['frontleft']], 
                                    [['frontright']],
                                    [['backleft',]],
                                    [['backright']]]),
                 'ignore_keywords': [],
                 'cam_params': {
                    'frontleft':  [2., 45., 315., 0., 0.],
                    'frontright': [2., 45., 45., 0., 0.],
                    'backleft':   [2., 330., 135., 0., 0.],
                    'backright':  [2., 330., 225., 0., 0.]
                 }
                },
    '4MV-Z_HM1': {'order': 'fix',
                 'grid_size': np.array([4, 1, 1]),
                 'mask_size': 1024,
                 'folder': 'rendered_points_heatmap_1102',
                 'names': np.array([[['frontleft']], 
                                    [['frontright']],
                                    [['backleft',]],
                                    [['backright']]]),
                 'ignore_keywords': [],
                 'cam_params': {
                    'frontleft':  [2., 45., 315., 0., 0.],
                    'frontright': [2., 45., 45., 0., 0.],
                    'backleft':   [2., 330., 135., 0., 0.],
                    'backright':  [2., 330., 225., 0., 0.]
                 }
                },
    '4MV-Z_HM2': {'order': 'fix',
                 'grid_size': np.array([4, 1, 1]),
                 'mask_size': 1024,
                 'folder': 'rendered_points_heatmap_AP1K0_1104',
                 'names': np.array([[['frontleft']], 
                                    [['frontright']],
                                    [['backleft',]],
                                    [['backright']]]),
                 'ignore_keywords': [],
                 'cam_params': {
                    'frontleft':  [2., 45., 315., 0., 0.],
                    'frontright': [2., 45., 45., 0., 0.],
                    'backleft':   [2., 330., 135., 0., 0.],
                    'backright':  [2., 330., 225., 0., 0.]
                 }
                },
    '4MV-Z_HM_MeshInf': {'order': 'fix',
                         'grid_size': np.array([4, 1, 1]),
                         'mask_size': 1024,
                         'names': np.array([[['frontleft']], 
                                            [['frontright']],
                                            [['backleft',]],
                                            [['backright']]]),
                         'ignore_keywords': [],
                         'cam_params': {
                            'frontleft':  [2., 45., 315., 0., 0.],
                            'frontright': [2., 45., 45., 0., 0.],
                            'backleft':   [2., 330., 135., 0., 0.],
                            'backright':  [2., 330., 225., 0., 0.]
                         }
            },
    '4MV-Z_HM_BM': {'order': 'fix',
                      'grid_size': np.array([4, 1, 1]),
                      'mask_size': 1024,
                      'names': np.array([[['frontleft']], 
                                        [['frontright']],
                                        [['backleft',]],
                                        [['backright']]]),
                      'ignore_keywords': [],
                      'mesh_folder': 'lowpoly_mesh_0507',
                      'folder': 'rendered_points_heatmap_1025',
                      'cam_params': {
                            'frontleft':  [2., 45., 315., 0., 0.],
                            'frontright': [2., 45., 45., 0., 0.],
                            'backleft':   [2., 330., 135., 0., 0.],
                            'backright':  [2., 330., 225., 0., 0.]
                     },
                      'mesh_cam_params': {
                            'frontleft':  [1.5, 45., 315., 0., 0.],
                            'frontright': [1.5, 45., 45., 0., 0.],
                            'backleft':   [1.5, 330., 135., 0., 0.],
                            'backright':  [1.5, 330., 225., 0., 0.]
                         }
                        
            },
    '10MV-Z_HM': {'order': 'fix',
                  'grid_size': np.array([10, 1, 1]),
                  'mask_size': 1024,
                  'folder': 'rendered_points_heatmap_1025',
                  'names': np.array([[['frontleft', 'frontright', 'top', 'front', 'left'],
                                       ['backleft', 'backright', 'bottom', 'back', 'right']]]),
                  'ignore_keywords': [],
                  'cam_params': {
                    'frontleft':  [2., 45., 315., 0., 0.],
                    'frontright': [2., 45., 45., 0., 0.],
                    'backleft':   [2., 330., 135., 0., 0.],
                    'backright':  [2., 330., 225., 0., 0.],
                    'top':        [2, 90, 0, 0., 0.],
                    'bottom':     [2, 270, 0, 0., 0.0],
                    'front':      [2, 0, 0, 0., 0.0],
                    'back':       [2, 0, 180, 0., 0.0],
                    'left':       [2, 0, 270, 0., 0.0],
                    'right':      [2, 0, 90, 0., 0.0],
                  },
                },
    '10MV-Z_HM1': {'order': 'fix',
                  'grid_size': np.array([10, 1, 1]),
                  'mask_size': 1024,
                  'folder': 'rendered_points_heatmap_1102',
                  'names': np.array([[['frontleft', 'frontright', 'top', 'front', 'left'],
                                       ['backleft', 'backright', 'bottom', 'back', 'right']]]),
                  'ignore_keywords': [],
                  'cam_params': {
                    'frontleft':  [2., 45., 315., 0., 0.],
                    'frontright': [2., 45., 45., 0., 0.],
                    'backleft':   [2., 330., 135., 0., 0.],
                    'backright':  [2., 330., 225., 0., 0.],
                    'top':        [2, 90, 0, 0., 0.],
                    'bottom':     [2, 270, 0, 0., 0.0],
                    'front':      [2, 0, 0, 0., 0.0],
                    'back':       [2, 0, 180, 0., 0.0],
                    'left':       [2, 0, 270, 0., 0.0],
                    'right':      [2, 0, 90, 0., 0.0],
                  },
                },
    '10MV-Z_HM2': {'order': 'fix',
                  'grid_size': np.array([10, 1, 1]),
                  'mask_size': 1024,
                  'folder': 'rendered_points_heatmap_AP1K0_1104',
                  'names': np.array([[['frontleft', 'frontright', 'top', 'front', 'left'],
                                       ['backleft', 'backright', 'bottom', 'back', 'right']]]),
                  'ignore_keywords': [],
                  'cam_params': {
                    'frontleft':  [2., 45., 315., 0., 0.],
                    'frontright': [2., 45., 45., 0., 0.],
                    'backleft':   [2., 330., 135., 0., 0.],
                    'backright':  [2., 330., 225., 0., 0.],
                    'top':        [2, 90, 0, 0., 0.],
                    'bottom':     [2, 270, 0, 0., 0.0],
                    'front':      [2, 0, 0, 0., 0.0],
                    'back':       [2, 0, 180, 0., 0.0],
                    'left':       [2, 0, 270, 0., 0.0],
                    'right':      [2, 0, 90, 0., 0.0],
                  },
                },
    '10MV-XY_Rand': {'order': 'rand',
                    'grid_size': np.array([1, 2, 2]),
                    'mask_size': 512,
                    'folder': 'rendered_points_0917',
                    'names': np.array([[['frontleft', 'frontright', 'top', 'front', 'left'],
                                       ['backleft', 'backright', 'bottom', 'back', 'right']]]),
                    'ignore_keywords': ['Refrigerator', 'Baseballbat'],
                    'cam_params': {'frontleft': None, 'frontright': None, 'top': None, 'front': None, 'left': None,
                                      'backleft': None, 'backright': None, 'bottom': None, 'back': None, 'right': None}
                },
}

HUMAN_VIEW_DICT = { 
    '4MV-Z_Vitru': 
                    {'order': 'fix',
                    'num_vertices': 6890,
                    'grid_size': np.array([4, 1, 1]),
                    'mask_size': 1024,
                    'folder': 'hcontact_vitruvian',
                    'pixel_to_vertex': 'pixel_to_vertex_map_1024.npz',
                    'bary_coords': 'bary_coords_map_1024.npz',
                    'contact_annot_f': 'contact_label_objectwise.pkl',
                    'body_parts_annot_f': 'body_parts_objectwise.pkl',
                    'names': np.array([[['topfront']],
                                       [['bottomfront']],
                                       [['topback']],
                                       [['bottomback']]]),
                    'ignore_keywords': [],
                    'cam_params': {
                        'topfront':    [2., 45., 315., 0., 0.],
                        'bottomfront': [2., 315., 315., 0., 0.3],
                        'topback':     [2., 45., 135., 0., 0.],
                        'bottomback':  [2., 315., 135, 0., 0.3]
                    }
                }, 
    '4MV-Z_Vitru_wScene':
                    {'order': 'fix',
                    'num_vertices': 6890,
                    'grid_size': np.array([4, 1, 1]),
                    'mask_size': 1024,
                    'folder': 'hcontact_vitruvian',
                    'pixel_to_vertex': 'pixel_to_vertex_map_1024.npz',
                    'bary_coords': 'bary_coords_map_1024.npz',
                    'contact_annot_f': 'contact_label_objectwise_wFootGround.pkl',
                    'body_parts_annot_f': 'body_parts_objectwise_wFootGround.pkl',
                    'names': np.array([[['topfront']],
                                       [['bottomfront']],
                                       [['topback']],
                                       [['bottomback']]]),
                    'ignore_keywords': ['supporting'], # Skip "supporting" class from DAMON as it might confuse with "scene" from RICH dataset
                    'cam_params': {
                        'topfront':    [2., 45., 315., 0., 0.],
                        'bottomfront': [2., 315., 315., 0., 0.3],
                        'topback':     [2., 45., 135., 0., 0.],
                        'bottomback':  [2., 315., 135, 0., 0.3]
                    }
                }, 
}

VALID_OBJ_NAMES_PIAD = list(AFFORD_PROB_PIAD.keys())

VALID_OBJ_NAMES_LEMON = list(AFFORD_PROB_LEMON.keys())

DAMON_CATEGORIES_MAPPING = {
    "transport": [
        'motorcycle', 'bicycle', 'boat', 'car', 'truck', 'bus', 'train', 'airplane',],
    "accessory": [
        'backpack', 'tie', 'handbag', 'baseball_glove'],
    "furniture": [
        'bench', 'chair', 'couch', 'bed', 'toilet', 'dining_table'],
    'everyday-objects': [
        'book', 'umbrella', 'cell_phone', 'laptop', 'kite', 'suitcase', 'bottle', 'remote',
        'toothbrush', 'teddy_bear', 'scissors', 'keyboard', 'hair drier', 'traffic light',
        'fire_hydrant', 'stop sign', 'tv', 'vase', 'parking meter', 'clock', 'potted plant',
        'mouse'],
    'sports': [
        'frisbee', 'sports_ball', 'tennis_racket', 'baseball_bat',
        'skateboard', 'snowboard', 'skis', 'surfboard'],
    'food': [
        'banana', 'cake', 'apple', 'carrot', 'pizza', 'donut', 'hot_dog',
        'sandwich', 'broccoli', 'orange'],
    'kitchen': [
        'knife', 'spoon', 'cup', 'wine_glass', 'oven', 'fork', 'bowl',
        'refrigerator', 'toaster', 'sink', 'microwave']
  }