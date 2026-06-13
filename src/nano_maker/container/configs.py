# unless you understand the system, pls dont touch anything!!!

skeleton_weight_filename = "skeleton_msI7e5.pt"
naanobot_weight_filename = "naanobot_msI7e5.pt"

# version code
version_code = "ms7e84"
# murcko scaffold split, iteration 8 -> probably 5 for skeleton and 2 again for naanobot
# edit: will train skeleton for 3 more epochs to see where performance goes.

# globals
max_angstroms = 33
radial_resolution = 100
block_size = 64
map4_res = 1024


radial_config = {
    'max_angstroms': max_angstroms,
    'radial_resolution': radial_resolution,
}

naanoeng_config = {
    'max_angstroms': max_angstroms,
    'block_size': block_size,
}

skeleton_config = {
    'n_embd': 512,
    'n_head': 8,
    'n_layers': 6,
    'block_size': block_size,
    'map4_res': map4_res,
    'max_angstroms': max_angstroms,
    'radial_resolution': radial_resolution,
    'dropout': 0.15
}

naanobot_config = {
    'n_embd': 512,
    'n_head': 8,
    'n_layers': 6,
    'block_size': block_size,
    'map4_res': map4_res,
    'max_angstroms': max_angstroms,
    'radial_resolution': radial_resolution,
    'dropout': 0.15,
    "n_spatial_features": 10,
    "loss_temperature": 0.2
}


# nord themed graph config
pocketwatcher_config = {
    'paper_bgcolor': "#2e3440",
    'grid_bgcolor': "#2e3440",
    'gridline_color': "#4c566a",
    'label_color': "#d8dee9",


    'colorscales':{
    "skeleton":       [[0.0, '#d8dee9'], [1.0, '#d8dee9']],
    "color_code":     [[0.0, '#5e81ac'], [0.2, '#b48ead'], [0.4, '#bf616a'], [0.6, '#d08770'], [0.8, '#ebcb8b'], [1.0, '#a3be8c']],
    "polar_character":     [[0.0, '#d8dee9'], [0.33, '#88c0d0'], [0.66, '#81a1c1'], [1.0, '#5e81ac']],
    "hydrophobicity":      [[0.0, '#bf616a'], [0.33, '#d08770'], [0.66, '#b48ead'], [1.0, '#5e81ac']],
    "flexibility":         [[0.0, '#bf616a'], [0.33, '#d08770'], [0.66, '#ebcb8b'], [1.0, '#a3be8c']],
    "steric_accessibility":[[0.0, '#d8dee9'], [0.33, '#ebcb8b'], [0.66, '#d08770'], [1.0, '#bf616a']],
},
    'broad_threshold': 14,
    # determines how far a dimension has to be occupied by amino acids to be considered "broadly" occupied
}