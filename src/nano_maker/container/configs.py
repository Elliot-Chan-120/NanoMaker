skeleton_weight_filename = "skeleton_E3I1.pt"
naanobot_weight_filename = "naanobot_I2.pt"

# globals
max_angstroms = 33
radial_resolution = 100
block_size = 75
map4_res = 1024


radial_config = {
    'max_angstroms': max_angstroms,
    'radial_resolution': radial_resolution,
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
}