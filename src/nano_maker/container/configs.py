skeleton_weight_filename = "skeleton_wtI7e5.pt"
naanobot_weight_filename = "naanobot_wtI7e4.pt"

# globals
max_angstroms = 33
radial_resolution = 100
block_size = 64   # 95th percentile was 67 aas so setting max context to this should be standard.
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
    "loss_temperature": 0.2
}