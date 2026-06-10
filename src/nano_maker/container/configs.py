# unless you understand the system, pls dont touch anything!!!

skeleton_weight_filename = "skeleton_gsI7e5.pt"
naanobot_weight_filename = "naanobot_gsI7e2.pt"

# version code
version_code = "gsI7e52"
# generic scaffold split, iteration 7, epochs -> 5 for skeleton and 2 for naanobot
# in progress: murcko scaffold split, iteration 8 -> probably 5 for skeleton and 2 again for naanobot

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