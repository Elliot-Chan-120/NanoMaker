from src.nano_maker.container.configs import *
from src.nano_maker.nanomaker import NanoMaker
from src.nano_maker.pocketwatcher import PocketWatcher

from rdkit import Chem
from rdkit.Chem import Crippen

smiles = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
n_trials = 10

scores = []
nmkr_model = NanoMaker(version_code=version_code,
                 skeleton_weight_filename=skeleton_weight_filename,
                 skeleton_cfg=skeleton_config,
                 naanobot_weight_filename=naanobot_weight_filename,
                 naanobot_cfg=naanobot_config,
                 radial_cfg=radial_config)
nmkr_model.ingest_chemical(smiles=smiles)
logp = Crippen.MolLogP(smiles)
print(f"LogP: {logp}")

pw_model = PocketWatcher(naanoeng_config=naanoeng_config,
                                     pocket_config=pocketwatcher_config)

for _ in range(n_trials):
    pocket_data = nmkr_model.generate_nanopkt_data(sampling_temp=0.05)
    summary, notes = pw_model.biochemical_summary(pocket_data['aa_sequence'])
    scores.append(summary["average_polar_character"])

print(f"average polar character = {sum(scores) / n_trials}")
