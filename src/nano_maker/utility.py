from src.paths import *

#globals

HEADER_PREFIX   = ">_"
TARGET_PREFIX   = "Target>_"
NANOPOCKET_START  = ">__nnpkt"

def save_nano_pocket(pocket_data, filename):
    version = pocket_data["version_code"]
    smiles = pocket_data["SMILES"]
    samp_temp = pocket_data["Sampling_temperature"]
    skeleton = pocket_data["3D_skeleton"]
    scaffold = pocket_data["Scaffold"]
    aa_sequence = pocket_data["aa_sequence"]

    # text elements
    header = f"{HEADER_PREFIX}{samp_temp}_{version}_{scaffold}\n"  # identifies temperature setting + version + target ligand, I'll be able to tell what stage of progress this was made
    # i mean ideally they don't give some crazy sized smiles
    target = f"{TARGET_PREFIX}{smiles}\n"

    nano_pocket = f"\n{NANOPOCKET_START}"  # identifier to start sequencing pocket data
    for idx in range(len(skeleton)):
        nano_pocket += f"\n{aa_sequence[idx]}\t{skeleton[idx]}"

    # generate fulltext
    full_content = ""
    full_content += header
    full_content += target
    full_content += nano_pocket
    with open(Path(POCKET_DATA) / f"{filename}.nanopkt", 'w', encoding="UTF-8") as outpath:
        outpath.write(full_content)

    print(f"Pocket data for {smiles} saved:")
    print(f"Find nano pocket file in: {outpath}")

    return True


def load_nano_pocket(filename):
    pocket_data = {
        "version_code": None,
        "SMILES": None,
        "Scaffold": None,
        "Sampling_temperature": None,
        "3D_skeleton": [],
        "aa_sequence": [],
    }

    try:
        with open(Path(POCKET_DATA) / f"{filename}.nanopkt", 'r', encoding="UTF-8") as inpath:
            lines = [line.strip() for line in inpath if line.strip()]
    except Exception as e:
        print(f"File retrieval error: {e}")
        return False

    header = lines[0]  # parse header line first
    h_parts = header.split("_", 3)
    pocket_data["Sampling_temperature"] = float(h_parts[1])
    pocket_data["version_code"] = str(h_parts[2])
    pocket_data["Scaffold"] = str(h_parts[3])

    target = lines[1]
    t_parts = target.split("_", 1)
    pocket_data["SMILES"] = t_parts[1]  # extract actual smiles target -> possible they did multiple targets w the same scaffold

    # parsing 3D block
    reached_3d = False
    skeleton = []
    aa_sequence = []
    for line in lines[1:]:
        if line == f"{NANOPOCKET_START}":
            reached_3d = True
            continue
        if reached_3d:
            aa_id, coords = line.split("\t", 1)
            skeleton.append([float(num) for num in coords.strip("[]").split(",")])
            aa_sequence.append(str(aa_id))

    pocket_data["3D_skeleton"]=skeleton
    pocket_data["aa_sequence"]=aa_sequence

    return pocket_data

