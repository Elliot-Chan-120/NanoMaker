from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from map4 import MAP4

# drug tokenizing: molecular fingerprint

# important note:
# input smiles should be screened first for drug-likeness beforehand during inference/generation
# parameters
mol_dim = 1024
map4_fprinter = MAP4(
    dimensions=mol_dim,
    radius=2,
    include_duplicated_shingles=True,
)
# look at study one molecular fingerprint to rule them all: drugs, biomolecules
# can always do a double take later

def eval_lipinski(smiles):
    """
    Use on user-input smiles, don't shut down inference run just flag molecule as non drug-like
    :param smiles:
    :return:
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES String")

    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
    except Exception as e:
        raise ValueError(f"Error calculating Lipinski Descriptors: {e}")

    rules_passed = 0
    if mw <= 500: rules_passed += 1
    if logp <= 5: rules_passed += 1
    if h_donors <= 5: rules_passed += 1
    if h_acceptors <= 10: rules_passed += 1

    return rules_passed


def smiles_fingerprint(smiles):
    """
    Extract molecular fingerprint from singular SMILES - fairly straightforward
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        map4_fp = map4_fprinter.calculate(mol=mol)

        return map4_fp

    except Exception as e:
        return None


## TEST BLOCK
# test_smiles = "COc1nc(OC[C@H]2C[C@@H]2c2ccc3ncccc3n2)nc(NCc2cnn(C)c2)c1C |r|"
# test_fp = mol_from_smiles(test_smiles)
# drug_like = eval_lipinski(test_smiles)
#
# print(drug_like)
# print(test_fp)  #array([1, 0, 1, ..., 0, 0, 0], shape=(1024,), dtype=uint8)