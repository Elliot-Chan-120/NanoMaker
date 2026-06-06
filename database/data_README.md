This file contains some more information on the data used.

Download the BindingDB_All_202605_tsv.zip file from the following link and put it in this folder:

BindingDB data:
https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202605_tsv.zip

Download the above file, plug it in this folder and run the notebook cells.

## General workflow:
- obtain high-affinity binding pairs between protein targets and drugs
- datamine online source, isolate binding pocket amino acids by absolute proximity to drug target
  - threshold is 8 Angstroms (i.e. get all amino acids within 8 Angstroms of the drug target)
  - obtain protein coordinate vectors relative to drug centroid
- get drug molecular fingerprints using map4


## File Information
**Pre-processed / Raw**
- BindingDB_All_202605_tsv.zip: raw dataset downloaded from BindingDB
- unique_pdb_cache.csv: all unique PDB IDs, isolated from their string cluster in the BindingDB_All tsv file
- unique_smiles_strings.csv: all unique SMILES from the same file

**Processed Data**
- AA3D_df.csv -> for each PDB ID (where available) contains isolated protein pocket amino acid identities and their 3d vectors relative to their bound ligand centroid
- SMILES_2_PDBhits.csv (all_pairs_df.csv in notebooks) -> for each processed unique PDB ID, lists each of its high-affinity SMILES hits

The following files are available straight from the repo unlike the above which are obtained by running the jupyter notebook cells.
- molfp_df.pkl -> molecular fingerprints for all unique SMILES
- radial_seq_df.pkl -> radial sequences for all unique proteins (look at README.md)
- test and training pointers_dict.parquet -> "pointer" files containing each high affinity pair's SMILES string and PDB_ID as well as all available window indexes for ease of downstream sampling which prevents data explosion.
  - 5k radial sequences, 8k smiles -> hundreds of sequences per smiles -> 5-107 proteins per sequence ==> 14 million training data points and 3 million validation
- Used dict to optimize training time via cutting down load time with dictionary