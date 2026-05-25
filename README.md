# NanoMaker

---

NanoMaker is a dual-transformer system that, when presented with a chemical 
structure in SMILES format, generates a 3D spatial arrangement of amino acid residues' alpha carbons 
that would form a high-affinity binding pocket. 
These can then be used as protein pocket patch templates for drug-delivery molecules.

NanoMaker separates the challenge of protein pocket design into two transformer tasks. 
Skeleton creates the 3D spatial arrangement, the "skeleton", of the upcoming protein cage, 
while NAAnoBot drops amino acids into that cage based on biochemical compatibility.
Both transformers are cross-attention models conditioned on drug structure, 
meaning that each protein cage is specific to that drug's properties.

Protein binding pockets are characterized as "radial" sequences of coordinates ordered by decreasing shell radius
and biochemical feature vectors. They are presented as such during training with the goal of autoregressively predicting the next set of vectors.
e.g.
```
[['A'], [x1, y1, z1]], [['V'], [x2, y2, z2]] ....
```
Where each subsequent amino acid's radius to the ligand centroid decreases. Each amino acid identity is mapped to its specific feature vector downstream.

## Data + Training
Data is resolved protein-drug complexes from BindingDB and PDB, with loss defined as a hybrid between Mean Squared Error and Shortest / Euclidian distance for both Skeleton and NAAnoBot.
The data split was done according to drug identity rather than a random split after combinatorial explosion of drug vs. sequence windows.
Training split comprised of 14 million training sequence windows. Validation set was comprised solely of molecules non-existent in training data, 
meaning that the models learn actual relationships b/w 3D arrangement, biochemistry and drug structure rather than memorization.


## Disclaimer + Note on Pathogenic Resemblance
This project is an independent research prototype, built for learning and exploration.
Generated protein pockets do not account for orientation of both the ligand or the amino acids themselves in 3D space.
The architecture, training pipeline and data representations are not validated against established benchmarks in structural biology or computational drug discovery.
Generated pockets should not be used to inform any protein design, clinical or therapeutic decisions.

There may also exist the possibility that the pockets generated might resemble certain pathogenic molecules since that's what the training data comprised of.
However, I should note that there is a distinction between:
- Pathogenic active / binding sites: exist within a pathogen + performs harmful function, comprised of more complex structures
- NanoMaker-generated pockets: 3D-designed pocket with the sole purpose of high binding affinity


## Skeleton: 3D structure generation
Model: Skeleton is responsible for generating the 3D spatial arrangement of the protein pocket
prior to the amino acid insertion into said pocket, hence the name "Skeleton". 

When presented with a chemical compound, it will say: "the protein cage surrounding this 
molecule should look like this". It then generates a series of 3D vectors corresponding to an undefined amino acid's alpha carbon
coordinates relative to the chemical compound's centroid (geometric center).

e.g.
```
alpha carbon 1: [55, 40, 64]
alpha carbon 2: [33, 21, 57]
alpha carbon 3: [43, 37, 39]
...
alpha carbon n: [ end ]
```


## NAANOBOT: Biochemical Environment Curation
Model: NAANOBOT is responsible for deciding which amino acid belongs in certain coordinates.
It actually doesn't interpret sequences via amino acid identities but rather their feature vectors.
Each aa is characterized by their physicochemical properties and chemical structure(s) / functional group(s) that 
distinguish them from the rest. 

This means that NAANOBOT doesn't say "Valine" or "Leucine" belongs here, instead it says
"An amino acid with size X belongs here, and a protein with a Guanidium ring should be here" and so on until the protein pocket is completed.


## Generalization to Unseen Chemistry (after conducting tests on your datasets)
As stated previously in the data section, 

might be good to have this here