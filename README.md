# NanoMaker

---
A two-stage cross-attention transformer system that generates a 3D cage of amino acid residues' alpha carbons 
that would form a high-affinity binding pocket to any given chemical in SMILES format. 
These can then be used as protein pocket patch templates for drug-delivery molecules.

NanoMaker separates the challenge of protein pocket design into two transformer tasks. 
Skeleton creates the 3D spatial arrangement, the "skeleton", of the upcoming protein cage, 
while NAAnoBot drops amino acids into that cage based on biochemical compatibility.
Both transformers are cross-attention models conditioned on drug structure, 
meaning that each protein cage is specific to that drug's properties.

Protein binding pockets are characterized as "radial" sequences of spherical coordinates ordered by decreasing shell radius
and biochemical feature vectors. The fineness of ordering is determined by a "radial_resolution" parameter (default 100). 
Resulting "radial sequences" are presented as such during training with the goal of autoregressively predicting the next set of vectors.
```
[[['A'], [rad1, az1, pl1]], [['V'], [rad2, az2, pl2]] .... [['VOID'], [0, 0, 0]]]
         # radius, azimuth, polar                            # end "tokens"
```
Coordinates consist of radius value in angstroms, azimuth and polar angles computed from relative XYZ values. 
Each subsequent amino acid's radius to the ligand centroid decreases. 
Each amino acid identity is mapped to its specific feature vector downstream.
Since protein cage generation is out->in, I interpreted hitting a radius of 0 and under is equivalent to encountering an "END" token in Natural Language Processing.

## Data + Training
Data is resolved protein-drug complexes from BindingDB and PDB, with loss defined as a composite across MSE of radius and unit circle angle distance for Skeleton. NAAnoBot's loss is (not defined right now) composed of a hybrid between MSE and Euclidean distance between feature vectors.
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
molecule should look like this". It then generates a series of spherical coordinate vectors corresponding to an undefined 
amino acid's alpha carbon's placement relative to the chemical compound's centroid (geometric center).

e.g.
```
alpha carbon 1: [14.13, -1.043, 1.56]
alpha carbon 2: [14.00, -1.95, 1.40]
alpha carbon 3: [13.8, -2.44, 1.53]
...
alpha carbon n: [0, 0, 0]
```


## NAANOBOT: Biochemical Environment Curation
Model: NAANOBOT is responsible for deciding which amino acid belongs in certain coordinates.
It actually doesn't interpret sequences via amino acid identities but rather their feature vectors.
Each aa is characterized by their physicochemical properties and chemical structure(s) / functional group(s) that 
distinguish them from the rest. 

This means that NAANOBOT doesn't say "Valine" or "Leucine" belongs here, instead it says
"An amino acid with size X belongs here, and a protein with a Guanidium ring should be here" and so on until the protein pocket is completed.


## todo: Generalization to Unseen Chemistry (after conducting tests on datasets)
As stated previously in the data section, the validation data split consisted not of 20-30% of XY data points, but rather
unseen SMILES that would then produce said XY data points. This was done to "encourage" potential zero-shot capabilities
for new molecules.

might be good to have this here later on 