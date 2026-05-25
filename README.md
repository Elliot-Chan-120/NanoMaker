# NanoMaker
### A cross-attention protein pocket designer: adapt to any drug structure

NanoMaker is a dual transformer-based system that, when presented with a chemical 
structure in SMILES format, generates a 3D spatial arrangement of amino acid residues' alpha carbons 
that would form a high-affinity binding pocket.

talk about radial shell 
cross attention with learned map4 fingerprint embedding
- condition 3D pocket generation on ligand identity



data is resolved protein-ligand complexes from PDB, loss defined as a hybrid between MSE and Euclidian distance
training data = 14 million raining sequence windows
validation set is comprised solely of molecules non-existent in training data -> model learns from actual relationships b/w 3D arrangement, biochemistry and drug structure rather than memorization

This project is an independent research prototype, build for learning and exploration.
The architecture, training pipeline and data representations are not validated against established benchmarks in structural biology or computational drug discovery.
Generated pocket geometries should not be used to inform any protein design, clinical or therapeutic decisions.

Important distinction between binding sites in terms of pathogenic resemblance.
- pathogenic active / binding sites: exist within a pathogen + performs harmful function
- nano_maker-generated pockets: extracted as a sequence pattern and re-embedded IRL into a designed protein with a completely different context


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
It actually doesn't "see" amino acid identities but rather their feature vectors curated
by a feature engineering class "NanoEng", that characterizes each amino acid by their 
physicochemical properties and their chemical structure(s) / functional group(s) that 
distinguish them from the rest. 

This means that NAANOBOT doesn't say "Valine" or "Leucine" belongs here, instead it says
"A protein with size X belongs here, and a protein with a Guanidium ring should be here and so on".
