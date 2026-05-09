Here are the links you can download files from to download the dataset that we'll use to train nano_maker.

BindingDB data:
https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202605_tsv.zip

Download the above file, plug it in this folder and run the notebook cells.

General workflow:
- obtain high-affinity binding pairs between protein targets and drugs
- datamine online source, isolate binding pocket amino acids by absolute proximity to drug target
  - threshold is 8 Angstroms (i.e. get all amino acids within 8 Angstroms of the drug target)
  - obtain protein coordinate vectors relative to drug centroid
- get drug molecular fingerprints using map4
