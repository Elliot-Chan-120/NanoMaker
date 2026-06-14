This folder contains all the jupyter notebooks I used to datamine.
These are intended to be run from an IDE or jupyterlab

All data starts off with this:

BindingDB data:
https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202605_tsv.zip

Download and move the file into "database".

Run the following in this order:
- [1] a00_raw_data_miner.ipynb
- [2] a01_radial_sequencing_prototype.ipynb
- [3] b00_skeleton_pointers.ipynb
- [4] c00_naanobot_pointers.ipynb

You should get the same files mentioned in the data_README.md, including the ones that this project came with.
The notebooks also contain (slightly casual) comments which hopefully slightly guide you through the process.

Prototype models for testing can be made in in skeleton_training_prototype.ipynb and naanobot_training_prototype.ipynb.
There you can play around with the parameters and how much training data they see before training on a cloud / GPU server.
