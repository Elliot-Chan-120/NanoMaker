This folder contains all the jupyter notebooks I used to mine data online.
All data starts off with this:

BindingDB data:
https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202605_tsv.zip

Download and move the file into "database".

Even though the folder is called "prototyping_notebooks", the data mining notebooks are essentially finalized code and what I used to get the main data. 
Notebooks just made data exploration and quality filtering easier.

Run the following in this order:
- [1] raw_data_miner.ipynb
- [2] radial_sequencing_prototype.ipynb
- [3] get_pointers.ipynb

You should get the same files mentioned in the data_README.md, including the ones that this project came with.
The notebooks also contain (casual language) comments which hopefully slightly guide you through the process.

Prototype models for testing can be made in in skeleton_training_prototype.ipynb and naanobot_training_prototype.ipynb.
There you can play around with the parameters and how much training data they see before training on a cloud / GPU server.