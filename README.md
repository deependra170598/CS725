# CS725
CNN model for jet tagging.
The whole process is inspired from following link: 
https://medium.com/@sbutalla2012/machine-learning-for-classifyingw-initiated-and-qcd-background-jets-24a44f570c71 .
This branch has five .py files.
1. unprocessed_jet_image.py
2. data_exploration.py
3. model.py
4. discriminant.py
5. prediction.py

Steps to see this project:
1. First Delphes software was run with input file "pp_zp_ww.hep" and output file "out_pp_zp_ww.root". 
2. Second "unprocessed_jet_image.py" file was run with input file "out_pp_zp_ww.root". This step generates a example jet image which is needed to process.
3. Third "model.py" file was run with input file as a publicly available data file on the page https://data.mendeley.com/datasets/4r4v785rgx/1 .
This step is to train and test CNN model over publicly available data and save the model's architecture with its parameters and plot accuracy graph. Model saved in step has folder name "model1_n=0.1".
4. Fourth "discriminant.py" file was run with the same input file as used in step no. 3. Model saved in step has folder name "model2_n=0.1". Thus we have two models.
5. Fifth "prediction.py" file was run to see the predictions from both the models. 
