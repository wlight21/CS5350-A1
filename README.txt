- running experiments
	
	- unzip submitted zip file Experiments.zip
	- from Experiments directory, execute run.sh on CADE with ./run.sh
	- this script is very short, it simply runs main.py

- output

	- all experiment outputs are organized into text files
	- baseline.txt contains baseline test information for the most common
	  label in train.txt
	- acc_* files contain accuracy info on train.csv and test.csv for decision trees
	  trained on train.csv
	- avg_acc_depth* files contain average accuracy information for depth
	  limited trees trained across the 5 different folds
	- id3* files contain the step by step breakdown of the id3 algorithm
	  for constructing the various trees
	- unless stated as "collision", all trees are trained using the standard,
	  shannon entropy definition

- output breakdown: question(s), file(s)

	- 1, baseline.txt
		- most common label in train.csv and its accuracy on test.csv
	- 2a, 2b, 2c, id3_shannon.txt
		- step by step decision breakdown of id3 algorithm building an unlimited depth decision tree 
		  using the standard, shannon entropy definition. the first section of attribute selection
		  displays the information gain of the root feature. the maximum depth of the tree can be found at the 
		  bottom of the file
	- 2d, 2e, acc_shannon.txt
		- accuracy stats of unlimited depth decision tree on train.csv using the standard, shannon entropy definition
	- 3a, avg_acc_depth*.txt
		- average accuracy and standard deviations of depth limited trees on the 5-fold cross
		  validation data sets
	- 3c, acc_depth5.txt
		- accuracy stats for a depth limited tree, according to the best tree from 3a, on train.csv
		  and test.csv

	- id3_collision.txt
		- step by step decision breakdown of id3 algorithm building an unlimited depth decision tree 
		  using the collision entropy definition. the first section of attribute selection
		  displays the information gain of the root feature. the maximum depth of the tree can be found at the 
		  bottom of the file
	- acc_collision.txt
		- accuracy stats of unlimited depth decision tree on train.csv using the collision entropy definition
	- id3_depth*.txt
		- step by step decision breakdown of id3 algorithm building depth limited decision trees
		  using the standard, shannon entropy definition. the first section of attribute selection
		  displays the information gain of the root feature. the maximum depth of the tree can be found at the 
		  bottom of the file
		 
