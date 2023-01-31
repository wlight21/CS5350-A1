import numpy as np
import pandas
import math
from string_builder import StringBuilder
from decision_tree import DecisionTree

def mode_label(train_set): return train_set["label"].mode()[0] 

def mode_acc(train_mode, test_set): return sum([1 if label == train_mode else 0 for label in test_set["label"]]) / len(test_set["label"])

def shannon_entropy(s):
	entropy = 0
	for value in s["label"].unique():
		sub_set = s.loc[s["label"] == value]
		p_label = len(sub_set) / len(s["label"])
		entropy -= p_label * math.log(p_label, 2)
	return entropy

def collision_entropy(s):
	entropy = 0
	for value in s["label"].unique():
		sub_set = s.loc[s["label"] == value]
		p_label = len(sub_set) / len(s["label"])
		entropy += p_label**2
	return -math.log(entropy, 2)

def main(): 

	features = { 
				 "buying" : ["vhigh", "high", "med", "low"], 
				 "maint" : ["vhigh", "high", "med", "low"],
				 "doors" : ["2", "3", "4", "5more"],
				 "persons" : ["2", "4", "more"], 
				 "lug_boot" : ["small", "med", "big"], 
				 "safety" : ["low", "high", "med"]
			   }
	train_set = pandas.read_csv("data\\data\\train.csv")
	test_set = pandas.read_csv("data\\data\\test.csv")

	with open("baseline.txt", "w") as f:
		out = StringBuilder()
		out.append_line("Most Common Label in train.csv: %s" % (mode := mode_label(train_set)), True)
		out.append_line("Most Common Label Accuracy on test.csv: %f" % mode_acc(mode, test_set), True)
		out.dump_output(f)
		f.close()

	tree_shannon = DecisionTree(shannon_entropy, train_set, features, StringBuilder())

	with open("id3_shannon.txt", "w") as f:
		tree_shannon.logger.dump_output(f)
		f.close()

	train_correct = tree_shannon.total_predict(train_set)
	test_correct = tree_shannon.total_predict(test_set)

	with open("acc_shannon.txt", "w") as f:
		out = StringBuilder()
		out.append_line("Decision Tree Accuracy on train.csv: %f" % (train_correct / len(train_set["label"])), True)
		out.append_line("Decision Tree Accuracy on test.csv: %f" % (test_correct / len(test_set["label"])), True)
		out.dump_output(f)
		f.close()

	tree_collision = DecisionTree(collision_entropy, train_set, features, StringBuilder())

	with open("id3_collision.txt", "w") as f:
		tree_collision.logger.dump_output(f)
		f.close()

	train_correct = tree_collision.total_predict(train_set)
	test_correct = tree_collision.total_predict(test_set)

	with open("acc_collision.txt", "w") as f:
		out = StringBuilder()
		out.append_line("Decision Tree Accuracy on train.csv: %f" % (train_correct / len(train_set["label"])), True)
		out.append_line("Decision Tree Accuracy on test.csv: %f" % (test_correct / len(test_set["label"])), True)
		out.dump_output(f)
		f.close()

	avg_accs = {}
	folds = { i:pandas.read_csv("data\\data\\CVfolds\\fold%s.csv" % i) for i in range(1, 6, 1) }
	for depth in (depths := [1,2,3,4,5]):

		avg_train_acc = 0
		avg_test_acc = 0

		for fold in folds.keys():

			indexes = [i for i in range(1, 6, 1) if i != fold]
			train_set_fold = pandas.concat([folds[i] for i in indexes])
			test_set_fold = folds[fold]
			tree_limited = DecisionTree(shannon_entropy, train_set_fold, features, StringBuilder(), max_depth=depth)
			avg_train_acc += tree_limited.total_predict(train_set_fold) / len(train_set_fold)
			avg_test_acc += tree_limited.total_predict(test_set_fold) / len(test_set_fold)

			with open("id3_depth%s.txt" % depth, "w") as f:
				tree_limited.logger.dump_output(f)
				f.close()

		avg_train_acc /= len(depths)
		avg_test_acc /= len(depths)

		with open("avg_acc_depth%s.txt" % depth, "w") as f:
			dlp_output = StringBuilder()
			dlp_output.append_line("Avg. Depth_%s Accuracy on 5-Fold Training Data: %f" % (depth, avg_train_acc), True)
			dlp_output.append_line("Avg. Depth_%s Accuracy on 5-Fold Test Data: %f" % (depth, avg_test_acc), True)
			dlp_output.dump_output(f)
			f.close()

		avg_accs[avg_test_acc] = tree_limited

	best_tree = avg_accs[max(avg_accs.keys())]
	tree_best_limit = DecisionTree(shannon_entropy, train_set_fold, features, StringBuilder(), best_tree.max_depth)
	with open("acc_depth%s.txt" % best_tree.max_depth, "w") as f:
		out = StringBuilder()
		out.append_line("Depth_%s Tree Acc. on train.csv: %f" % 
						(tree_best_limit.max_depth, tree_best_limit.total_predict(train_set) / len(train_set)), True)
		out.append_line("Depth_%s Tree Acc. on train.csv: %f" % 
						(tree_best_limit.max_depth, tree_best_limit.total_predict(test_set) / len(test_set)), True)
		out.dump_output(f)


if __name__ == "__main__": main()