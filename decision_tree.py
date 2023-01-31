def dict_sub(dic, key):
	copy = dic.copy()
	copy.pop(key)
	return copy


class DecisionTree:

	class Node:

		def __init__(self, label, parent=None):
			self.label = label
			self.parent = parent
			self.edges = {}

		def add_edge(self, value, att): self.edges[value] = att
		def is_leaf(self): return len(self.edges) == 0
		def is_root(self): return self.parent == None

		def predict(self, data):
			if self.is_leaf(): return self.label
			else: return self.edges[data[self.label]].predict(data)

		def depth(self):
			if self.is_leaf(): return 0
			depth = -1
			for value in self.edges:
				node_depth = 1 + self.edges[value].depth()
				if node_depth > depth: depth = node_depth
			return depth

	def __init__(self, entropy_def, train_set, atts, logger, max_depth=-1): 
		self.entropy_def = entropy_def
		self.train_set = train_set
		self.atts = atts

		self.logger = logger
		self.logger.append_line("Entropy of train.csv: %f" % self.entropy_def(train_set), True)

		self.max_depth = max_depth
		self.root = self.id3(self.train_set, self.atts, 0)
		self.logger.append_line("Root Attribute: %s" % self.root.label)
		self.logger.append_line("Max Depth: %s" % self.depth(), True)

	def root_att(self, s, atts):
		self.logger.append_line("Choosing best attribute from: %s" % str(atts.keys()))
		self.logger.append_line("Entropy of dataset: %f" % self.entropy_def(s))

		max_att = ""; max_info_gain = 0
		for att in atts:
			if (info_gain := self.info_gain(s, att)) > max_info_gain:
				max_att = att
				max_info_gain = info_gain
			self.logger.append_line("Info Gain for %s: %f" % (att, info_gain))

		self.logger.append_line("Chosen Attribute: %s" % max_att, True)

		return max_att

	def info_gain(self, s, attribute):
		entropy = self.entropy_def(s)
		entropy_new = 0
		for value in s[attribute].unique():
			sub_set = s.loc[s[attribute] == value]
			entropy_new += (len(sub_set)/len(s)) * self.entropy_def(sub_set)

		return entropy - entropy_new

	def id3(self, s, atts, depth):

		if depth == self.max_depth:
			self.logger.append_line("Max Depth Reached", True)
			return self.Node(s["label"].mode()[0])

		if len(s["label"].unique()) == 1: 
			self.logger.append_line("Consistent Label for all Instances", True)
			return self.Node(s["label"].mode()[0])

		root = self.Node(self.root_att(s, atts))

		for value in atts[root.label]:
			self.logger.append_line("Building Tree for Branch: (%s)--->(%s)" % (root.label, value))
			if len((subset := s.loc[s[root.label] == value])) < 1: 
				self.logger.append_line("No Instances in Subset", True)
				root.add_edge(value, self.Node(s["label"].mode()[0]))
			else: root.add_edge(value, self.id3(subset, dict_sub(atts, root.label), depth + 1))

		return root

	def predict(self, data): return self.root.predict(data)
	def total_predict(self, s): return sum([1 if self.predict(row) == row["label"] else 0 for i,row in s.iterrows()])
	def depth(self): return self.root.depth()