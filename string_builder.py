class StringBuilder:
	def __init__(self): 
		self.output = []
		self.sep = "<--------------------------------------------------->"
	def append_line(self, text, sep=False): 
		self.output.append(text)
		if sep: self.append_sep()
	def append_sep(self): self.append_line(self.sep)
	def dump_output(self, file):
		file.write("\n".join(self.output))
		self.output = []