import orange

class Learner(object):
	def __new__(cls, examples=None, **kwds):
		learner = object.__new__(cls, **kwds)
		if examples:
			learner.__init__(**kwds)
			return learner(examples)
		else:
			return learner

	def __init__(self, probability=0.5, name='dietterich classifier', **kwds):
		self.__dict__.update(kwds)
		self.probability = probability
		self.name = name

	def __call__(self, examples, weight=None, **kwds):
		for k in kwds.keys():
			self.__dict__[k] = kwds[k]
		domain = examples.domain

		return Classifier(probability = self.probability, domain=domain, name=self.name)

class Classifier:
	def __init__(self, **kwds):
		self.__dict__.update(kwds)

	def __call__(self, example, result_type=orange.GetValue):
		v = orange.Value(self.domain.classVar, 0)
		p = self.probability
		
		# return the value based on requested return type
		if result_type == orange.GetValue:
			return v
		if result_type == orange.GetProbabilities:
			return p
		return (v,p)
		
data = orange.ExampleTable("voting")
classifier = Learner(data)
print classifier(data[0])
