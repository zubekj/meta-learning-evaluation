'''
Class that embeds artificial classifier, with embedded information about 
how often it mistakes.

@author: mlukasik
'''

import orange, random

class Learner(object):
	def __new__(cls, examples=None, **kwds):
		learner = object.__new__(cls)
		if examples:
			learner.__init__(**kwds)
			return learner(examples)
		else:
			return learner
	
	def __init__(self, accuracy, field_name = "field_name", name='diettrich learner', **kwds):
		'''
		@param field_name: name of a meta attribute storing information about
		 true classification. The reason for this argument is that functions
		 for testing in Orange delete information about classification.
		'''
		self.__dict__.update(kwds)
		self.accuracy = accuracy
		self.field_name = field_name
		self.name = name
	
	def __call__(self, examples, weight=None, **kwds):
		for k in kwds.keys():
			self.__dict__[k] = kwds[k]
		return Classifier(accuracy = self.accuracy, field_name = 
						self.field_name, name=self.name, 
						domain=examples.domain)

class Classifier:
	def __init__(self, **kwds):
		self.__dict__.update(kwds)

	def __call__(self, example, resultType = orange.GetValue):
		'''
		Classify a sample.
		@param example: a sample to be classified. Assumed, that there is a 
		meta-attribute called field_name.
		@param resulttype: Orange framework style
		'''
		if random.random() < self.accuracy:
			value = example[self.field_name]#example.getclass()
		else:
			#randomly select other value (uniformly)
			classification_values = self.domain[self.field_name].values
			try:
				index = self.domain[self.field_name].values. \
					index(str(example[self.field_name]))
			except:
				print "probably unknown class - return whatever"
				index = random.randint(0, len(classification_values)-1)
			wrong_index = random.randint(0, len(classification_values)-2)
			if wrong_index >= index:
				wrong_index += 1
			value = orange.Value(self.domain[self.field_name], wrong_index)
			
			
		p = [0.] * len(self.domain.classVar.values)
		for i, vi in enumerate(self.domain.classVar.values):
			if vi == str(value):
				p[i] = self.accuracy
			else:
				p[i] = (1. - self.accuracy)/(len(self.domain.classVar.values)-1)
				
		if resultType == orange.GetValue: return value
		elif resultType == orange.GetProbabilities: return self.accuracy
		else: return (value, p)
