import unittest
import sys

sys.path.append('../../src/experiments_synthetic_classifiers/')

import dietterich
import orange, Orange

class TestDietterich(unittest.TestCase):

	data = orange.ExampleTable("voting")
	#add meta-information about examples' class
	name = type(data.domain.class_var)("field_name")
	name.values = data.domain.class_var.values
	data.domain.add_meta(Orange.feature.Descriptor.new_meta_id(), name)
	for e in data:
		e[name] = e.getclass()
	
	def test0(self):
		dl = Orange.evaluation.testing. \
		cross_validation([dietterich.Learner(accuracy=0, \
											field_name="field_name")], \
						self.data)
		self.assertEqual(Orange.evaluation.scoring.CA(dl), [0])
		
	def test1(self):
		dl = Orange.evaluation.testing. \
		cross_validation([dietterich.Learner(accuracy=1, \
											field_name="field_name")], \
						self.data)
		self.assertEqual(Orange.evaluation.scoring.CA(dl), [1])


if __name__ == '__main__':
	unittest.main()
