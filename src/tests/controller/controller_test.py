#!/usr/bin/python
from magnum import *
import unittest

# Controller
class ControllerTest(unittest.TestCase):

  def test_empty_parameters(self):
    c = Controller(lambda: None, [])
    self.assertEqual([], c.getAllParameters())

  def test_simple_parameters(self):
    c = Controller(lambda x: None, [1, 3, 5])
    self.assertEqual([(1,), (3,), (5,)], c.getAllParameters())

  def test_product_parameters(self):
    c = Controller(lambda x, y: None, [(1, [2, 3])])
    self.assertEqual([(1, 2), (1, 3)], c.getAllParameters())

  def test_complicated_parameters(self):
    c = Controller(lambda x, y: None, [(1, [2, 3]), (4, 5)])
    self.assertEqual([(1, 2), (1, 3), (4, 5)], c.getAllParameters())

# start tests
if __name__ == '__main__':
  unittest.main()
