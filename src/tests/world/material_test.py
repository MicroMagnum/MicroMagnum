#!/usr/bin/python
from magnum.micromagnetics import *
import unittest

# Material
class MaterialTest(unittest.TestCase):
  def setUp(self):
    self.py = Material.Py()

  def test_getter(self):
    mat = Material({'Ms':8e3})
    self.assertEqual(8e3, mat.get('Ms'))

  def test_permalloy(self):
    self.assertEqual(8e5, self.py.get('Ms'))
    self.assertEqual(8e5, self.py.Ms)

# start tests
if __name__ == '__main__':
  unittest.main()
