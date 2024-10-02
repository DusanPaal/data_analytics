from unittest import TestCase, TextTestRunner
import numpy as np
import solver
import time

class TestSolver(TestCase):

  def setUp(self):

    self.a = np.matrix([
      [1, 2],
      [3, 4]
    ])

    self.b = np.array([5, 6])

  def test_solve_linear_system(self):
    
    x = solver.solve_linear_system(self.a, self.b, method='numpy')
    self.assertTrue(np.allclose(x, np.array([-4, 4.5])))

    x = solver.solve_linear_system(self.a, self.b, method='scipy')
    self.assertTrue(np.allclose(x, np.array([-4, 4.5])))

    x = solver.solve_linear_system(self.a, self.b, method='lu')
    self.assertTrue(np.allclose(x, np.array([-4, 4.5])))

    with self.assertRaises(ValueError):
      x = solver.solve_linear_system(self.a, self.b, method='invalid')

if __name__ == "__main__":

  # Estimate pi with 10,000 samples
  print(solver.monte_carlo_pi(1000000))

  # runner = TextTestRunner()
  # runner.run(TestSolver('test_solve_linear_system'))
