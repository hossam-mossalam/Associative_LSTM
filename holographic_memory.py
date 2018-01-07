'''
    This is the file containing the holographic memory class.
    The holographic memory is used for Associative LSTM to store information.
'''

from functools import reduce
import numpy as np

class HolographicMemory:
  '''
  Holographic memory class
  '''

  def __init__(self, cells_count, cell_size):
    '''
    Initialize the holographic memory to have cells_count cells,
    and each cell has the size of cell_size and it's own permutation.
    '''
    self.cell_size = cell_size
    self.cells_count = cells_count
    self.cells = np.zeros((cells_count, cell_size))
    self.permutations = np.array(
        [np.random.permutation(cell_size // 2)
        for _ in range(self.cells_count)])
    self.permutations = np.append(self.permutations,
                                  self.permutations + cell_size // 2,
                                  axis = 1)

  # keys: batch_size * num_units
  # data: batch_size * num_units
  def write(self, keys, data):
    '''
        The write method which takes a key and a value, and uses each
        cells permutation to permute the key and write the value to the
        memory cell.
    '''
    keys = self.permute(keys)
    # add a new dimension to the data
    data = np.expand_dims(data, axis = 0)
    output = HolographicMemory.complex_multiplication(keys, data, self.cell_size)
    self.cells += np.sum(output, axis = 1)


  def read(self, keys):
    '''
        Reads the values associated with the permuted key from every memory
        cell and returns the average.
    '''
    keys = self.permute(keys)
    HolographicMemory.inverse(keys, self.cell_size)
    # add a new dimension to the data
    cells = np.expand_dims(self.cells, axis = 1)
    output = HolographicMemory.complex_multiplication(keys, cells, self.cell_size)
    output = np.mean(output, axis = 0)
    return output

  # d: batch_size * num_inputs
  # output: num_copies * batch_size * num_inputs
  def permute(self, d):
    '''
        Permutes input d based on permutation p.
    '''
    output = []
    for e in d:
      output.append(e[self.permutations])
    output = np.array(output)
    output = np.swapaxes(output, 0, 1)
    return output

  @classmethod
  def complex_multiplication(cls, u, v, cell_size):
    '''
        Returns the mutliplication of the two complex numbers u and v.
    '''
    u_r = u[:,:,:cell_size//2]
    u_i = u[:,:,cell_size//2:]
    v_r = v[:,:,:cell_size//2]
    v_i = v[:,:,cell_size//2:]
    out_r = u_r * v_r - u_i * v_i
    out_i = u_r * v_i + u_i * v_r
    output = np.append(out_r, out_i, axis = 2)
    return output


  # k: num_copies * batch_size * num_inputs
  @classmethod
  def inverse(cls, k, cell_size):
    '''
        Returns the complex_conjugate of k.
    '''
    k[:,:,cell_size//2:] *= -1
