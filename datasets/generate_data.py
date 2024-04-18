import numpy as np

def generate_matrices(data_shape, sparsity, use_vector, data_type, precision,mean, deviation):
    """
    Generates two matrices or vectors based on the provided specifications.

    Args:
        data_scale (tuple): A tuple containing two tuples, each specifies the dimensions
                            (ROWS, COLS) of the matrices to be generated, in format like ((1000, 500), (500, 1000))
        sparsity (float): The fraction of elements in the matrices that are zero.
        use_vector (bool): If True, generates vectors instead of matrices.
        data_type (str): 'binary' for binary matrices, 'normal' for normal continuous values.
        precision (str): 'float' for floating point numbers, 'int' for integers.


    Returns:
        tuple: Two numpy arrays as specified by the input parameters.
    """
    def create_matrix(dimensions, sparsity, is_vector, data_type, precision):
        rows, cols = dimensions
        if is_vector:
            cols = 1  # If it's a vector, ensure there's only one column

        # Set the data type based on the precision argument
        if precision == 'float':
            dtype = np.float32
        elif precision == 'int':
            dtype = np.int32
        else:
            raise ValueError("Unsupported precision type. Use 'float' or 'int'.")

        # Generate the matrix based on the data type
        if data_type == 'binary':
            matrix = np.random.choice([0, 1], size=(rows, cols), p=[sparsity, 1-sparsity], replace=True).astype(dtype)
        elif data_type == 'normal':
            matrix = np.random.normal(loc=mean, scale=deviation, size=(rows, cols)).astype(dtype)
            if sparsity > 0:
                mask = np.random.choice([0, 1], size=(rows, cols), p=[sparsity, 1-sparsity], replace=True)
                matrix *= mask
        else:
            raise ValueError("Unsupported data type. Use 'binary' or 'normal'.")

        return matrix

    # Generate the two matrices/vectors
    matrix1 = create_matrix(data_shape[0], sparsity, use_vector, data_type, precision)
    matrix2 = create_matrix(data_shape[1], sparsity, use_vector, data_type, precision)

    return (matrix1, matrix2)

if __name__ == '__main__':
    # Example usage:
    data_scale = ((1000, 500), (500, 1000))  # Dimensions for two matrices
    sparsity = 0 # 30% elements are zero
    use_vector = True  # Generate matrices, not vectors
    data_type = 'normal'  # Normal continuous values
    precision = 'float'  # Floating point numbers

    mat1, mat2 = generate_matrices(data_scale, sparsity, use_vector, data_type, precision,1,1)
    print("Matrix 1:\n", mat1)
    print("Matrix 2:\n", mat2)
