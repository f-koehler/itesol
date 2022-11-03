import itesol_core

backend = itesol_core.EigenDenseBackendFloat64()
matrix = backend.create_matrix(64, 64)
linear_operator = backend.make_linear_operator(matrix)
algorithm = itesol_core.PowerMethodFloat64(64, backend, 10000, 1e-8)
observer = itesol_core.BasePowerMethodObserverFloat64()
algorithm.compute(linear_operator, observer)
