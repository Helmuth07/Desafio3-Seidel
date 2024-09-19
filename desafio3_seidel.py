import numpy as np

# Matriz de coeficientes A
A = np.array([
    [52, 20, 25],
    [30, 50, 20],
    [18, 30, 55]
])

# Vector de resultados B
B = np.array([4800, 5210, 5690])

# Número de incógnitas
n = len(B)

# Valores iniciales
x = np.zeros(n)

# Umbral de tolerancia
tolerance = 1e-5

# Número máximo de iteraciones
max_iterations = 1000

def gauss_seidel(A, B, x, tolerance, max_iterations):
    n = len(B)
    for iteration in range(max_iterations):
        x_old = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (B[i] - sum1 - sum2) / A[i, i]
        # Verificar la convergencia
        norm = np.linalg.norm(x - x_old, ord=np.inf)
        if norm < tolerance:
            return x, iteration + 1
    return x, max_iterations

# Ejecutar el método de Gauss-Seidel
solution, iterations = gauss_seidel(A, B, x, tolerance, max_iterations)

print(f"Solución encontrada después de {iterations} iteraciones:")
print(f"x1 = {solution[0]:.5f}")
print(f"x2 = {solution[1]:.5f}")
print(f"x3 = {solution[2]:.5f}")
