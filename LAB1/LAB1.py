import math
import copy

N = 24
P = math.log10(N)

# Метод Гаусса по строке
def gaussian_method(A, b):
    n = len(A)
    swap_count = 0
    perm = list(range(n))
    # Прямой ход
    for k in range(n):
        # выбор главного элемента по строке
        max_col = k
        for j in range(k + 1, n):
            if abs(A[k][j]) > abs(A[k][max_col]):
                max_col = j

        if abs(A[k][max_col]) < 1e-12:
            raise ValueError("Матрица вырождена, единственного решения нет")

        # перестановка столбцов
        if max_col != k:
            for i in range(n):
                A[i][k], A[i][max_col] = A[i][max_col], A[i][k]
            perm[k], perm[max_col] = perm[max_col], perm[k]
            swap_count += 1

        # зануление
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]

            for j in range(k, n):
                A[i][j] -= m * A[k][j]

            b[i] -= m * b[k]
    # Треугольная матрица
    triangle_m = copy.deepcopy(A)
    # обратный ход
    x_temp = [0.0] * n

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += A[i][j] * x_temp[j]
        x_temp[i] = (b[i] - s) / A[i][i]

    # восстановление порядка неизвестных
    x = [0.0] * n
    for i in range(n):
        x[perm[i]] = x_temp[i]

    # определитель
    det = 1.0
    for i in range(n):
        det *= A[i][i]
    if swap_count % 2:
        det *= -1

    return x, det, triangle_m

# Обратная матрица
def inverse_matrix(A):
    n = len(A)

    A = [A[i] + [1 if i == j else 0 for j in range(n)]
         for i in range(n)]

    for i in range(n):
        pivot = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[pivot][i]):
                pivot = j

        if abs(A[pivot][i]) < 1e-12:
            raise ValueError("Матрица необратима")

        A[i], A[pivot] = A[pivot], A[i]
        pv = A[i][i]
        for j in range(2 * n):
            A[i][j] /= pv

        for j in range(n):
            if j != i:
                factor = A[j][i]
                for k in range(2 * n):
                    A[j][k] -= factor * A[i][k]
    A = [row[n:] for row in A]
    return A

# Умножение матриц (для проверки обратной)
def matrix_multiplication(A, B):
    n = len(A)
    E = [[0.0]*n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                E[i][j] += A[i][k] * B[k][j]
    return E

# Метод верхней релаксации
def iteration_method(A, b, omega=1.2, eps=1e-12, max_iter=1000):
    n = len(A)
    x = [0.0] * n

    for iteration in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            if A[i][i] == 0:
                raise ValueError("Нулевой диагональный элемент")
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))

            x[i] = (
                (1 - omega) * x_old[i]
                + omega / A[i][i] * (b[i] - sum1 - sum2)
            )

        error = max(abs(x[i] - x_old[i]) for i in range(n))
        if error < eps:
            print(f"Количество итераций: {iteration + 1}")
            return x

    print("Достигнут предел итераций")
    return x

# Невязка
def residual(A, x, b):
    n = len(A)
    r = []

    for i in range(n):
        s = sum(A[i][j] * x[j] for j in range(n))
        r.append(abs(b[i] - s))
    return r

#Норма-максимум
def norm_max(v):
    return max(abs(x) for x in v)

#Евклидова норма
def norm_euclide(v):
    return math.sqrt(sum(x * x for x in v))

def print_vector(name, v, gauss=False):
    print(name)
    if gauss and not v:
        print("Матрица вырождена и не имеет единственного решения")
    for i, val in enumerate(v):
        if gauss:
            print(f"x{i+1} = {val:.5e}")
        else:
            print(f"{val:.5f}")
    print()

def print_matrix(name, matrix, exp_f=False):
    print(name)
    for row in matrix:
        if exp_f:
            print(*[f"{x:.2e}" for x in row])
        else:
            print(*[f"{x:.5f}" for x in row])
    print()

# Основная часть программы
A = [
    [0, 4, 2, 2],
    [4, 0, 0, 2],
    [2, 0, 9, -4],
    [2, 2, -4, 0]
]
A[0][0] = P + 10 * N 
A[1][1] = N + 5
A[3][3] = 8 * (N + P)

b = [2 * N * math.sin(N), 
            5 * (math.sin(N) - math.cos(N)),
            7 * (math.cos(N) + math.sin(N)),
            3 * math.sin(N)]

print_matrix("Матрица А:", A)
print_vector("Вектор b:", b)

x_gauss, det, triangle_m = gaussian_method(copy.deepcopy(A), copy.deepcopy(b))
print_matrix("Треугольная матрица:", triangle_m, True)
print_vector("Решение методом Гаусса по строке:", x_gauss, True)

print("Определитель: ", det)

if det:
    inv_matrix = inverse_matrix(copy.deepcopy(A))
    print_matrix("\nОбратная матрица:", inv_matrix, True)
    E = matrix_multiplication(inv_matrix, A)
    print_matrix("Проверка обратной матрицы A^-1 * A = E:", E, True)
else:
    print("Обратной матрицы не существует")

print("Метод верхней релаксации, w = 1.2:")
x_iteration = iteration_method(A, b, omega=1.2)
print_vector("Решение:", x_iteration, True)

res = residual(A, x_iteration, b)
print("Невязка системы r = b - A*x:")
[print(f"{num:.3e}") for num in res]
print("\nНорма-максимум:", f"{norm_max(res):.3e}")
print("Евклидова норма:", f"{norm_euclide(res):.3e}")