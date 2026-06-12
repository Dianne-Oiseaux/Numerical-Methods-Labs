import math
import pandas as pd
import matplotlib.pyplot as plt

# Производная
def func(x, y):
    return - pow((y / x), 2) - (y / x) - 1

# Точное решение
def exact_solution(table):
    table["Точное решение"] = table["x"].apply(lambda x: -(x * math.log(x)) / (math.log(x) + 1))

# Явный м. Эйлера
def explicit_Euler(table, column="Явный м. Эйлера", h=0.1):
    y_0, x_0 = 0, 1
    y_set = [y_0]
    n = round((2 - 1) / h)
    for i in range(n):
        x = x_0 + i * h
        y_next = y_0 + h * func(x, y_0)
        y_set.append(y_next)
        y_0 = y_next
    table[column] = y_set
    return y_set

# Уточненный м. Эйлера
def improved_Euler(table, column="Уточн. м. Эйлера", h=0.1):
    y_0, x_0 = 0, 1
    n = round((2 - 1) / h)
    y_set = [y_0, y_0 + h * func(x_0, y_0)]
    for i in range(1, n):
        x_i = x_0 + i * h
        y_next = y_set[i - 1] + 2 * h * func(x_i, y_set[i])
        y_set.append(y_next)
    table[column] = y_set
    return y_set

# м. Рунге-Кутта 4 порядка
def Runge_Kutta(table, column="Рунге-Кутта 4 пор.", h=0.1):
    y_0, x_0 = 0, 1
    n = round((2 - 1) / h)
    y_set = [y_0]
    for i in range(n):
        x_1 = x_0 + i * h
        k_1 = func(x_1, y_0)
        k_2 = func(x_1 + h / 4, y_0 + (h / 4) * k_1)
        k_3 = func(x_1 + 0.5 * h, y_0 + 0.5 * h * k_2)
        k_4 = func(x_1 + h, y_0 + h * k_1 - 2 * h * k_2 + 2 * h * k_3)
        y_next = y_0 + h/6 * (k_1 + 4 * k_3 + k_4)
        y_set.append(y_next)
        y_0 = y_next
    table[column] = y_set
    return y_set

# Уточнение по м. Рунге-Ромберга
def Runge_Romberg(table, column, y_h, y_2h, p): # y_h для h=0.2, y_2h для h=0.1
    n = len(y_2h)
    R = [0.0] * n
    # четные узлы
    for i in range(0, n, 2):
        R[i] = (y_2h[i] - y_h[i // 2]) / (2 ** p - 1)
    # нечетные узлы
    for i in range(1, n - 1, 2):
        R[i] = (R[i - 1] + R[i + 1]) / 2
    if n % 2 == 0:
        R[-1] = R[-2]
    # уточненное решение
    y_rr = [y_2h[i] + R[i] for i in range(n)]
    table[column] = y_rr
    return R

# Вычисление погрешности с точным решением
def eps_calc(table, res_column, to_substract):
    table[res_column] = abs(table["Точное решение"] - table[to_substract])
    table[res_column] = table[res_column].apply(lambda x: f"{x:e}")

# Основная часть программы
methods_table = pd.DataFrame(columns=["x", "Точное решение", 
                              "Явный м. Эйлера", "Eps явн. м. Эйлера",
                              "Уточн. м. Эйлера", "Eps уточн. м. Эйлера",
                              "Рунге-Кутта 4 пор.", "Eps м. Рунге-Кутта 4 пор."])
methods_table["x"] = [x/10 for x in range(10, 21)]

exact_solution(methods_table)
explicit_Euler(methods_table)
eps_calc(methods_table, "Eps явн. м. Эйлера", "Явный м. Эйлера")
improved_Euler(methods_table)
eps_calc(methods_table, "Eps уточн. м. Эйлера", "Уточн. м. Эйлера")
Runge_Kutta(methods_table)
eps_calc(methods_table, "Eps м. Рунге-Кутта 4 пор.", "Рунге-Кутта 4 пор.")
print(methods_table, end="\n\n")

# Для h = 0.2:
rename_cols = {"Явный м. Эйлера":"Явн м. Эйлера h=0.1", "Уточн. м. Эйлера":"Уточн. м. Эйлера h=0.1", \
             "Рунге-Кутта 4 пор.":"Рунге-Кутта 4 пор h=0.1"}
RR_table = methods_table.loc[:, ["x", "Точное решение", "Явный м. Эйлера", "Уточн. м. Эйлера", "Рунге-Кутта 4 пор."]].reset_index(drop=True)
RR_table = RR_table.rename(columns=rename_cols)
RR_cols = ["Явн м. Эйлера h=0.2", "Рунге-Ромберг (явн. м. Эйлера)", "Уточн. м. Эйлера h=0.2", \
           "Рунге-Ромберг (уточн. м. Эйлера)", "Рунге-Кутта 4 пор h=0.2", "Рунге-Ромберг (Рунге-Кутта)"]
RR_table[RR_cols] = None

table_h02 = pd.DataFrame({"x": [x/10 for x in range(10, 21, 2)]})

exp_E = explicit_Euler(table_h02, "Явн м. Эйлера h=0.2", 0.2)
imp_E = improved_Euler(table_h02, "Уточн. м. Эйлера h=0.2", 0.2)
rr_m = Runge_Kutta(table_h02, "Рунге-Кутта 4 пор h=0.2", 0.2)

R_exp_E = Runge_Romberg(RR_table, "Рунге-Ромберг (явн. м. Эйлера)", table_h02["Явн м. Эйлера h=0.2"], \
              RR_table["Явн м. Эйлера h=0.1"], 1)
R_imp_E =Runge_Romberg(RR_table, "Рунге-Ромберг (уточн. м. Эйлера)", table_h02["Уточн. м. Эйлера h=0.2"], \
              RR_table["Уточн. м. Эйлера h=0.1"], 2)
R_RRm = Runge_Romberg(RR_table, "Рунге-Ромберг (Рунге-Кутта)", table_h02["Рунге-Кутта 4 пор h=0.2"], \
              RR_table["Рунге-Кутта 4 пор h=0.1"], 4)

RR_table["Уточнение R явн. Э"] = [f'{x:e}' for x in R_exp_E]
RR_table["Уточнение R уточн. Э"] = [f'{x:e}' for x in R_imp_E]
RR_table["Уточнение R Рунге-Кутта"] = [f'{x:e}' for x in R_RRm]

for i in range(0, len(RR_table), 2):
    RR_table.loc[i, "Явн м. Эйлера h=0.2"] = exp_E[i // 2]
    RR_table.loc[i, "Уточн. м. Эйлера h=0.2"] = imp_E[i // 2]
    RR_table.loc[i, "Рунге-Кутта 4 пор h=0.2"] = rr_m[i // 2]

eps_calc(RR_table, "Eps Р-Р явн. м. Эйлера", "Рунге-Ромберг (явн. м. Эйлера)")
eps_calc(RR_table, "Eps Р-Р уточн. м. Эйлера", "Рунге-Ромберг (уточн. м. Эйлера)")
eps_calc(RR_table, "Eps Р-Р Рунге-Кутта", "Рунге-Ромберг (Рунге-Кутта)")

print("Уточнение по Рунге-Ромбергу для Явного м. Эйлера:")
print(RR_table[["x", "Явн м. Эйлера h=0.1", "Явн м. Эйлера h=0.2", "Уточнение R явн. Э", \
                           "Рунге-Ромберг (явн. м. Эйлера)", "Eps Р-Р явн. м. Эйлера"]].fillna("-"), end="\n\n")
print("Уточнение по Рунге-Ромбергу для Уточненного м. Эйлера:")
print(RR_table[["x", "Уточн. м. Эйлера h=0.1", "Уточн. м. Эйлера h=0.2", "Уточнение R уточн. Э",\
                           "Рунге-Ромберг (уточн. м. Эйлера)", "Eps Р-Р уточн. м. Эйлера"]].fillna("-"), end="\n\n")
print("Уточнение по Рунге-Ромбергу для м. Рунге-Кутта 4 порядка:")
print(RR_table[["x", "Рунге-Кутта 4 пор h=0.1", "Рунге-Кутта 4 пор h=0.2", "Уточнение R Рунге-Кутта",\
                           "Рунге-Ромберг (Рунге-Кутта)", "Eps Р-Р Рунге-Кутта"]].fillna("-"), end="\n\n")

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(methods_table["x"], methods_table["Точное решение"], linewidth=3, linestyle="--", label="Точное решение")
plt.plot(methods_table["x"], methods_table["Явный м. Эйлера"], marker="o", label="Явный м. Эйлера")
plt.plot(methods_table["x"], methods_table["Уточн. м. Эйлера"], marker="s", label="Уточнённый м. Эйлера")
plt.plot(methods_table["x"], methods_table["Рунге-Кутта 4 пор."], marker="^", label="м. Рунге-Кутта 4 порядка")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Сравнение методов с точным решением")
plt.grid(True)
plt.legend()
plt.savefig("methods_plot.png", bbox_inches="tight")
plt.show()