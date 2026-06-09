import math
import pandas as pd
import matplotlib.pyplot as plt

# Производная
def func(x, y):
    return - pow((y / x), 2) - (y / x) - 1

# Точное решение
def exact_solution(table):
    table["Точное решение"] = table["x"].apply(lambda x: -(x * math.log(x)) / (math.log(x) + 1))
    return table

# Явный м. Эйлера
def explicit_Euler(table, column="Явный м. Эйлера", h=0.1):
    y_0, x_0 = 0, 1
    y_set = [y_0]
    n = round((2 - 1) / h)
    for i in range(n): # считаем до предпоследнего вкл, последний не считаем
        x = x_0 + i * h
        y_next = y_0 + h * func(x, y_0)
        y_set.append(y_next)
        y_0 = y_next
    table[column] = y_set
    return table

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
    return table

# м. Рунге-Кутта
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
    return table

def Runge_Romberg(table, column, y_h, y_h2, p):
    table[column] = y_h2 + (y_h2 - y_h) / (2 ** p - 1)

def error_explicit_Euler(table):
    table["Eps явн. м. Эйлера"] = abs(table["Точное решение"] - table["Явный м. Эйлера"])
    table["Eps явн. м. Эйлера"] = table["Eps явн. м. Эйлера"].apply(lambda x: f"{x:e}")
    return table

def error_improved_Euler(table):
    table["Eps уточн. м. Эйлера"] = abs(table["Точное решение"] - table["Уточн. м. Эйлера"])
    table["Eps уточн. м. Эйлера"] = table["Eps уточн. м. Эйлера"].apply(lambda x: f"{x:e}")
    return table

def error_Runge_Kutta(table):
    table["Eps м. Рунге-Кутта 4 пор."] = abs(table["Точное решение"] - table["Рунге-Кутта 4 пор."])
    table["Eps м. Рунге-Кутта 4 пор."] = table["Eps м. Рунге-Кутта 4 пор."].apply(lambda x: f"{x:e}")
    return table

# Основной код
methods_table = pd.DataFrame(columns=["x", "Точное решение", 
                              "Явный м. Эйлера", "Eps явн. м. Эйлера",
                              "Уточн. м. Эйлера", "Eps уточн. м. Эйлера",
                              "Рунге-Кутта 4 пор.", "Eps м. Рунге-Кутта 4 пор."])
methods_table["x"] = [x/10 for x in range(10, 21)]

exact_solution(methods_table)

explicit_Euler(methods_table)
error_explicit_Euler(methods_table)

improved_Euler(methods_table)
error_improved_Euler(methods_table)

Runge_Kutta(methods_table)
error_Runge_Kutta(methods_table)

print(methods_table, end="\n\n")

# Для h = 0.2:

rename_cols = {"Явный м. Эйлера":"Явн м. Эйлера h=0.1", "Уточн. м. Эйлера":"Уточн. м. Эйлера h=0.1", \
             "Рунге-Кутта 4 пор.":"Рунге-Кутта 4 пор h=0.1"}
RR_table = methods_table.iloc[::2].loc[:, ["x", "Явный м. Эйлера", "Уточн. м. Эйлера", "Рунге-Кутта 4 пор."]].reset_index(drop=True)
RR_table = RR_table.rename(columns=rename_cols)

RR_cols = ["Явн м. Эйлера h=0.2", "Рунге-Ромберг (явн. м. Эйлера)", "Уточн. м. Эйлера h=0.2", \
           "Рунге-Ромберг (уточн. м. Эйлера)", "Рунге-Кутта 4 пор h=0.2", "Рунге-Ромберг (Рунге-Кутта)"]
RR_table[RR_cols] = None

explicit_Euler(RR_table, "Явн м. Эйлера h=0.2", 0.2)
improved_Euler(RR_table, "Уточн. м. Эйлера h=0.2", 0.2)
Runge_Kutta(RR_table, "Рунге-Кутта 4 пор h=0.2", 0.2)

Runge_Romberg(RR_table, "Рунге-Ромберг (явн. м. Эйлера)", RR_table["Явн м. Эйлера h=0.2"], \
              RR_table["Явн м. Эйлера h=0.1"], 1)

Runge_Romberg(RR_table, "Рунге-Ромберг (уточн. м. Эйлера)", RR_table["Уточн. м. Эйлера h=0.2"], \
              RR_table["Уточн. м. Эйлера h=0.1"], 2)

Runge_Romberg(RR_table, "Рунге-Ромберг (Рунге-Кутта)", RR_table["Рунге-Кутта 4 пор h=0.2"], \
              RR_table["Рунге-Кутта 4 пор h=0.1"], 4)

RR_table.reset_index(drop=True)
print("Рунге-Ромберг для Явного м. Эйлера:")
print(RR_table[["x", "Явн м. Эйлера h=0.1", "Явн м. Эйлера h=0.2", \
                           "Рунге-Ромберг (явн. м. Эйлера)"]], end="\n\n")

print("Рунге-Ромберг для Уточненного м. Эйлера:")
print(RR_table[["x", "Уточн. м. Эйлера h=0.1", "Уточн. м. Эйлера h=0.2", \
                           "Рунге-Ромберг (уточн. м. Эйлера)"]], end="\n\n")

print("Рунге-Ромберг для м. Рунге-Кутта 4 порядка:")
print(RR_table[["x", "Рунге-Кутта 4 пор h=0.1", "Рунге-Кутта 4 пор h=0.2", \
                           "Рунге-Ромберг (Рунге-Кутта)"]], end="\n\n")