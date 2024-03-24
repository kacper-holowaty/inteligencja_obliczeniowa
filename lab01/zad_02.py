# Strzelanie z trebeusza

import random
from math import pi, sin, cos, sqrt
import matplotlib.pyplot as plt

h = 100
v0 = 50
g = 9.81

cel = random.randint(50, 340)

def stopnie_na_radiany(stopnie):
    return stopnie * (pi / 180)

def trajektoria(x, alfa):
    return (-(g / (2 * v0 ** 2 * cos(stopnie_na_radiany(alfa)) ** 2)) * x ** 2) + ((sin(stopnie_na_radiany(alfa)) / cos(stopnie_na_radiany(alfa))) * x) + h

def wyswietl_trajektorie(alfa, distance):

    x_values = range(int(distance)+10)

    y_values = [trajektoria(x, alfa) for x in x_values]

    y_mniejsze_od_0 = next((idx for idx, y in enumerate(y_values) if y <= 0), None)

    plt.plot(x_values[:y_mniejsze_od_0 ], y_values[:y_mniejsze_od_0 ], label='Trajektoria pocisku')
    plt.xlabel('Odległość (m)')
    plt.ylabel('Wysokość (m)')
    plt.title('Trajektoria pocisku')
    plt.axhline(0, color='black', linewidth=2)
    plt.axvline(x=x_values[y_mniejsze_od_0], color='red', linestyle='--', label=f'Odległość d = {round(distance,2)} m')
    plt.arrow(0, h, v0 * cos(stopnie_na_radiany(alfa)), v0 * sin(stopnie_na_radiany(alfa)), color='orange', width=0.5, head_width=5, head_length=10, linestyle='-', label='v0 = 50 m/s')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

print(f"Cel ustawiony jest w odległości {cel} metrów")

czy_trafiony = False
liczba_prob = 0
while czy_trafiony == False:
    liczba_prob += 1
    alfa = float(input("Podaj wartość kąta pod jakim wystrzelimy pocisk (w stopniach): "))

    distance = (v0 * sin(stopnie_na_radiany(alfa)) + sqrt((v0**2)*(sin(stopnie_na_radiany(alfa))**2)+2*g*h))*((v0*cos(stopnie_na_radiany(alfa)))/g)

    if distance <= cel + 5 and distance >= cel - 5:
        czy_trafiony = True
        print(f"Wystrzelono pocisk na odległośc {round(distance, 2)} metrów i trafiono cel w {liczba_prob} próbie.")

        wyswietl_trajektorie(alfa, distance)

    else:
        print(f"Pudło! Pocisk wystrzelono na odległość {round(distance, 2)} metrów. Spróbuj ponownie.")


