# Biorytmy - rozwiązanie samodzielne. Czas przeznaczony na rozwiązanie: około 45 minut

from math import sin, pi
from datetime import date

def fizyczna_fala(t):
    return sin(((2 * pi)/23)*t)

def emocjonalna_fala(t):
    return sin(((2 * pi)/28)*t)

def intelektualna_fala(t):
    return sin(((2 * pi)/33)*t)

def main():
    imie = input("Jak masz na imię? ")
    rok_urodzenia = int(input("Podaj rok urodzenia: "))
    miesiac_urodzenia = int(input("Podaj miesiąc urodzenia (numer 1-12): "))
    dzien_urodzenia = int(input("Podaj dzień urodzenia: "))

    data_urodzenia = date(rok_urodzenia, miesiac_urodzenia, dzien_urodzenia)
    dzisiaj = date.today()
    liczba_dni = (dzisiaj - data_urodzenia).days
    fizyczna = fizyczna_fala(liczba_dni)
    emocjonalna = emocjonalna_fala(liczba_dni)
    intelektualna = intelektualna_fala(liczba_dni)

    print(f"Witaj {imie}! Dziś jest twój {liczba_dni} dzień życia.")
    print(f"Analizując biorytmy mamy:\nFala fizyczna: {fizyczna},\nFala emocjonalna: {emocjonalna},\nFala intelektualna: {intelektualna}")
    if fizyczna > 0.5:
        print("Świetny wynik! Masz w sobie dużo energii i siły 💪")
    if fizyczna <= -0.5:
        jutro = fizyczna_fala(liczba_dni+1)
        if jutro > fizyczna:
            print("Nie przejmuj się, jutro poczujesz się fizycznie lepiej!")
        else:
            print("Trzymaj się! Jutrzejszy dzień będzie dla ciebie jeszcze cięższy fizycznie🫂")
    if -0.5 <= fizyczna <= 0.5:
        print("Jesteś w fizycznej równowadze 🧘‍♂️")
    if emocjonalna > 0.5:
        print("Tak trzymaj! Czujesz się świetnie emocjonalnie!😀")
    if emocjonalna < -0.5:
        jutro = emocjonalna_fala(liczba_dni+1)
        if jutro > emocjonalna:
            print("Nie przejmuj się, jutro będziesz w lepszym nastroju! ")
        else:
            print("Trzymaj się! Jutro będziesz w jeszcze gorszym nastroju :(")
    if -0.5 <= emocjonalna <= 0.5:
        print("Jesteś w emocjonalnej równowadze 🧘‍♂️")
    if intelektualna > 0.5:
        print("Świetny wynik! Jesteś w topowej formie intelektualnej! 🧠")
    if intelektualna < -0.5:
        jutro = intelektualna_fala(liczba_dni+1)
        if jutro > intelektualna:
            print("Nic się nie dzieje, jutro będziesz w lepszej formie inteletualnej.")
        else:
            print("Trzymaj się! Następnego dnia będziesz w jeszcze gorszej formie intelektualnej 🫂")
    if -0.5 <= intelektualna <= 0.5:
        print("Jesteś w intelektualnej równowadze 🧘‍♂️")

main()
