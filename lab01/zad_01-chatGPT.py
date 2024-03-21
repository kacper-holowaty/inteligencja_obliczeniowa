# Biorytmy - kod wygenerowany w stu procentach przez AI (ChatGPT w wersji 3.5)
# rozwiąznie zajęło około 25 minut

import math
from datetime import datetime, timedelta

def calculate_biorhythms_days(birth_date):
    today = datetime.now()
    days_since_birth = (today - birth_date).days
    return days_since_birth

def calculate_biorhythms(days_since_birth):
    physical = math.sin((2 * math.pi / 23) * days_since_birth)
    emotional = math.sin((2 * math.pi / 28) * days_since_birth)
    intellectual = math.sin((2 * math.pi / 33) * days_since_birth)
    return physical, emotional, intellectual

def main():
    name = input("Podaj swoje imię: ")
    year = int(input("Podaj rok urodzenia: "))
    month = int(input("Podaj miesiąc urodzenia (w formacie liczby 1-12): "))
    day = int(input("Podaj dzień urodzenia: "))

    birth_date = datetime(year, month, day)
    days_since_birth = calculate_biorhythms_days(birth_date)
    physical, emotional, intellectual = calculate_biorhythms(days_since_birth)

    print(f"Witaj {name}!")
    print(f"Od Twoich narodzin minęło już {days_since_birth} dni.")
    print("Twoje biorytmy na dziś to:")
    print(f"Fizyczny: {physical}")
    print(f"Emocjonalny: {emotional}")
    print(f"Intelektualny: {intellectual}")

    # Sprawdzanie biorytmów
    if physical > 0.5:
        print("Gratulacje! Twój biorytm fizyczny jest wysoki.")
    elif physical < -0.5:
        next_day_physical = calculate_biorhythms(days_since_birth + 1)[0]
        if next_day_physical > physical:
            print("Nie martw się, następny dzień będzie lepszy dla biorytmu fizycznego.")
        else:
            print("Uwaga! Wartość biorytmu fizycznego na następny dzień będzie niższa.")
    elif -0.5 <= physical <= 0.5:
        print("Twój biorytm fizyczny jest w normie.")

    if emotional > 0.5:
        print("Gratulacje! Twój biorytm emocjonalny jest wysoki.")
    elif emotional < -0.5:
        next_day_emotional = calculate_biorhythms(days_since_birth + 1)[1]
        if next_day_emotional > emotional:
            print("Nie martw się, następny dzień będzie lepszy dla biorytmu emocjonalnego.")
        else:
            print("Uwaga! Wartość biorytmu emocjonalnego na następny dzień będzie niższa.")
    elif -0.5 <= emotional <= 0.5:
        print("Twój biorytm emocjonalny jest w normie.")

    if intellectual > 0.5:
        print("Gratulacje! Twój biorytm intelektualny jest wysoki.")
    elif intellectual < -0.5:
        next_day_intellectual = calculate_biorhythms(days_since_birth + 1)[2]
        if next_day_intellectual > intellectual:
            print("Nie martw się, następny dzień będzie lepszy dla biorytmu intelektualnego.")
        else:
            print("Uwaga! Wartość biorytmu intelektualnego na następny dzień będzie niższa.")
    elif -0.5 <= intellectual <= 0.5:
        print("Twój biorytm intelektualny jest w normie.")

if __name__ == "__main__":
    main()