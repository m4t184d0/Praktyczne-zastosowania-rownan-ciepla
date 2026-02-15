# Symulacja Rozchodzenia się Ciepła w Mieszkaniu

Projekt realizuje numeryczne rozwiązanie równania przewodnictwa cieplnego w dwuwymiarowym rzucie mieszkania. Symulacja wykorzystuje **Metodę Różnic Skończonych (FDM)** oraz schemat niejawny, co zapewnia stabilność obliczeń dla dużych kroków czasowych.

Głównym celem projektu jest analiza efektywności energetycznej różnych strategii sterowania ogrzewaniem (np. stały komfort vs tryb Eco) w realistycznym modelu fizycznym uwzględniającym straty ciepła przez ściany i okna.

## 🚀 Główne Funkcjonalności

*   **Solver FDM (Schemat Niejawny):** Rozwiązywanie układu równań liniowych $A \cdot T^{n+1} = b$ z wykorzystaniem macierzy rzadkich (`scipy.sparse`).
*   **Warunki Brzegowe Robina:** Realistyczne modelowanie ucieczki ciepła na styku ściana-otoczenie (zależne od współczynnika przenikania ciepła i temperatury zewnętrznej).
*   **Złożona Geometria:** Obsługa nieregularnych kształtów pomieszczeń, ścian działowych, okien i drzwi zdefiniowanych w pliku konfiguracyjnym.
*   **Modelowanie Materiałów:** Uwzględnienie różnych parametrów fizycznych (przewodność $\lambda$, dyfuzyjność $\alpha$) dla powietrza, cegły, izolacji, szkła itp.
*   **Sterowanie Ogrzewaniem:** Symulacja termostatu z możliwością definiowania harmonogramów grzania (Strategia Komfort vs Strategia Oszczędna).
*   **Pełna Konfiguracja JSON:** Wszystkie parametry symulacji (wymiary, fizyka, warunki początkowe) są oddzielone od kodu.

## 📂 Struktura Projektu

```text
.
├── data/
│   └── config.json       # Główny plik konfiguracyjny (parametry fizyczne, geometria, materiały)
├── pipeline/             # Kod źródłowy symulacji
│   ├── grid.py           # Klasa Grid: wczytywanie geometrii, budowa siatki materiałów
│   └── solver.py         # Klasa Solver: implementacja FDM, budowa macierzy Laplasjana
├── notebooks/            # Raporty i wizualizacje (Jupyter Notebook)
│   ├── Geometria.ipynb      # Wizualizacja układu mieszkania i tabela stałych
│   ├── Analiza_Bledu.ipynb         # Badanie zbieżności siatki (wpływ parametru hx)
│   └── Cieplo.ipynb  # Porównanie strategii grzania + Animacje wyników
├── requirements.txt      # Lista wymaganych bibliotek
└── README.md             # Dokumentacja projektu
```

## 🛠️ Instalacja i Wymagania
Projekt wymaga zainstalowanego środowiska Python (zalecana wersja 3.8+).

Uruchom serwer Jupyter:

`jupyter notebook`

Zainstaluj wymagane biblioteki:

`pip install -r requirements.txt`

## ▶️ Jak uruchomić symulację
Cała analiza i prezentacja wyników odbywa się w notatnikach Jupyter.

Otwórz pliki w folderze notebooks/ w następującej kolejności:

1. Geometria.ipynb: Sprawdź poprawność wczytanej geometrii i materiałów.

2. Analiza_Bledu.ipynb: Uruchom analizę zbieżności, aby uzasadnić dobór kroku przestrzennego hx.

3. Cieplo.ipynb: Przeprowadź główną symulację, porównaj zużycie energii i wygeneruj animacje rozkładu temperatury.

## ⚙️ Konfiguracja (config.json)
Plik data/config.json jest sercem symulacji. Możesz w nim zmieniać parametry bez ingerencji w kod Python:

* **simulation**: Czas trwania symulacji, temperatury docelowe termostatu.

* **config**: Ustawienia siatki numerycznej i wymiary mieszkania.

* **physics**: Stałe termodynamiczne powietrza używane do bilansu energii.

* **materials**: Właściwości fizyczne materiałów.

  * **Lambda**: Przewodność cieplna [W/mK].

  * **Alpha**: Dyfuzyjność cieplna [m²/s] (dla powietrza zastosowano wartość efektywną uwzględniającą konwekcję).
 
* **boundaries**: Temperatury otoczenia (sąsiedzi, klatka schodowa, dwór).
 
* **heater**: Moc grzejników.

* **geometry**: Definicja ścian i obiektów w przestrzeni (współrzędne prostokątów).

## 📊 Przykładowe Wyniki
Projekt pozwala na generowanie wykresów porównawczych zużycia energii oraz komfortu cieplnego dla dwóch scenariuszy:

* **Strategia A**: Utrzymywanie stałej temperatury komfortowej (np. 21°C) przez całą dobę.

* **Strategia B**: Wyłączenie termostatu na czas nieobecności domowników i dogrzewanie po powrocie.

Wyniki symulacji (dostępne w notatniku nr 3) pozwalają oszacować potencjalne oszczędności energii wynikające z inteligentnego sterowania ogrzewaniem.

## 📝 Autor
Mateusz Bado