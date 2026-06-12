# Plan Implementacji: Heat Abnormal (APO Laboratoria 1-4)

## Tło i Motywacja
Projekt wymaga stworzenia aplikacji do przetwarzania obrazów ("Heat Abnormal") przy użyciu języka Rust i biblioteki Slint, spełniającej wymagania zawarte w specyfikacjach PDF dla laboratoriów APO (Algorytmy Przetwarzania Obrazów).

**Ocena Aktualnego Stanu:**
Znaczna część podstawowych wymagań jest już zaimplementowana:
*   **Lab 1:** Wczytywanie/zapis plików, ręczne obliczanie histogramu (ze średnią i odchyleniem standardowym), ręczna equalizacja histogramu oraz podstawowe rozciąganie selektywne.
*   **Lab 2:** Ręczna posteryzacja, filtr medianowy (OpenCV) oraz własny filtr liniowy 3x3 (OpenCV).
*   **Lab 3:** Szkieletyzacja i dylatacja (OpenCV).
*   **Lab 4:** Podstawowe interaktywne progowanie (thresholding).

## Zakres Planu
Plan ten określa kroki niezbędne do ukończenia aplikacji poprzez implementację brakujących funkcji oraz dopracowanie istniejących tak, aby w pełni odpowiadały specyfikacji.

## Proponowane Rozwiązanie: Plan Etapowy

### Etap 1: Lab 1 (Dopracowanie i Brakujące Podstawy)
*   **Przestrzenie Barw:** Dodanie konwersji RGB do HSV, RGB do Lab oraz możliwości rozdzielenia obrazu kolorowego na 3 oddzielne obrazy w skali szarości (kanały R, G, B).
*   **Aktualizacje UI:**
    *   Dodanie okna "O autorze/programie" zgodnie z wymogami.
    *   Aktualizacja UI i logiki dla "Rozciągania Selektywnego", aby poprawnie przyjmowało zakres wejściowy (p1-p2) oraz wyjściowy (q3-q4).
*   **Operacje Punktowe:** Implementacja manualnej operacji "Negacji".

### Etap 2: Lab 2 (Operacje Sąsiedztwa i Dwuargumentowe)
*   **Zdefiniowane Filtry Liniowe:** Rozszerzenie UI o predefiniowane maski: wygładzanie, wyostrzanie (Laplasjan) oraz detekcja krawędzi (Sobel, Canny, Prewitt).
*   **Obsługa Brzegów:** Umożliwienie użytkownikowi wyboru strategii obsługi pikseli brzegowych (isolated, reflect, replicate) dla operacji sąsiedztwa.
*   **Operacje Dwuargumentowe:** Implementacja operacji łączących dwa obrazy: dodawanie, odejmowanie, mieszanie (blending) oraz operacje logiczne (AND, OR, NOT, XOR).
*   **Filtracja Dwuetapowa:** Implementacja mechanizmu tworzenia maski 5x5 z dwóch masek 3x3.

### Etap 3: Lab 3 (Morfologia i Geometria)
*   **Morfologia Matematyczna:** Rozszerzenie narzędzi o Erozję, Otwarcie i Zamknięcie. Możliwość wyboru elementu strukturalnego (romb, kwadrat).
*   **Transformata Hougha:** Detekcja linii prostych przy użyciu OpenCV.
*   **Linia Profilu:** Narzędzie do wizualizacji intensywności pikseli wzdłuż linii wyznaczonej przez użytkownika (algorytm Bresenhama).
*   **Piramida Obrazów:** Operacje skalowania w górę i w dół (piramidy Gaussa/Laplace'a, 2 poziomy).

### Etap 4: Lab 4 (Zaawansowana Segmentacja i Analiza)
*   **Progowanie:** Rozszerzenie segmentacji o progowanie adaptacyjne i metodę Otsu.
*   **Zaawansowana Segmentacja:** Integracja algorytmów GrabCut i Watershed.
*   **Rekonstrukcja i Kompresja:**
    *   Inpainting (naprawianie obrazu).
    *   Kompresja RLE wraz z obliczaniem i wyświetlaniem stopnia kompresji.
*   **Analiza Obiektów:** Obliczanie wektora cech dla obiektów binarnych (Momenty, Pole, Obwód, współczynniki kształtu).

## Weryfikacja
*   **Testy Funkcjonalne:** Sprawdzenie każdego algorytmu na standardowych obrazach testowych (np. Lena).
*   **Zgodność ze Specyfikacją:** Porównanie wyników z przykładami zawartymi w prezentacjach PDF.
