import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.animation import FuncAnimation
import matplotlib
import matplotlib.pyplot as plt

# Debug okna z animacja
matplotlib.use('TkAgg')

from grid import Grid


class Solver:
    def __init__(self, grid_obj):
        self.hx = grid_obj.hx
        self.hy = grid_obj.hy
        self.ht = grid_obj.ht
        self.nx = grid_obj.nx
        self.ny = grid_obj.ny
        self.grid = grid_obj.material_grid
        self.N = self.nx * self.ny

    def mapa_alfa(self):
        # TODO wpisać w config
        grid_plaski = self.grid.flatten()
        alfa_powietrze = 0.00025
        alfa_sciana_prz = 0.00000044
        alfa_kartongips = 0.00000008
        alfa_sciana_izo = 0.0000005
        alfa_okno = 0.0000004

        alfa_map = np.ones(self.N) * alfa_powietrze
        alfa_map[grid_plaski == 1] = alfa_sciana_izo
        alfa_map[grid_plaski == 2] = alfa_sciana_prz
        alfa_map[grid_plaski == 3] = alfa_kartongips
        alfa_map[grid_plaski == 4] = alfa_okno
        return alfa_map

    def D2(self, N):
        offsets = [-1, 0, 1]
        data = [np.ones(N), -2 * np.ones(N), np.ones(N)]
        D2 = sp.diags(data, offsets, shape=(N, N), format='lil')
        D2[0, 0] = -1
        D2[0, 1] = 1
        D2[-1, -1] = -1
        D2[-1, -2] = 1
        return D2.tocsr()

    def solve(self, strategia='A'):
        # Mapy parametrów
        grid_flat = self.grid.flatten()
        alfa_map = self.mapa_alfa()

        # Równanie (11): lambda_material / lambda_air
        lambda_air = 0.025
        K_map = np.ones(self.N) * (0.03 / lambda_air)  # Izolacja
        K_map[grid_flat == 2] = 0.5 / lambda_air  # Cegła
        K_map[grid_flat == 4] = 0.8 / lambda_air  # Okno
        K_map[grid_flat == 5] = 0.2 / lambda_air  # Drzwi

        # Krawedzie
        idx_gora = np.arange(0, self.nx)
        idx_dol = np.arange(self.N - self.nx, self.N)
        idx_lewo = np.arange(self.nx, self.N - self.nx, self.nx)
        idx_prawo = np.arange(2 * self.nx - 1, self.N - self.nx, self.nx)

        wszystkie_brzegi_idx = np.unique(np.concatenate([idx_gora, idx_dol, idx_lewo, idx_prawo]))

        # Temperatrury Celcjusz
        T_zew_map = np.ones(self.N) * -5.0  # na zewnątrz TODO dodać implementację pogody
        T_zew_map[idx_lewo] = 19.0  # sąsiad po lewo
        T_zew_map[idx_prawo] = 19.0  # sąsiad po prawo
        T_zew_map[idx_dol] = 10.0  # klatka schodowa

        # Zmiana na Kelwiny
        T_zew_kelwin = T_zew_map + 273.15

        # Laplasjan
        id_Nx = sp.eye(self.nx)
        id_Ny = sp.eye(self.ny)
        D2x = self.D2(self.nx)
        D2y = self.D2(self.ny)
        laplacian = sp.kron(id_Ny, D2x) / self.hx ** 2 + sp.kron(D2y, id_Nx) / self.hy ** 2

        # Macierz A - rzadka
        F = sp.diags(alfa_map * self.ht)
        A = sp.eye(self.N, format='csr') - F.dot(laplacian)
        A = A.tolil()

        dx = self.hx
        # Warunki brzegowe Robina
        A[idx_gora, :] = 0.0
        A[idx_gora, idx_gora] = 1.0 + K_map[idx_gora] * dx
        A[idx_gora, idx_gora + self.nx] = -1.0

        A[idx_dol, :] = 0.0
        A[idx_dol, idx_dol] = 1.0 + K_map[idx_dol] * dx
        A[idx_dol, idx_dol - self.nx] = -1.0

        A[idx_lewo, :] = 0.0
        A[idx_lewo, idx_lewo] = 1.0 + K_map[idx_lewo] * dx
        A[idx_lewo, idx_lewo + 1] = -1.0

        A[idx_prawo, :] = 0.0
        A[idx_prawo, idx_prawo] = 1.0 + K_map[idx_prawo] * dx
        A[idx_prawo, idx_prawo - 1] = -1.0

        A = A.tocsr()  # zmiana na macierz rzadką

        # Wektor b
        b_robin_koniec = np.zeros(self.N)
        b_robin_koniec[wszystkie_brzegi_idx] = (K_map[wszystkie_brzegi_idx] * dx * T_zew_kelwin[wszystkie_brzegi_idx])

        # Indeksy grzejników i powietrza
        maska_grzejniki = (grid_flat == 6)
        maska_powietrze = (grid_flat == 0)

        # Stałe TODO wpisać w config
        p_atm = 101325.0
        r_pow = 287.05
        c_pow = 1005.0

        P_total = 1000.0

        # Moc grzejnika na pixel TODO implementacja termostatu
        n_pix_grzejnik = np.sum(maska_grzejniki)
        pole_pixela = self.hx * self.hy
        P_pixel = P_total / n_pix_grzejnik if n_pix_grzejnik > 0 else 0
        wsp_mocy = (P_pixel * r_pow) / (p_atm * pole_pixela * c_pow)

        # Warunek początkowy
        T = np.ones(self.N) * (19.0 + 273.15)

        rho_pow = 1.2  # gęstość (średnia)

        historia = []
        historia_temp_czujnika = []
        historia_komfortu = []
        historia_energii = []
        calkowita_energia_J = 0

        czas_symulacji = 12 * 3600
        liczba_krokow = int(czas_symulacji / self.ht)

        # Pętla
        for i in range(liczba_krokow):
            b = T.copy()

            czas_kroku = i * self.ht

            # Czujnik temperatury
            center_x = self.nx // 2
            center_y = self.ny // 2

            idx_czujnika = center_y * self.nx + center_x
            temp_czujnika = T[idx_czujnika]

            if strategia == 'A':
                # Stałe grzanie na 21 stopni
                S_termostat = 21.0 + 273.15
            else:
                # Strategia B: wychłodzenie (8h) + dogrzewanie (4h)
                if czas_kroku < 8 * 3600:
                    S_termostat = 7.0 + 273.15
                else:
                    S_termostat = 21.0 + 273.15

            # Termostat + Grzejnik
            if temp_czujnika < S_termostat:
                b[maska_grzejniki] += T[maska_grzejniki] * wsp_mocy * self.ht
                f_grzejnika = T[maska_grzejniki] * wsp_mocy

                # Równanie 16
                calkowita_energia_J += np.sum(f_grzejnika) * (self.hx * self.hy) * self.ht * (rho_pow * c_pow)

            # Warunki brzegowe Robina
            b[wszystkie_brzegi_idx] = b_robin_koniec[wszystkie_brzegi_idx]

            # Rozwiązujemy równanie macierzowe
            T = spla.spsolve(A, b)

            if i % 10 == 0:
                temp_celcjusz_powietrze = T[maska_powietrze] - 273.15

                historia.append(T.copy().reshape(self.ny, self.nx) - 273.15)
                historia_temp_czujnika.append(T[idx_czujnika] - 273.15)
                historia_komfortu.append(np.std(temp_celcjusz_powietrze))
                historia_energii.append(calkowita_energia_J / 3600000)  # zamiana J -> kWh

        return {
            'historia': historia,
            'temp': historia_temp_czujnika,
            'komfort': historia_komfortu,
            'energia': historia_energii,
            'total_kWh': calkowita_energia_J / 3600000
        }


grid_obj = Grid()
solver = Solver(grid_obj)

print("Symulacja Strategii A (Ciągłe grzanie)...")
wyniki_A = solver.solve(strategia='A')

print("Symulacja Strategii B (Oszczędzanie)...")
wyniki_B = solver.solve(strategia='B')

# --- WYKRESY PORÓWNAWCZE ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

v_min, v_max = -5, 40

im1 = ax1.imshow(wyniki_A['historia'][0], cmap='inferno', vmin=v_min, vmax=v_max)
ax1.set_title("Strategia A: Ciągłe grzanie")
plt.colorbar(im1, ax=ax1, label="Temp [°C]")

im2 = ax2.imshow(wyniki_B['historia'][0], cmap='inferno', vmin=v_min, vmax=v_max)
ax2.set_title("Strategia B: Wychłodzenie + Dogrzewanie")
plt.colorbar(im2, ax=ax2, label="Temp [°C]")

# Zaznaczamy czujnik na obu wykresach (środek siatki)
cx, cy = solver.nx // 2, solver.ny // 2
ax1.plot(cx, cy, 'go', markersize=8, label='Czujnik')
ax2.plot(cx, cy, 'go', markersize=8)

# Tekst z czasem i energią
txt_time = fig.suptitle("", fontsize=16)


def update(frame):
    # Aktualizacja obrazów
    im1.set_array(wyniki_A['historia'][frame])
    im2.set_array(wyniki_B['historia'][frame])

    # Aktualizacja nagłówka
    czas_min = (frame * 10 * solver.ht) / 60  # 10 bo historia co 10 kroków
    txt_time.set_text(f"Czas symulacji: {czas_min:.1f} min\n"
                      f"Energia A: {wyniki_A['energia'][frame]:.2f} kWh | "
                      f"Energia B: {wyniki_B['energia'][frame]:.2f} kWh")

    return [im1, im2, txt_time]


# Uruchomienie animacji
# frames to długość krótszej historii (na wypadek różnic)
num_frames = min(len(wyniki_A['historia']), len(wyniki_B['historia']))
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

t_os = np.linspace(0, 12, len(wyniki_A['temp']))
# 1. Wykres Temperatury
plt.subplot(3, 1, 1)
plt.plot(t_os, wyniki_A['temp'], label='Strategia A (Stała)')
plt.plot(t_os, wyniki_B['temp'], label='Strategia B (Wyłączanie)', linestyle='--')
plt.axhline(21, color='red', alpha=0.3, label='Cel')
plt.ylabel('Temp. na czujniku [°C]')
plt.legend()

# 2. Wykres Komfortu (Odchylenie standardowe)
plt.subplot(3, 1, 2)
plt.plot(t_os, wyniki_A['komfort'], label='Komfort A')
plt.plot(t_os, wyniki_B['komfort'], label='Komfort B')
plt.ylabel('Odchylenie $\sigma_u$ [°C]')
plt.title('Nierównomierność rozkładu (im mniej tym lepiej)')
plt.legend()

# 3. Wykres Zużycia Energii
plt.subplot(3, 1, 3)
plt.plot(t_os, wyniki_A['energia'], label=f"Suma A: {wyniki_A['total_kWh']:.2f} kWh")
plt.plot(t_os, wyniki_B['energia'], label=f"Suma B: {wyniki_B['total_kWh']:.2f} kWh")
plt.ylabel('Energia [kWh]')
plt.xlabel('Czas [h]')
plt.legend()

plt.tight_layout()
plt.show()