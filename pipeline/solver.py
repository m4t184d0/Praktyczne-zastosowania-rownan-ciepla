import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from grid import Grid


class Solver:
    def __init__(self, grid_obj):
        # Ładowanie configu
        self.cfg = grid_obj.config

        # Skróty dla czytelności (żeby nie pisać ciągle self.cfg['...'])
        self.c_sim = self.cfg['config']
        self.c_mat = self.cfg['materials']
        self.c_phys = self.cfg['physics']
        self.c_bound = self.cfg['boundaries']
        self.c_set = self.cfg['simulation']

        self.hx = self.c_sim['hx']
        self.hy = self.c_sim['hy']
        self.ht = self.c_sim['ht']
        self.nx = grid_obj.nx
        self.ny = grid_obj.ny
        self.grid = grid_obj.material_grid
        self.N = self.nx * self.ny

    def mapa_alfa(self):
        grid_plaski = self.grid.flatten()

        # Pobieranie wartości z configu
        alfa_powietrze = self.c_mat['0']['alpha']
        alfa_sciana_izo = self.c_mat['1']['alpha']
        alfa_sciana_prz = self.c_mat['2']['alpha']
        alfa_kartongips = self.c_mat['3']['alpha']
        alfa_okno = self.c_mat['4']['alpha']
        # Drzwi i grzejnik też mogą mieć swoją alfę jeśli są w configu
        alfa_drzwi = self.c_mat['5']['alpha']
        alfa_grzejnik = self.c_mat['6']['alpha']

        # Budowanie mapy dokładnie tak jak w Twoim kodzie
        alfa_map = np.ones(self.N) * alfa_powietrze
        alfa_map[grid_plaski == 1] = alfa_sciana_izo
        alfa_map[grid_plaski == 2] = alfa_sciana_prz
        alfa_map[grid_plaski == 3] = alfa_kartongips
        alfa_map[grid_plaski == 4] = alfa_okno
        alfa_map[grid_plaski == 5] = alfa_drzwi
        alfa_map[grid_plaski == 6] = alfa_grzejnik

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
        # Pobieramy lambdy z configu
        lambda_air = self.c_mat['0']['lambda']
        lambda_izo = self.c_mat['1']['lambda']
        lambda_cegla = self.c_mat['2']['lambda']
        lambda_okno = self.c_mat['4']['lambda']
        lambda_drzwi = self.c_mat['5']['lambda']

        # Twoja logika budowania K_map
        K_map = np.ones(self.N) * (lambda_izo / lambda_air)  # Izolacja jako domyślne tło? (tak było w kodzie)
        K_map[grid_flat == 2] = lambda_cegla / lambda_air  # Cegła
        K_map[grid_flat == 4] = lambda_okno / lambda_air  # Okno
        K_map[grid_flat == 5] = lambda_drzwi / lambda_air  # Drzwi

        # Krawedzie
        idx_gora = np.arange(0, self.nx)
        idx_dol = np.arange(self.N - self.nx, self.N)
        idx_lewo = np.arange(self.nx, self.N - self.nx, self.nx)
        idx_prawo = np.arange(2 * self.nx - 1, self.N - self.nx, self.nx)

        wszystkie_brzegi_idx = np.unique(np.concatenate([idx_gora, idx_dol, idx_lewo, idx_prawo]))

        # Temperatrury Celcjusz z configu
        t_out = self.c_bound['temp_outside']
        t_left = self.c_bound['temp_left']
        t_right = self.c_bound['temp_right']
        t_bottom = self.c_bound['temp_bottom']
        t_top = self.c_bound['temp_top']

        T_zew_map = np.ones(self.N) * t_out  # na zewnątrz
        T_zew_map[idx_lewo] = t_left  # sąsiad po lewo
        T_zew_map[idx_prawo] = t_right  # sąsiad po prawo
        T_zew_map[idx_dol] = t_bottom  # klatka schodowa
        T_zew_map[idx_gora] = t_top  # góra (jeśli zdefiniowana)

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
        # Warunki brzegowe Robina - DOKŁADNIE TAK JAK MIAŁEŚ
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

        # Stałe fizyczne z configu
        p_atm = self.c_phys['p_atm']
        r_pow = self.c_phys['R_air']
        c_pow = self.c_phys['c_air']
        rho_pow = self.c_phys['rho_air']

        # Moc z configu (Heater - material 6)
        P_total = self.c_mat['6']['power_total_watts']

        # Moc grzejnika na pixel
        n_pix_grzejnik = np.sum(maska_grzejniki)
        pole_pixela = self.hx * self.hy
        P_pixel = P_total / n_pix_grzejnik if n_pix_grzejnik > 0 else 0
        wsp_mocy = (P_pixel * r_pow) / (p_atm * pole_pixela * c_pow)

        # Warunek początkowy
        t_init = self.c_set['initial_temp_c']
        T = np.ones(self.N) * (t_init + 273.15)

        historia = []
        historia_temp_czujnika = []
        historia_komfortu = []
        historia_energii = []
        calkowita_energia_J = 0

        czas_symulacji = self.c_set['total_time_h'] * 3600
        liczba_krokow = int(czas_symulacji / self.ht)

        # Cele temperatur z configu
        target_temp = self.c_set['target_temp_c'] + 273.15
        eco_temp = self.c_set['eco_temp_c'] + 273.15

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
                # Stałe grzanie
                S_termostat = target_temp
            else:
                # Strategia B: wychłodzenie (8h) + dogrzewanie (4h)
                # Zakładam że 8h to 2/3 czasu symulacji, jeśli chcesz na sztywno 8h to wpisz 8*3600
                if czas_kroku < 8 * 3600:
                    S_termostat = eco_temp
                else:
                    S_termostat = target_temp

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


