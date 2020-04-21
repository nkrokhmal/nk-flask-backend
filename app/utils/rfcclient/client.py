import math
import scipy
import scipy.special as special
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
import scipy.io as sio
from scipy.fftpack import fftn, ifftn
import matplotlib.animation as animation
from scipy.special import factorial


class Object:
    def __init__(self, a=0.0001, rho=1125., c_l=2620., c_t=1080., k_l=0., k_t=0.):
        self.a = a
        self.rho = rho
        self.c_l = c_l
        self.c_t = c_t
        self.sigma = (c_l ** 2 / 2 - c_t ** 2) / (c_l ** 2 - c_t ** 2)
        self.k_l = k_l
        self.k_t = k_t


class Wave:
    def __init__(self, f=1.000e6, c=1500., rho=1000.):
        self.f = f
        self.c = c
        self.k = 2 * math.pi * f / c
        self.rho = rho


class Spectrum:
    def __init__(self, dx=0.00025, dk=0, Nx=0, Ny=0):
        self.dx = dx
        self.dk = dk
        self.Nx = Nx
        self.Ny = Ny


class Coordinates:
    def __init__(self, x=np.array([0.]), y=np.array([0.0]), z=np.arange(-0.02, 0.02, 0.001)):
        self.x = x
        self.y = y
        self.z = z
        self.N_points = x.shape[0] * y.shape[0] * z.shape[0]
        self.nx = x.shape[0]
        self.ny = y.shape[0]
        self.nz = z.shape[0]

class Points:
    def __init__(self, coordinates, obj, wave, spectrum, path):
        self.coordinates = coordinates
        self.wave = wave
        self.obj = obj
        self.spectrum = spectrum
        self.obj.k_l = 2 * math.pi * self.wave.f / self.obj.c_l
        self.obj.k_t = 2 * math.pi * self.wave.f / self.obj.c_t
        self.n_global = int(3 * np.ceil(self.wave.k * self.obj.a))
        self.n = np.arange(self.n_global + 1)
        self.path = path
        self.points = self.init_points()
        self.init_dif_bessel_kla()
        self.init_dif_bessel_kta()
        self.init_dif_bessel_ka()
        self.init_bessel()
        self.init_c_n()
        self.init_angle_spectrum()
        self.init_wave_number_space()
        self.init_spherical_legandre()

    def init_points(self):
        points = np.array(
            [[x, y, z] for x in self.coordinates.x for y in self.coordinates.y for z in self.coordinates.z])
        return points

    def calculate_force(self):
        force = np.zeros((self.coordinates.N_points, 3)) * 1.0
        return force

    def init_dif_bessel_kla(self):
        self.dif_bessel_kla = self.n * (0. + 1j * 0.)
        self.dif2_bessel_kla = self.n * (0. + 1j * 0.)

    def init_dif_bessel_kta(self):
        self.dif_bessel_kta = self.n * (0. + 1j * 0.)
        self.dif2_bessel_kta = self.n * (0. + 1j * 0.)

    def init_dif_bessel_ka(self):
        self.dif_bessel_ka = self.n * (0. + 1j * 0.)
        self.dif2_bessel_ka = self.n * (0. + 1j * 0.)

    def init_bessel(self):
        self.sph_bessel_kla = scipy.special.spherical_jn(self.n, self.obj.k_l * self.obj.a)
        self.sph_bessel_kta = scipy.special.spherical_jn(self.n, self.obj.k_t * self.obj.a)
        self.sph_bessel_ka = scipy.special.spherical_jn(self.n, self.wave.k * self.obj.a)
        self.sph_hankel = scipy.special.spherical_jn(self.n,
                                                     self.wave.k * self.obj.a) + 1j * scipy.special.spherical_yn(self.n,
                                                                                                                 self.wave.k * self.obj.a)

        self.dif_hankel = self.n * (0. + 1j * 0.)

        self.dif_bessel_kla[0] = - self.sph_bessel_kla[1]
        self.dif_bessel_kta[0] = - self.sph_bessel_kta[1]
        self.dif_bessel_ka[0] = - self.sph_bessel_ka[1]
        self.dif_hankel[0] = - self.sph_hankel[1]

        for k in range(1, self.n_global + 1):
            self.dif_bessel_kla[k] = self.sph_bessel_kla[k - 1] - (k + 1) / (self.obj.k_l * self.obj.a) * \
                                     self.sph_bessel_kla[k]
            if self.obj.k_t == 0:
                self.dif_bessel_kta[k] = self.sph_bessel_kta[k - 1]
            else:
                self.dif_bessel_kta[k] = self.sph_bessel_kta[k - 1] - (k + 1) / (self.obj.k_t * self.obj.a) * \
                                         self.sph_bessel_kta[k]
            self.dif_bessel_ka[k] = self.sph_bessel_ka[k - 1] - (k + 1) / (self.wave.k * self.obj.a) * \
                                    self.sph_bessel_ka[k]
            self.dif_hankel[k] = self.sph_hankel[k - 1] - (k + 1) / (self.wave.k * self.obj.a) * self.sph_hankel[k]

        for i in range(0, self.n_global):
            self.dif2_bessel_kla[i] = i / (2 * i + 1) * self.dif_bessel_kla[i - 1] - (i + 1) / (2 * i + 1) * \
                                      self.dif_bessel_kla[i + 1]
            self.dif2_bessel_kta[i] = i / (2 * i + 1) * self.dif_bessel_kta[i - 1] - (i + 1) / (2 * i + 1) * \
                                      self.dif_bessel_kta[i + 1]
            self.dif2_bessel_ka[i] = i / (2 * i + 1) * self.dif_bessel_ka[i - 1] - (i + 1) / (2 * i + 1) * \
                                     self.dif_bessel_ka[i + 1]

    def init_c_n(self):
        alf = self.sph_bessel_kla - self.obj.k_l * self.obj.a * self.dif_bessel_kla
        bet = (self.n ** 2 + self.n - 2) * self.sph_bessel_kta + (self.obj.k_t * self.obj.a) ** 2 * self.dif2_bessel_kta
        delta = 2 * self.n * (self.n + 1) * self.sph_bessel_kta
        ksi = self.obj.k_l * self.obj.a * self.dif_bessel_kla
        nu = 2 * self.n * (self.n + 1) * (self.sph_bessel_kta - self.obj.k_t * self.obj.a * self.dif_bessel_kta)
        eps = self.obj.k_l ** 2 * self.obj.a ** 2 * (
                    self.sph_bessel_kla * self.obj.sigma / (1 - 2 * self.obj.sigma) - self.dif2_bessel_kla)

        G = self.wave.rho * self.obj.k_t ** 2 * self.obj.a ** 2 / 2 / self.obj.rho * (alf * delta + bet * ksi) / (
                    alf * nu + bet * eps)
        self.c_n = - (G * self.sph_bessel_ka - self.wave.k * self.obj.a * self.dif_bessel_ka) / (
                    G * self.sph_hankel - self.wave.k * self.obj.a * self.dif_hankel)

    def load_file(self):
        mat_fname = pjoin(self.path)
        mat_contents = sio.loadmat(mat_fname)
        return mat_contents

    '''to do: spectrum'''

    def init_angle_spectrum(self):
        mat_contents = self.load_file()
        keys = sorted(mat_contents.keys())
        pressure_field = mat_contents[keys[0]]
        angle_spectrum_0 = fftn(pressure_field, pressure_field.shape)
        self.spectrum.Nx = angle_spectrum_0.shape[0]
        self.spectrum.Ny = angle_spectrum_0.shape[1]

        lin_l = (-1) ** np.arange(self.spectrum.Nx)
        lin_l = lin_l[:, np.newaxis]
        lin_m = lin_l.T
        lin_lm = lin_l @ lin_m
        self.angle_spectrum = angle_spectrum_0.conj() * lin_lm

    '''to do: split arrays and angles'''

    def init_wave_number_space(self):
        if (-1) ** self.spectrum.Nx > 0:
            x_array = np.concatenate([np.arange(self.spectrum.Nx / 2 + 1.), np.arange(- self.spectrum.Nx / 2 + 1., 0.)])
        elif (-1) ** self.spectrum.Nx < 0:
            x_array = np.concatenate(
                [np.arange((self.spectrum.Nx + 1.) / 2), np.arange(- (self.spectrum.Nx - 1.) / 2, 0.)])
        x_array = x_array[:, np.newaxis]
        x_array = self.spectrum.dx * x_array.astype(float)
        y_array = x_array.copy()
        y_array = y_array.T
        self.r_array = np.sqrt(x_array ** 2 + y_array ** 2)
        self.r_array[0, 0] = 1e-12

        self.spectrum.dk = 2 * math.pi / (self.spectrum.dx * self.spectrum.Nx)
        self.kx_array = self.spectrum.dk / self.spectrum.dx * x_array.copy()
        self.ky_array = self.spectrum.dk / self.spectrum.dx * y_array.copy()
        self.kr_array = np.sqrt(self.kx_array ** 2 + self.ky_array ** 2)
        self.kr_array[0, 0] = 1e-6

        self.k_window = 0.5 * (np.sign(self.wave.k - self.kr_array - 0.0001) + 1)

        kr2 = np.float_power(self.kr_array, 2)
        kr22 = (self.wave.k ** 2 - kr2) * self.k_window
        self.kz_array = np.float_power(kr22, 1 / 2)

        self.phi_k = (math.pi + np.arctan2(self.kx_array, self.ky_array)) * self.k_window
        self.cos_th_k = np.float_power((1 - self.kr_array ** 2 / self.wave.k ** 2) * self.k_window, 0.5)
        self.th_k = np.arccos(self.cos_th_k) * self.k_window

        self.phi = (math.pi + np.arctan2(x_array, y_array))
        self.cos_th = 0 * self.r_array
        self.th = np.arccos(self.cos_th)

    def calculate_spharm(self, n, m, Y_sph):
        if m >= 0:
            SP = Y_sph[n, m]
        elif m < 0:
            SP = np.conj(Y_sph[n, -m]) * (- 1) ** (- m)
        return SP


    def init_spherical_legandre(self):
        Nx = np.size(self.th_k, 0)
        Ny = np.size(self.th_k, 1)

        P = np.zeros((self.n_global + 1, self.n_global + 1, Nx, Ny)) * (0. + 1j * 0)
        self.Y_sph = P.copy()
        Kn = P.copy()

        for nn in range(self.n_global + 1):
            for mm in range(nn + 1):
                K = ((2 * nn + 1) / 4 / math.pi * factorial(nn - mm, exact=True) / factorial(nn + mm, exact=True)) ** (
                            1 / 2)
                Kn[nn, mm, :, :] = K * np.exp(1j * (mm * self.phi_k))
                P[nn, mm, :, :] = scipy.special.lpmv(mm, nn, self.cos_th_k)
        self.Y_sph = P * Kn

        P_sc = np.zeros((self.n_global + 1, self.n_global + 1, Nx, Ny)) * (0. + 1j * 0)
        self.Y_sph_sc = P_sc.copy()
        Kn_sc = P_sc.copy()

        for nn in range(self.n_global + 1):
            for mm in range(nn + 1):
                K_sc = ((2 * nn + 1) / 4 / math.pi * factorial(nn - mm, exact=True) / factorial(nn + mm,
                                                                                                exact=True)) ** (1 / 2)
                Kn_sc[nn, mm, :, :] = K_sc * np.exp(1j * (mm * self.phi))
                P_sc[nn, mm, :, :] = scipy.special.lpmv(mm, nn, self.cos_th)
        self.Y_sph_sc = P_sc * Kn_sc

    def calculate_force(self):
        force = np.zeros((self.coordinates.N_points, 3)) * 1.0
        Nx = np.size(self.th_k, 0)
        Ny = np.size(self.th_k, 1)
        scat_p = np.zeros((self.coordinates.N_points, Nx, Ny)) * (1 + 1j)
        p_newpoint = scat_p.copy()
        for glob_n in range(self.coordinates.N_points):
            H_nm = np.zeros((self.n_global + 1, 2 * self.n_global + 1)) * (0. + 1j * 0.)
            phase_mult = np.exp(1j * self.kx_array * self.points[glob_n, 0] + 1j * self.ky_array * self.points[
                glob_n, 1] + 1j * self.kz_array * self.points[glob_n, 2])
            s_newpoint = phase_mult * self.angle_spectrum * self.k_window

            p = ifftn(s_newpoint, s_newpoint.shape)
            p = np.fft.fftshift(p, axes=0)
            p = np.fft.fftshift(p, axes=1)
            p_newpoint[glob_n, :, :] = np.flip(p)

            tmp = self.calculate_spharm(0, 0, self.Y_sph).conj() * s_newpoint
            H_nm[0, 0] = tmp.sum(axis=1).sum()

            for nn in range(1, self.n_global + 1):
                for mm in range(- nn, nn + 1):
                    tmp = self.calculate_spharm(nn, mm, self.Y_sph).conj() * s_newpoint
                    H_nm[nn, mm] = tmp.sum(axis=1).sum()

            sc_tmp2 = self.r_array * 0
            sc_tmp = self.r_array * 0
            for nn in range(0, self.n_global + 1):
                for mm in range(- nn, nn + 1):
                    sc_tmp = sc_tmp + H_nm[nn, mm] * self.calculate_spharm(nn, mm, self.Y_sph_sc)
                sc_tmp2 = sc_tmp2 + sc_tmp * (1j ** nn) * (
                            (1 + self.c_n[nn]) * scipy.special.spherical_jn(nn, self.wave.k * self.r_array) +
                            self.c_n[nn] * 1j * scipy.special.spherical_yn(nn, self.wave.k * self.r_array))
                sc_tmp = sc_tmp * 0
            o_window1 = 0.5 * (np.sign(self.obj.a - self.r_array) - 1)
            o_window2 = 0.5 * (np.sign(3 * self.obj.a - self.r_array) + 1)
            scat = 1 / np.pi * sc_tmp2 * o_window1 * o_window2
            scat1 = np.fft.fftshift(scat, axes=0)
            scat2 = np.fft.fftshift(scat1, axes=1)
            scat_p[glob_n, :, :] = np.flip(scat2)
            f_nx = np.zeros((self.n_global, 1)) * (0. + 1j * 0.)
            f_ny = np.zeros((self.n_global, 1)) * (0. + 1j * 0.)
            f_nz = np.zeros((self.n_global, 1)) * (0. + 1j * 0.)
            for nn in range(self.n_global):
                psi = (1 + 2 * self.c_n[nn]) * (1 + 2 * self.c_n[nn + 1].conj()) - 1
                f_x = np.zeros((2 * self.n_global + 1, 1)) * (0. + 1j * 0.)
                f_y = f_x.copy()
                f_z = f_x.copy()
                for mm in range(-nn, nn + 1):
                    A_nm = np.sqrt((nn + mm + 1) * (nn + mm + 2) / (2 * nn + 1) / (2 * nn + 3))
                    B_nm = np.sqrt((nn + mm + 1) * (nn - mm + 1) / (2 * nn + 1) / (2 * nn + 3))
                    f_x[mm + nn] = A_nm * (H_nm[nn, mm] * H_nm[nn + 1, mm + 1].conj() - H_nm[nn, - mm] * H_nm[
                        nn + 1, - mm - 1].conj())

                    f_y[mm + nn] = A_nm * (H_nm[nn, mm] * H_nm[nn + 1, mm + 1].conj() + H_nm[nn, - mm] * H_nm[
                        nn + 1, - mm - 1].conj())
                    f_z[mm + nn] = B_nm * H_nm[nn, mm] * H_nm[nn + 1, mm].conj()
                f_nx[nn] = psi * f_x.sum()
                f_ny[nn] = psi * f_y.sum()
                f_nz[nn] = psi * f_z.sum()

            coef = 1 / 8 / math.pi ** 2 / self.wave.rho / self.wave.c ** 2 / self.wave.k ** 2
            force_x = f_nx.sum()
            force_x = coef * force_x.real
            force_y = f_ny.sum()
            force_y = coef * force_y.imag
            force_z = f_nz.sum()
            force_z = -2 * coef * force_z.real

            force[glob_n] = [force_x, force_y, force_z]
        return force, force_x, force_y, force_z, scat_p

    def build_rad_force(self, force):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.points[:, 2] * 1000, 0 * force[:, 0])
        ax.plot(self.points[:, 2] * 1000, force[:, 1])
        ax.plot(self.points[:, 2] * 1000, force[:, 2])

        ax.legend(['F_x', 'F_y', 'F_z'], loc='upper right', shadow=True)
        ax.set_xlabel('z, mm')
        ax.set_ylabel('Radiation force, N')
        ax.set_title('Radiation force on z-axis')
        return fig

    def build_animation(self, scat_p):
        fig, ax = plt.subplots()

        def animate(i):
            s = np.abs(scat_p[int(i), :, :])
            s[0, 0] = np.max(np.max(np.max(np.abs(scat_p), axis=0), axis=1))
            im = ax.matshow(s)
            return im

        sin_animation = animation.FuncAnimation(fig,
                                                animate,
                                                frames=np.linspace(0, 39, 40),
                                                interval=400,
                                                repeat=True)
        return sin_animation