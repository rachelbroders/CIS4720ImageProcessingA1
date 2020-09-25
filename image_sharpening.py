import numpy as np
import cv2
import os
import statistics as stat


# Note that sets of 3 values are BGR values - blue, green and red

########################################################################################################################
#                     Image Enhancement via Adaptive Unsharp Masking
########################################################################################################################
class adaptiveUnsharpMasking(object):
    def __init__(self, img_in, tau_1, tau_2, alpha_dh_1, alpha_dh_2, mu, beta):
        self.img_in = img_in
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.alpha_dh_1 = alpha_dh_1
        self.alpha_dh_2 = alpha_dh_2
        self.mu = mu
        self.beta = beta

    def calc_z_x(self, n, m):
        if m == 0:
            z_x = 2 * self.img_in[n][m] - self.img_in[n][m + 1]
        elif m == self.img_in.shape[1] - 1:
            z_x = 2 * self.img_in[n][m] - self.img_in[n][m - 1]
        else:
            z_x = 2 * self.img_in[n][m] - self.img_in[n][m - 1] - self.img_in[n][m + 1]

        return z_x

    def calc_z_y(self, n, m):
        if n == 0:
            z_y = 2 * self.img_in[n][m] - self.img_in[n - 1][m] - self.img_in[n + 1][m]
        elif n == self.img_in.shape[0] - 1:
            z_y = 2 * self.img_in[n][m] - self.img_in[n - 1][m]
        else:
            z_y = 2 * self.img_in[n][m] - self.img_in[n - 1][m] - self.img_in[n + 1][m]

        return z_y

    def get_range(self, nm, length):
        if nm == 0:
            range_nm = [nm, nm+1]
        elif nm == length - 1:
            range_nm = [nm-1, nm]
        else:
            range_nm = [nm-1, nm, nm+1]

        return range_nm

    def calc_v(self, n, m):
        v = 0
        range_n = self.get_range(n, self.img_in.shape[0])
        range_m = self.get_range(m, self.img_in.shape[1])

        for i in range_n:
            for j in range_m:
                v = v + (self.img_in[i][j] - stat.mean(np.ndarray.flatten(self.img_in)))**2

        return v/9.0

    def calc_alpha(self, n, m):
        v = self.calc_v(n, m)

        if v < self.tau_1:
            return 1
        elif v < self.tau_2:
            return self.alpha_dh_1 # some value greater than 1
        else:
            return self.alpha_dh_2 # some value between 1 and self.alpha_dh_1

    def apply_g(self, n, m, func):
        if n == 0:
            if m == 0:
                filter = [[8, -1], [-1, -1]]
                indices = [[func(n, m), func(n, m+1)], [func(n+1, m), func(n+1, m+1)]]
            elif m == self.img_in.shape[1] -1:
                filter = [[-1, 8], [-1, -1]]
                indices = [[func(n, m-1), func(n, m)], [func(n+1, m-1), func(n+1, m)]]
            else:
                filter = [[-1, 8, -1], [-1, -1, -1]]
                indices = [[func(n, m-1), func(n, m), func(n, m+1)], [func(n+1, m-1), func(n+1, m), func(n+1, m+1)]]

        elif n == self.img_in.shape[0] - 1:
            if m == 0:
                filter = [[-1, -1], [8, -1]]
                indices = [[func(n-1, m), func(n-1, m+1)], [func(n, m), func(n, m+1)]]
            elif m == self.img_in.shape[1] - 1:
                filter = [[-1, -1], [-1, 8]]
                indices = [[func(n-1, m-1), func(n-1, m)], [func(n, m-1), func(n, m)]]
            else:
                filter = [[-1, -1, -1], [-1, 8, -1]]
                indices = [[func(n-1, m-1), func(n-1, m), func(n-1, m+1)], [func(n, m-1), func(n, m), func(n, m+1)]]
        else:
            if m == 0:
                filter = [[-1, -1], [8, -1], [-1, -1]]
                indices = [[func(n-1, m), func(n-1, m+1)], [func(n, m), func(n, m+1)], [func(n+1, m), func(n+1, m+1)]]
            elif m == self.img_in.shape[1] - 1:
                filter = [[-1, -1], [-1, 8], [-1, -1]]
                indices = [[func(n-1, m-1), func(n-1, m)], [func(n, m-1), func(n, m)], [func(n+1, m-1), func(n+1, m)]]
            else:
                filter = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
                indices = [[func(n-1, m-1), func(n-1, m), func(n-1, m+1)], [func(n, m-1), func(n, m), func(n, m+1)], [func(n+1, m-1), func(n+1, m), func(n+1, m+1)]]

        return np.dot(filter, indices)

    def direct(self, n, m):
        return self.img_in[n][m]

    def apply_g_d(self, n, m):
        return self.calc_alpha(n, m)*self.apply_g(n, m, self.direct)

    def apply_g_y(self, n, m):
        return self.apply_g(n, m, self.direct) + np.dot(np.transpose(self.calc_delta(n, m)), self.calc_G(n, m))

    def calc_delta(self, n, m):
        if m != 0:
            return self.calc_delta(n, m - 1) + 2*self.mu*self.calc_e(n, m-1)*np.linalg.inv(self.calc_R(n, m-1))*self.calc_G(n, m-1)
        else:
            return np.transpose(np.array([1, 1]))

    def calc_G(self, n, m):
        g_z_x = self.apply_g(n, m, self.calc_z_x)
        g_z_y = self.apply_g(n, m, self.calc_z_y)

        return np.transpose([g_z_x, g_z_y])

    def calc_e(self, n, m):
        return self.apply_g_d(n, m) - self.apply_g_y(n, m)

    def calc_R(self, n, m):
        if m != 0:
            return (1 - self.beta)*self.calc_R(n, m-1) + self.beta*np.dot(self.calc_G(n, m), np.linalg.inv(self.calc_G(n, m)))
        else:
            return np.transpose(np.array([1, 1]))
    def calc_y(self, n, m):
        d = self.calc_delta(n, m)
        return self.img_in[n][m] + d[0]*self.calc_z_x(n, m) + d[1]*self.calc_z_y(n, m)

    def main(self):
        img_out = np.zeros(shape=self.img_in.shape, dtype=self.img_in.dtype)
        for n in range(self.img_in.shape[0]):
            for m in range(self.img_in.shape[1]):
                img_out[n][m] = self.calc_y(n, m)

        return img_out












