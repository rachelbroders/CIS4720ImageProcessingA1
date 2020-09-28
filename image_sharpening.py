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
        print("~~~calc_z_x~~~")
        if m == 0:
            z_x = 2 * self.img_in[n][m] - self.img_in[n][m + 1]
        elif m == self.img_in.shape[1] - 1:
            z_x = 2 * self.img_in[n][m] - self.img_in[n][m - 1]
        else:
            z_x = 2 * self.img_in[n][m] - self.img_in[n][m - 1] - self.img_in[n][m + 1]

        return z_x

    def calc_z_y(self, n, m):
        print("~~~calc_z_y~~~")
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
        print("~~~calc_v~~~")
        v = 0
        range_n = self.get_range(n, self.img_in.shape[0])
        range_m = self.get_range(m, self.img_in.shape[1])
        pixel_support = self.img_in[range_n[0]:range_n[1]+1, range_m[0]:range_m[1]+1]

        for i in range_n:
            for j in range_m:
                v = v + (self.img_in[i][j] - stat.mean(np.ndarray.flatten(pixel_support)))**2

        return v/9.0

    def calc_alpha(self, n, m):
        print("~~~calc_alpha~~~")
        v = self.calc_v(n, m)

        if v < self.tau_1:
            return 1
        elif v < self.tau_2:
            return self.alpha_dh_1 # some value greater than 1
        else:
            return self.alpha_dh_2 # some value between 1 and self.alpha_dh_1

    def apply_g_old(self, n, m, func):
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

    def apply_g(self, n, m, func):
        print("~~~apply_g~~~")
        filter = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        if n == 0:
            if m == 0:
                indices = [[0, 0, 0], [0, func(n, m), func(n, m+1)], [0, func(n+1, m), func(n+1, m+1)]]
            elif m == self.img_in.shape[1] -1:
                indices = [[0, 0, 0], [func(n, m-1), func(n, m), 0], [func(n+1, m-1), func(n+1, m), 0]]
            else:
                indices = [[0, 0, 0], [func(n, m-1), func(n, m), func(n, m+1)], [func(n+1, m-1), func(n+1, m), func(n+1, m+1)]]

        elif n == self.img_in.shape[0] - 1:
            if m == 0:
                indices = [[0, func(n-1, m), func(n-1, m+1)], [0, func(n, m), func(n, m+1)], [0, 0, 0]]
            elif m == self.img_in.shape[1] - 1:
                indices = [[func(n-1, m-1), func(n-1, m), 0], [func(n, m-1), func(n, m), 0], [0, 0, 0]]
            else:
                indices = [[func(n-1, m-1), func(n-1, m), func(n-1, m+1)], [func(n, m-1), func(n, m), func(n, m+1)], [0, 0, 0]]
        else:
            if m == 0:
                indices = [[0, func(n-1, m), func(n-1, m+1)], [0, func(n, m), func(n, m+1)], [0, func(n+1, m), func(n+1, m+1)]]
            elif m == self.img_in.shape[1] - 1:
                indices = [[func(n-1, m-1), func(n-1, m), 0], [func(n, m-1), func(n, m), 0], [func(n+1, m-1), func(n+1, m), 0]]
            else:
                indices = [[func(n-1, m-1), func(n-1, m), func(n-1, m+1)], [func(n, m-1), func(n, m), func(n, m+1)], [func(n+1, m-1), func(n+1, m), func(n+1, m+1)]]
        return np.sum(np. multiply(filter, indices))

    def direct(self, n, m):
        return self.img_in[n][m]

    def apply_g_d(self, n, m):
        print("~~~apply_g_d~~~")
        return self.calc_alpha(n, m)*self.apply_g(n, m, self.direct)

    def apply_g_y(self, n, m):
        print("~~~apply_g_y~~~")
        d = self.calc_delta(n, m)
        G = self.calc_G(n, m)
        g = self.apply_g(n, m, self.direct)
        print("\tself.calc_delta(n, m).shape: ", d.shape)
        print("\tself.calc_G(n, m).shape: ", G.shape)
        print("\tself.apply_g(n, m, self.direct).shape: ", g.shape)
        print("g_y: ", g + np.dot(np.transpose(d), G))
        #return d, G, g
        return int(g + np.dot(np.transpose(d), G))

    def calc_delta(self, n, m):
        print("~~~calc_delta~~~")
        if m != 0:
            d = self.calc_delta(n, m - 1)
            e = self.calc_e(n, m-1)
            R = self.calc_R(n, m-1)
            G = self.calc_G(n, m-1)
            print("\td.shape: ", d.shape)
            print("\te: ", e)
            print("\tR.shape: ", R.shape)
            print("\tG.shape: ", G.shape)

            return d + 2*self.mu*e*np.dot(np.linalg.inv(R), G)
        else:
            return np.array([[1], [1]])

    def calc_G(self, n, m):
        print("~~~calc_G~~~")
        g_z_x = self.apply_g(n, m, self.calc_z_x)
        g_z_y = self.apply_g(n, m, self.calc_z_y)

        return np.array([[g_z_x], [g_z_y]])

    def calc_e(self, n, m):
        print("~~~calc_e~~~")
        return int(self.apply_g_d(n, m) - self.apply_g_y(n, m))

    def calc_R(self, n, m): # Returns 2x2 matrix
        print("~~~calcR~~~")
        if m != 0:
            R = (1 - self.beta)*self.calc_R(n, m-1) + self.beta*np.dot(self.calc_G(n, m), np.transpose(self.calc_G(n, m)))
        else:
            R = np.array([[1, 0], [0, 1]])
            #R = np.transpose(np.array([[1, 0], [0, 1]]))

        print("R: ", R)
        return R

    def calc_y(self, n, m):
        print("~~~calc_y~~~")
        d = self.calc_delta(n, m)

        print("d: ", d)
        print("self.calc_z_x(n, m): ", self.calc_z_x(n, m))
        print("self.calc_z_y(n, m): ", self.calc_z_y(n, m))

        return self.img_in[n][m] + d[0]*self.calc_z_x(n, m) + d[1]*self.calc_z_y(n, m)

    def main(self):
        img_out = np.zeros(shape=self.img_in.shape, dtype=self.img_in.dtype)
        for n in range(self.img_in.shape[0]):
            for m in range(self.img_in.shape[1]):
                print("***y: ", self.calc_y(n, m))
                #img_out[n][m] = self.calc_y(n, m)

        return img_out












