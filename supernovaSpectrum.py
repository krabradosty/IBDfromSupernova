import numpy
import util
# CONSTANTS
DIST = 3.0856776 * 10 ** 22  # cm
E_nu_TOTAL = 6.2415096516 * 10 ** 5 * 3. * 10 ** 53 / 6.  # MeV
E_nu_AVERAGE = 12.  # MeV
DELT = 939.565378 - 938.272046  # MeV
M_e = 0.510999  # MeV
NUM_p = 9 * 10 ** 31  # numbers of proton's tagerts


class SupernovaSpectrum:
    def __init__(self, eNu):
        self.eNu = eNu
        self.p_probability = []
        self.p_energyBin = []
        self.init()

    def init(self):
        """
        init self.p_probability & self.p_energyBin
        """
        summ = 0.
        norma = 0.
        try:
            init = self.eNu[0]
            step = self.eNu[1] - self.eNu[0]
        except IndexError:
            print('Nevernii format eNu')
            return
        dN_dE = util.getdF_dEnu(self.eNu)
        dN_dE[0] = 0.0
        for i in range(0, self.eNu.size):
            norma += dN_dE[i]
        for i in range(0, self.eNu.size):
            summ += dN_dE[i]/norma
            self.p_probability.append(summ)
            self.p_energyBin.append(init + i*step)

    def shootEnu(self, number):
        """
        :param number: number of request value eNu
        :return: numpy.array of eNu according dF_dEnu (MeV)
        """
        eNu = []
        val = numpy.random.ranf(number)
        for v in val:
            for i in range(0, self.eNu.size):
                if self.p_probability[i] >= v:
                    deltaX = v - self.p_probability[i]
                    y = self.p_energyBin[i] - self.p_energyBin[i-1]
                    x = self.p_probability[i] - self.p_probability[i-1]
                    eNu.append(deltaX*y/x + self.p_energyBin[i])
                    break

        return numpy.asarray(eNu)

