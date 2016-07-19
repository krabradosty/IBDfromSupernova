from numpy import linspace, random, asarray
import util



class FirstOrder:
    """
    Generate energy and angle in first order
    """
    cosTeta = linspace(-1, 1, 201)

    def __init__(self, eNu):
        self.ev = eNu  # neutrino energy
        self.p_probability = []
        self.p_energyBin = []
        self.init()  # init self.p_probability & self.p_energyBin

    def init(self):
        summ = 0.
        norma = 0.
        init = -1.
        step = 0.01
        dS_dc = util.getdSigma_dcos(self.ev, FirstOrder.cosTeta)
        dS_dc[0] = 0.0
        for i in range(0, FirstOrder.cosTeta.size):
            norma += dS_dc[i]

        for i in range(0, FirstOrder.cosTeta.size):
            summ += dS_dc[i] / norma
            self.p_probability.append(summ)
            self.p_energyBin.append(init + i * step)

    def shootPosAngle(self, n):
        """
        cosine of positron angel generator
        :param n: number of generated angles
        :return: numpy.array of cos positron angels
        """
        angleList = []
        val = random.ranf(n)
        for v in val:  # цикл по множеству необходимых значений косинуса
            for i in range(0, FirstOrder.cosTeta.size):
                if self.p_probability[i] >= v:
                    deltaX = v - self.p_probability[i]
                    y = self.p_energyBin[i] - self.p_energyBin[i - 1]
                    x = self.p_probability[i] - self.p_probability[i - 1]
                    angleList.append(deltaX * y / x + self.p_energyBin[i])
                    if deltaX * y / x + self.p_energyBin[i] < -0.999995:
                        print('Yes, you got it!')
                    if deltaX * y / x + self.p_energyBin[i] > 0.999995:
                        print('Ups, you got it!')
                    break

        return asarray(angleList)
