from supernovaSpectrum import *
import util
from numpy import linspace, random, asarray


class IBDFromSupernova:

    enNuList = linspace(0, 50, 200)
    cosPosList = linspace(-1, 1, 201)

    def __init__(self):
        self.supernovaGen = SupernovaSpectrum(IBDFromSupernova.enNuList)
        self.ibdGen = None
        self.cosPos = []
        self.enPos = []
        self.cosN = []
        self.kinN = []

    def shoot(self, num):
        enNuArray = self.supernovaGen.shootEnu(num)
        for enNu in enNuArray:
            if enNu < 1.806:
                continue

            cosPos = self.shootPosAngle(enNu)
            enPos, kinN, cosN = util.getIbdAll(enNu, cosPos)
            self.cosPos.append(cosPos)
            self.enPos.append(enPos)
            self.cosN.append(cosN)
            self.kinN.append(kinN)
        print('Complete!')

    def get(self):
        return self.cosPos, self.enPos, self.cosN, self.kinN

    def shootPosAngle(self, enNu):
        p_probability = []
        p_energyBin = []
        summ = 0.
        norma = 0.
        init = IBDFromSupernova.cosPosList[0]
        step = IBDFromSupernova.cosPosList[1] - IBDFromSupernova.cosPosList[0]

        dN_dc = util.getdSigma_dcos(enNu, IBDFromSupernova.cosPosList) * util.getdF_dEnu(enNu)
        dN_dc[0] = 0.0
        for i in range(0, IBDFromSupernova.cosPosList.size):
            norma += dN_dc[i]

        for i in range(0, IBDFromSupernova.cosPosList.size):
            summ += dN_dc[i] / norma
            p_probability.append(summ)
            p_energyBin.append(init + i * step)

        val = random.rand()
        angle = None
        for i in range(0, IBDFromSupernova.cosPosList.size):
            if p_probability[i] >= val:
                deltaX = val - p_probability[i]
                y = p_energyBin[i] - p_energyBin[i - 1]
                x = p_probability[i] - p_probability[i - 1]
                angle = (deltaX * y / x + p_energyBin[i])
                break

        return angle




