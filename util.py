from numpy import linspace, random, asarray
from math import sqrt, pi
import numpy

# CONSTANTS:
f = 1.
g = 1.26
f2 = 3.706
DELT = 939.565378 - 938.272046
M = (939.565378 + 938.272046) / 2.
M_e = 0.510999
mn = 939.565378
mp = 938.272046
cosUCab = 0.97428
deltInRad = 0.024
Gfermi = 1.166 * 10 ** (-11)
y = (DELT ** 2 - M_e ** 2) / 2.
SIGMA_0 = ((Gfermi ** 2) * cosUCab ** 2) * (1 + deltInRad) / pi

E_nu_TOTAL = 6.2415096516 * 10 ** 5 * 3. * 10 ** 53 / 6.  # MeV
E_nu_AVERAGE = 12.  # MeV
DIST = 3.0856776 * 10 ** 22  # cm


def getSpectrumMaxwBoltz(eNu):
    """
    :param eNu: neutrino energy (MeV)
    :return: Maxwell-Boltzmann spectrum(supernova eNu distribution)
    """
    return 128. / 3. * eNu ** 3 / E_nu_AVERAGE ** 4 * numpy.exp(-4 * eNu / E_nu_AVERAGE)


def getdF_dEnu(eNu):
    """
    :param eNu: neutrino energy (MeV)
    :return: time-integrated flux for a single neutrino flavor (MeV*cm**2)**(-1)
    """
    return 1. / (4 * pi * DIST ** 2) * E_nu_TOTAL / E_nu_AVERAGE * getSpectrumMaxwBoltz(eNu)

def getdSigma_dcos(eNu, cosTeta):
    """
    :param eNu: neutrino energy
    :param cosTeta: cosine of positron angel
    :return: value of differential cross section of inverse beta decay (in unit of cosine)
    """
    e0 = eNu - DELT  # positron energy in null order

    temp = (SIGMA_0 / 2.) * ((f ** 2 + 3 * g * g) + (f ** 2 - g ** 2) * ((((getEnPos(eNu, cosTeta)) ** 2 - M_e ** 2) ** (1. / 2.)) / (getEnPos(eNu, cosTeta))) * cosTeta) \
           * getEnPos(eNu, cosTeta) * ((getEnPos(eNu, cosTeta)) ** 2 - M_e ** 2) ** (1. / 2.) \
           - (SIGMA_0 / 2.) * (getBigGamma(eNu, cosTeta) / M) * e0 * (e0 ** 2 - M_e ** 2) ** (1. / 2.)
    return temp


def getBigGamma(eNu, cosTeta):
    """
    part of dSigma_dcos expression
    :param eNu: neutrino energy
    :param cosTeta: cosine of positron angel
    :return: G
    """
    e0 = eNu - DELT  # positron energy in null order
    ve0 = numpy.sqrt(e0 * e0 - M_e * M_e) / e0  # positron velocity in null order

    G = 2 * (f + f2) * g * ((2 * e0 + DELT) * (1 - ve0 * cosTeta) - (M_e ** 2) / e0) + \
         (f ** 2 + g ** 2) * (DELT * (1 + ve0 * cosTeta) + (M_e ** 2) / e0) + \
         (f ** 2 + 3 * g * g) * ((e0 + DELT) * (1 - (1 / ve0) * cosTeta) - DELT) + \
         (f ** 2 - g ** 2) * ((e0 + DELT) * (1 - (1 / ve0) * cosTeta) - DELT) * ve0 * cosTeta

    return G


def getEnPos(eNu, cosTeta):
    """
    :param eNu: neutrino energy
    :param cosTeta: cosine of positron angel
    :return: positron energy in first order
    """
    e0 = eNu - DELT  # positron energy in null order
    ve0 = numpy.sqrt(numpy.power(e0, 2) - M_e * M_e) / e0  # positron velocity in null order

    enPos = e0 * (1 - (eNu / M) * (1 - ve0 * cosTeta)) - y ** 2 / M
    return enPos


def getKinNeutron(eNu, cosTeta):
    """
    :param eNu: neutrino energy
    :param cosTeta: cosine of positron angel
    :return: kinetic energy of neutron in ibd
    """
    e0 = eNu - DELT  # positron energy in null order
    ve0 = numpy.sqrt(e0 * e0 - M_e * M_e) / e0  # positron velocity in null order

    temp = (eNu * e0 / M) * (1 - ve0 * cosTeta) + y ** 2 / M
    return temp


def getNeutronAngle(eNu, cosTeta):
    """
    :param eNu: neutrino energy
    :param cosTeta: cosine of positron angel
    :return: cosine of neutron angel
    """
    tNeutr = getKinNeutron(eNu, cosTeta)
    ePos = getEnPos(eNu, cosTeta)

    # temp = (self.ev -((e_positr**2-me**2)**(1./2.))*cosTeta)/ \
    # (((t_neutr+mn)**2-mn**2)**(1./2.))
    temp = (mp ** 2 + M_e ** 2 - 2 * mp * ePos - mn ** 2 + 2 * eNu * (tNeutr + mn)) / \
           (2 * ((tNeutr + mn) ** 2 - mn ** 2) ** (1. / 2.) * eNu)
    return temp


def getIbdAll(eNu, cosTeta):
    """
    :param eNu: neutrino energy
    :param cosTeta: cosine of positron angel
    :return:
    """
    eP = getEnPos(eNu, cosTeta)
    kN = getKinNeutron(eNu, cosTeta)
    nAngle = getNeutronAngle(eNu, cosTeta)

    return eP, kN, nAngle


def getdSigma_dEpos(eNu):
    e0 = eNu - DELT  # positron energy in null order
    p0 = numpy.sqrt(e0*e0 - M_e*M_e)

    #return p0 + 2*e0*e0/ (2 * p0)
    return p0*e0


def getEnPosWithCS(eNu, cosTheta):

    dSigma_dEpos = getdSigma_dEpos(eNu)
    enPos = getEnPos(eNu, cosTheta) * dSigma_dEpos

    return enPos