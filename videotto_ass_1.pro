;+
; :Author: Alex
;-
PRO videotto_ass_1

kb = 1.38064852D-23
bigG = 6.67408D-11
eRad = 6.378D8 / 100
R_exo_e = eRad + (5D7/100)
mRad = 1.737D8 / 100
R_exo_m = mRad

eMass = 5.974D27 / 1000
mMass = 7.35D25 / 1000

tsun = 5778D
R_sun = 69.5508D9 / 100
d_moon = 1.496D13 / 100
moonAlb = 0.12

T_exo_e = 1000.
T_exo_m = tsun * SQRT(R_sun/d_moon) * ((1 - moonAlb)/4.)^(1./4)


meanMolWeight = FINDGEN(44,START = 1)
meanMolMass = meanMolWeight / (1000*6.022140857D23)

lambdaEscMoon = (bigG * mMass * meanMolMass) / (R_exo_m * T_exo_m * kb)
v0Moon = SQRT((2*kb*T_exo_m)/(meanMolMass))


lambdaEscEarth = (bigG * eMass * meanMolMass) / (R_exo_e * T_exo_e * kb)
v0Earth = SQRT((2*kb*T_exo_e)/(meanMolMass))

phij_per_N_Moon = (1./(2*SQRT(!DPI))) * v0Moon * (1 + lambdaEscMoon) * EXP(-lambdaEscMoon)
phij_per_N_Earth = (1./(2*SQRT(!DPI))) * v0Earth * (1 + lambdaEscEarth) * EXP(-lambdaEscEarth)
Mname = STRTRIM('Moon, Exobase T ='+STRING(T_exo_m)+'K',2)
P1 = PLOT(meanMolWeight, ALOG10(phij_per_N_Earth), COLOR = 'blue', TITLE = 'Plot of Jeans Escape Flux per Unit Number Density', $
  XTITLE = 'Mean Molecular Weight', YTITLE = 'Jeans Escape Flux / $n_{exo}$ $(m^{5}s^{-1})$', NAME = 'Earth, Exobase T = 1000K')
P2 = PLOT(meanMolWeight, ALOG10(phij_per_N_Moon), COLOR = 'black', /OVERPLOT, NAME = Mname)
l1 = LEGEND(TARGET=[P1,P2], SHADOW = 0)


END