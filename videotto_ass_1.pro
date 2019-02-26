;+
; :Author: Alex
;-
PRO videotto_ass_1

;Boltzmann constant mks
kb = 1.38064852D-23
;gravitational constant mks
bigG = 6.67408D-11
;Radius of the earth
eRad = 6.378D8 / 100
;Radius of the earth's exobase
R_exo_e = eRad + (5D7/100)
;radius of the moon
mRad = 1.737D8 / 100
;radius of the moon's exobase
R_exo_m = mRad

;mass of the earth and moon in mks
eMass = 5.974D27 / 1000
mMass = 7.35D25 / 1000

;Solar effective temperature
tsun = 5778D
;solar radius in mks
R_sun = 69.5508D9 / 100
;distance to the moon from the sun mks
d_moon = 1.496D13 / 100
;lunar bond albedo
moonAlb = 0.12

;given temperature of earth exobase
T_exo_e = 1000.
;Estimate equilibrium temperature of the moon
T_exo_m = tsun * SQRT(R_sun/d_moon) * ((1 - moonAlb)/4.)^(1./4)


meanMolWeight = FINDGEN(44,START = 1)
meanMolMass = meanMolWeight / (1000*6.022140857D23)

;Determine jeans escape parameter for the moon
lambdaEscMoon = (bigG * mMass * meanMolMass) / (R_exo_m * T_exo_m * kb)
;determine v0 for the moon
v0Moon = SQRT((2*kb*T_exo_m)/(meanMolMass))

;determine jeans escape parameter for the earth
lambdaEscEarth = (bigG * eMass * meanMolMass) / (R_exo_e * T_exo_e * kb)
;vo for the earth
v0Earth = SQRT((2*kb*T_exo_e)/(meanMolMass))

;compute phij per molecular mass for the earth and moon
phij_per_N_Moon = (1./(2*SQRT(!DPI))) * v0Moon * (1 + lambdaEscMoon) * EXP(-lambdaEscMoon)
phij_per_N_Earth = (1./(2*SQRT(!DPI))) * v0Earth * (1 + lambdaEscEarth) * EXP(-lambdaEscEarth)
Mname = STRTRIM('Moon, Exobase T ='+STRING(T_exo_m)+'K',2)

;Plot data on a semi-log plot
P1 = PLOT(meanMolWeight, ALOG10(phij_per_N_Earth), COLOR = 'blue', $
  TITLE = 'Plot of Jeans Escape Flux per Unit Number Density', $
  XTITLE = 'Mean Molecular Weight', $
  YTITLE = 'Jeans Escape Flux / $n_{exo}$ $(ms^{-1})$', $
  NAME = 'Earth, Exobase T = 1000K', LINE = '--')
  
P2 = PLOT(meanMolWeight, ALOG10(phij_per_N_Moon),$
  COLOR = 'black', /OVERPLOT, NAME = Mname)

l1 = LEGEND(TARGET=[P1,P2], SHADOW = 0)
;Observe neglibeble escape rates for all but the lightest molecules for the earth
; for the lunar case, even massive molecules like CO2 would have non-negligeble escape rates
;This explains the Earth's secondary atmosphere, and why the moon is all but completely lacking in atmosphere 
END