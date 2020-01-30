---
title: 'Maelstrom: A Python package for identifying companions to pulsating stars from their light travel time variations'
tags:
  - Python
  - astronomy
  - variable stars
authors:
  - name: Daniel R. Hey
    orcid: 0000-0003-3244-5357
    affiliation: "1, 2"
  - name: Simon J. Murphy
    orcid: 0000-0002-5648-3107
    affiliation: "1, 2"
  - name: Daniel Foreman-Mackey
    orcid: 0000-0002-9328-5652
    affiliation: "3"
affiliations:
 - name: School of Physics, Sydney Institute for Astronomy (SIfA), The University of Sydney, NSW 2006, Australia
   index: 1
 - name: Stellar Astrophysics Centre, Department of Physics and Astronomy, Aarhus University, DK-8000 Aarhus C, Denmark
   index: 2
 - name: Center for Computational Astrophysics, Flatiron Institute, 162 5th Ave, New York, NY 10010, USA
   index: 3
date: 28 January 2020
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
Most stars are members of binary systems. Observations of these systems provide
optimal testing grounds for stellar models. Traditional observations of these
systems rely on eclipses, where the secondary body occludes light from the
primary star and vice versa. This leads to a strong bias on the orbital
parameters -- to observe these eclipses the system must have a high inclination
with respect to Earth, and the orbital period must be sufficiently short to be
observed within the time-span of the data. Other methods for detecting binarity
suffer from similar constraints: radial velocity measurements are useful mostly
for orbital periods less than tens of days, whilst long-baseline interferometry
is suited only to relatively nearby and bright stars.

However, in some binary systems, one or both of the components are pulsating
variable stars. Some variable stars make excellent clocks, possessing stable
pulsations that do not vary significantly throughout their orbit. One such type
are the $\delta$ Scuti variables, a class of A/F type stars lying along the
classical instability strip. As the pulsating star is tugged around by the
gravity of its companion the time taken for its light to reach Earth varies,
which periodically modulates the pulsation phases. A map of the binary orbit
can then be constructed by observing the phase of the pulsations over time,
which can be converted into time delays -- a measure of the relative time taken
for the light to reach Earth (as shown in Fig. 1). This method, phase
modulation, is uniquely suited to observing intermediate period binaries
[@Murphy2015Deriving].

![Phase modulation of the binary system, KIC 4471379, which is composed of two $\delta$ Scuti pulsating stars. The top panel shows the observed flux of the system (the light curve). The middle panel shows the amplitude spectrum of the light curve, which is the superposition of each star's pulsation spectrum. The bottom panel shows the time delay derived from the pulsations. Blue and orange points correspond to the first and second stars in the system respectively. As they orbit each other, the time taken for the light to reach us changes over time. Since both stars are pulsating, we can identify which pulsations belong to which star.](PB2_KIC_4471379_JOSS.png)

Previous work has analysed these variations by splitting the light curve into
equally sized subdivisions and calculating the time delay in each division
[@Murphy2018Finding]. Whilst useful for longer period binaries (>20 d), shorter
period and eccentric binaries suffer from a smearing of the orbital signal due
to large variations when the stars are at their closest approach
[@Murphy2016Finding]. Since the phase uncertainty is inversely proportional to
the size of the subdivision, shorter period binaries can not be accurately
determined. We have developed a novel technique for mitigating this problem by
forward modelling the time-delay effect directly onto the light curve, which
allows for every data point in the light curve to be modelled simultaneously,
removing the need to choose a subdivision size. This technique is explained in
detail in our corresponding paper.

We have developed a Python package, ``Maelstrom``, which implements this
technique. ``Maelstrom`` is written using the popular Bayesian inference
framework, ``PyMC3``, allowing for the use of gradient based samplers such as
No-U-Turn [@Hoffman2011NoUTurn] and Hamiltonian Monte Carlo [@Duane1987Hybrid].
``Maelstrom`` features a series of pre-defined ``PyMC3`` models for analysing
binary motion within stellar pulsations. These are powered by the ``orbit``
module, which returns a light curve given the frequencies of pulsation and the
classical orbital parameters. Using this light curve, one can compare with
photometric data from the *Kepler* and *TESS* space missions to fit for binary
motion. For more complex systems outside the pre-defined scope, the ``orbit``
module can be used to construct custom models with different priors, and
combine them with other ``PyMC3`` codes, such as exoplanet
[@DanForeman-Mackey2019Dfm]. To the best of our knowledge, ``Maelstrom`` is
currently the only available open code for analysing time delay signals.

The documentation of `maelstrom` consists of pages describing the various
available functions, as well as tutorial notebooks.

# References