# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
from setuptools import setup

setup(
    name='hensmith',
    packages=['hensmith'],
    license='NCSA',
    version='0.1.2',
    description=('Heat Exchanger Network Synthesis, Modeling, Integration, '
                 'Thermodynamics, and Heuristics'),
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Sarang Bhagwat',
    author_email='sarangbhagwat.developer@gmail.com',
    maintainer='Sarang Bhagwat',
    maintainer_email='sarangbhagwat.developer@gmail.com',
    # biosteam 2.54.0 is the first release without the bundled
    # biosteam.facilities.hxn copy of this package; older releases would
    # coexist with hensmith as two distinct HeatExchangerNetwork classes
    # (the bundled copy winning bst.HeatExchangerNetwork), silently breaking
    # isinstance checks downstream.
    install_requires=['biosteam>=2.54.0'],
    python_requires='>=3.12',
    platforms=['Windows', 'Mac', 'Linux'],
    url='https://github.com/BioSTEAMDevelopmentGroup/hensmith',
    download_url='https://github.com/BioSTEAMDevelopmentGroup/hensmith.git',
    classifiers=['License :: OSI Approved :: University of Illinois/NCSA Open Source License',
                 'Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Manufacturing',
                 'Intended Audience :: Science/Research',
                 'Natural Language :: English',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: POSIX :: BSD',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: Unix',
                 'Programming Language :: Python :: 3.12',
                 'Programming Language :: Python :: 3.13',
                 'Programming Language :: Python :: Implementation :: CPython'],
    keywords=['heat exchanger network', 'pinch analysis', 'heat integration',
              'process design', 'chemical process simulation', 'biorefinery',
              'techno-economic analysis', 'BioSTEAM'],
)
