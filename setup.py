# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
# Copyright (C) 2020-, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
from setuptools import setup

setup(
    name='hensmith',
    packages=['hensmith'],
    license='MIT',
    version='0.1.0',
    description=('Heat Exchanger Network Synthesis, Modeling, Integration, '
                 'Thermodynamics, and Heuristics'),
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Sarang Bhagwat',
    author_email='sarangbhagwat.developer@gmail.com',
    maintainer='Sarang Bhagwat',
    maintainer_email='sarangbhagwat.developer@gmail.com',
    install_requires=['biosteam>=2.53.11'],
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
