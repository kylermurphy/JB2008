# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:05:13 2025

@author: murph
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
      ext_modules = cythonize(["jb08/jb2008_subfuncs_cy.pyx"], language_level="3", 
                              annotate=True)
      )