# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
#sys.path.insert(0, os.path.abspath(r'C:\Users\daria\OneDrive\Desktop\CIAO\CMEPDA-EXAM'))
sys.path.insert(0, os.path.abspath('..'))

project = 'CMEPDA-EXAM'
copyright = '2025, Spinelli Brando, Ungolo Daria'
author = 'Brando Spinelli, Daria Ungolo'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # Per estrarre automaticamente docstring
    'sphinx.ext.napoleon',   # Per supportare docstring in stile Google/NumPy
    'sphinx.ext.viewcode'    # Per collegamenti al codice sorgente
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "navigation_depth": 3,       # Profondità massima della navigazione laterale
    "show_nav_level": 1,         # Mostra il primo livello nella navigazione
    "collapse_navigation": False,  # Mantieni espansa la navigazione laterale
    "sticky_navigation": True,   # Navigazione sempre visibile mentre scorri
}

html_static_path = ['_static']

