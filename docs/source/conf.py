# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project= 'CMEPDA-EXAM'

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

sys.path.insert(0, package_root)
sys.path.insert(0, os.path.join(package_root, 'ML_codes'))
sys.path.insert(0, os.path.join(package_root, 'main'))
sys.path.insert(0, os.path.join(package_root, 'CNN_codes'))

autodoc_mock_imports = [
    "tensorflow",
    "keras",
    "graphviz",
    "scipy",
    "nibabel",
    "loguru",
    "sklearn",
    "matlabengine",
    "matlab",
    "matlab.engine",
]



copyright = '2025, Brando Spinelli, Daria Ungolo'
author = 'Brando Spinelli , Daria Ungolo'
#release = '0.0.1'

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

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "navigation_depth": 3,       # Profondità massima della navigazione laterale
    "show_nav_level": 1,         # Mostra il primo livello nella navigazione
    "collapse_navigation": False,  # Mantieni espansa la navigazione laterale
}
html_static_path = ['_static']


