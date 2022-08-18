import datetime
import doctest

import tsl

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxext.opengraph',
    'sphinx.ext.githubpages',
]

autosummary_generate = True

source_suffix = '.rst'
master_doc = 'index'

# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Project information -----------------------------------------------------
project = 'tsl'
author = 'Andrea Cini, Ivan Marisca'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

version = tsl.__version__
release = tsl.__version__

html_theme = 'furo'

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pd': ('https://pandas.pydata.org/docs/', None),
    'PyTorch': ('https://pytorch.org/docs/stable/', None),
    'pytorch_lightning': (
    'https://pytorch-lightning.readthedocs.io/en/latest/', None),
    'PyG': ('https://pytorch-geometric.readthedocs.io/en/latest/', None)
}

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#ee4c2c",
        "color-brand-content": "#ee4c2c",
    },
    "dark_css_variables": {
        # "color-code-background": "#131416",
        # "color-background-primary": "#131416",
        "color-brand-primary": "#ee4c2c",
        "color-brand-content": "#ee4c2c",
    }
}
html_title = "Torch Spatiotemporal"
pygments_style = "tango"
pygments_dark_style = "material"

html_static_path = ['_static']
html_logo = '_static/img/tsl_logo.svg'
html_favicon = '_static/img/tsl_logo.svg'

html_css_files = [
    'css/custom.css',
]

rst_context = {'tsl': tsl}

add_module_names = False
# autodoc_inherit_docstrings = False

# OpenGraph metadata
ogp_site_url = "https://torch-spatiotemporal.readthedocs.io/en/latest/"
ogp_image = ogp_site_url + "_static/tsl_logo.svg"


def setup(app):
    def rst_jinja_render(app, docname, source):
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect("source-read", rst_jinja_render)
