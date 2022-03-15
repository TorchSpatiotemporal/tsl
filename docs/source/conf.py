import datetime
import sphinx_rtd_theme
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
    'sphinx.ext.githubpages',
]

autosummary_generate = True
templates_path = ['_templates']

source_suffix = '.rst'
master_doc = 'index'

# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Project information -----------------------------------------------------
project = 'tsl'
author = 'Andrea Cini, Ivan Marisca'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

version = tsl.__version__
release = tsl.__version__

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pd': ('https://pandas.pydata.org/docs/', None),
    'PyTorch': ('https://pytorch.org/docs/stable/', None),
    'pytorch_lightning': ('https://pytorch-lightning.readthedocs.io/en/latest/', None),
    'PyG': ('https://pytorch-geometric.readthedocs.io/en/latest/', None)
}

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
    'navigation_depth': 2,
}

html_static_path = ['_static']
html_logo = '_static/img/tsl_logo.svg'
html_favicon = '_static/img/tsl_logo.svg'

html_context = {'css_files': ['_static/css/custom.css']}
rst_context = {'tsl': tsl}

add_module_names = False

def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__repr__',
            '__weakref__',
            '__dict__',
            '__module__',
        ]
        return True if name in members else skip

    def rst_jinja_render(app, docname, source):
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect('autodoc-skip-member', skip)
    app.connect("source-read", rst_jinja_render)
