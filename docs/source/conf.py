import datetime
import doctest
import os

from docutils import nodes

import tsl

os.environ["PYTORCH_JIT"] = "0"  # generate doc for torch.jit.script methods

# -- Project information -----------------------------------------------------
#

project = "tsl"
author = "Andrea Cini, Ivan Marisca"
copyright = "{}, {}".format(datetime.datetime.now().year, author)

version = tsl.__version__
release = tsl.__version__

# -- General configuration ---------------------------------------------------
#

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "myst_nb",
    "hoverxref.extension",
]

autosummary_generate = True

source_suffix = ".rst"
master_doc = "index"

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = "bysource"

rst_context = {"tsl": tsl}

add_module_names = False
# autodoc_inherit_docstrings = False

# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

napoleon_custom_sections = [("Shape", "params_style"), ("Shapes", "params_style")]

numfig = True  # Enumerate figures and tables

# -- Options for intersphinx -------------------------------------------------
#

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pd": ("https://pandas.pydata.org/docs/", None),
    "PyTorch": ("https://pytorch.org/docs/stable/", None),
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/latest/", None),
    "PyG": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
}

# -- Theme options -----------------------------------------------------------
#

html_title = "Torch Spatiotemporal"
html_theme = "furo"
language = "en"

html_baseurl = ""
html_static_path = ["_static"]
html_logo = "_static/img/tsl_logo_text.svg"
html_favicon = "_static/img/tsl_logo.svg"

html_css_files = [
    "css/custom.css",
]

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#D34317",
        "color-brand-content": "#D34317",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FF5722",
        "color-brand-content": "#FF5722",
    },
}

pygments_style = "tango"
pygments_dark_style = "material"

# -- Notebooks options -------------------------------------------------------
#

nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath"]
myst_dmath_allow_space = True
myst_dmath_double_inline = True
nb_code_prompt_hide = "Hide code cell outputs"

# -- OpenGraph options -------------------------------------------------------
#

ogp_site_url = "https://torch-spatiotemporal.readthedocs.io/en/latest/"
ogp_image = ogp_site_url + "_static/tsl_logo.svg"

# -- Hoverxref options -------------------------------------------------------
#

hoverxref_auto_ref = True
hoverxref_roles = ["class", "mod", "doc", "meth", "func"]
hoverxref_mathjax = True
hoverxref_intersphinx = ["PyG", "numpy"]


# -- Setup options -----------------------------------------------------------
#


def logo_role(name, rawtext, text, *args, **kwargs):
    if name == "tsl":
        url = f"{html_baseurl}/_static/img/tsl_logo.svg"
    elif name == "hydra":
        url = f"{html_baseurl}/_static/img/logos/hydra-head.svg"
    elif name in ["pyg", "pytorch", "lightning"]:
        url = f"{html_baseurl}/_static/img/logos/{name}.svg"
    else:
        raise RuntimeError
    node = nodes.image(uri=url, alt=str(name).capitalize() + " logo")
    node["classes"] += ["inline-logo", name]
    if text != "null":
        node["classes"].append("with-text")
        span = nodes.inline(text=text)
        return [node, span], []
    return [node], []


def setup(app):
    def rst_jinja_render(app, docname, source):
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect("source-read", rst_jinja_render)

    app.add_role("tsl", logo_role)
    app.add_role("pyg", logo_role)
    app.add_role("pytorch", logo_role)
    app.add_role("hydra", logo_role)
    app.add_role("lightning", logo_role)
