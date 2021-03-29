import re
from IPython.display import display, Markdown, Latex
def _to_latex(text):
    text = str(text)
    text = text.replace('\\', r"\lambda ").replace('exists', r'\exists').replace('all', r'\forall')
    text = text.replace('<->', r'\leftrightarrow').replace('->', r'\rightarrow').replace('&', r'\land').replace('|', r'\lor').replace(' ', r'\ ')
    text = re.sub('z(\d+)', 'z_{\g<1>}', text)
    return text

def display_latex(expr, with_types=False):
    if with_types:
        return display(Markdown(f'${_to_latex(expr)}$ = `{expr.__repr__()}`'))
    else:
        return display(Markdown(f'${_to_latex(expr)}$'))

def display_translation(text, expr, to_latex=True):
    """
    :text: string to be translated to the representation.
    :expr: nltk semantic logical/lambda representation.
    :with_types: if the translation must include the
    """
    if to_latex:
        return display(Markdown(f'"{text}": ${_to_latex(expr)}$'))
    else:
        expr = expr.__repr__()
        return display(Markdown(f'"{text}": `{expr}`'))

def _repr_png_(tree):
    """
    Draws and outputs in PNG for ipython.
    PNG is used instead of PDF, since it can be displayed in the qt console and
    has wider browser support.
    """
    import os
    import base64
    import subprocess
    import tempfile
    from nltk.draw.tree import tree_to_treesegment
    from nltk.draw.util import CanvasFrame
    from nltk.internals import find_binary

    _canvas_frame = CanvasFrame()
    widget = tree_to_treesegment(_canvas_frame.canvas(), tree, node_color='#005990', node_font='arial 13 bold')
    _canvas_frame.add_widget(widget)
    x, y, w, h = widget.bbox()
    # print_to_file uses scrollregion to set the width and height of the pdf.
    _canvas_frame.canvas()["scrollregion"] = (0, 0, w, h)
    with tempfile.NamedTemporaryFile() as file:
        in_path = "{0:}.ps".format(file.name)
        out_path = "{0:}.png".format(file.name)
        _canvas_frame.print_to_file(in_path)
        _canvas_frame.destroy_widget(widget)
        try:
            subprocess.call(
                [
                    find_binary(
                        "gs",
                        binary_names=["gswin32c.exe", "gswin64c.exe"],
                        env_vars=["PATH"],
                        verbose=False,
                    )
                ]
                + "-q -dEPSCrop -sDEVICE=png16m -r90 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -dSAFER -dBATCH -dNOPAUSE -sOutputFile={0:} {1:}".format(
                    out_path, in_path
                ).split()
            )
        except LookupError:
            pre_error_message = str(
                "The Ghostscript executable isn't found.\n"
                "See http://web.mit.edu/ghostscript/www/Install.htm\n"
                "If you're using a Mac, you can try installing\n"
                "https://docs.brew.sh/Installation then `brew install ghostscript`"
            )
            print(pre_error_message, file=sys.stderr)
            raise LookupError

        with open(out_path, "rb") as sr:
            res = sr.read()
        os.remove(in_path)
        os.remove(out_path)
        return base64.b64encode(res).decode()

def display_tree(tree):
    return display(Markdown(f"<img src=\"data:image/png;base64,{_repr_png_(tree)}\">"))
