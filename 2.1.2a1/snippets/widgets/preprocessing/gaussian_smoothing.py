import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from panseg.viewer_napari.widgets.preprocessing import Preprocessing_Tab

tab = Preprocessing_Tab()
w = tab.factory_gaussian_smoothing()

html = render_widget(w, skip_name=True, skip_doc=False)
print(html)
