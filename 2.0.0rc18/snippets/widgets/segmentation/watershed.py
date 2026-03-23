import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from panseg.viewer_napari.widgets.segmentation import Segmentation_Tab

tab = Segmentation_Tab()
w = tab.widget_dt_ws

html = render_widget(w, skip_name=True, skip_doc=False)
print(html)
