import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from plantseg.viewer_napari.widgets.segmentation import Segmentation_Tab

tab = Segmentation_Tab()
w = tab.widget_agglomeration

html = render_widget(w, skip_name=True, skip_doc=True)
print(html)
