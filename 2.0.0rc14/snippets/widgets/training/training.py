import sys

sys.path.append("docs/snippets")

from napari_widgets_render import render_widget

from plantseg.viewer_napari.widgets.training import Training_Tab

tab = Training_Tab(None)
w = tab.widget_unet_training


html = render_widget(w, skip_name=True, skip_doc=True)
print(html)
