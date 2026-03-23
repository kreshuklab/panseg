import sys

sys.path.append("docs/snippets")

from napari_widgets_render import render_widget

from plantseg.viewer_napari.widgets.output import Output_Tab

tab = Output_Tab()
w = tab.widget_export_image


html = render_widget(w, skip_name=True, skip_doc=False)
print(html)
