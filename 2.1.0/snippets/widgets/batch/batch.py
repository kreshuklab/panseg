import sys

sys.path.append("docs/snippets")

from napari_widgets_render import render_widget

from panseg.viewer_napari.widgets.output import Batch_Tab

tab = Batch_Tab()
w = tab.widget_export_workflow


html = render_widget(w, skip_name=True, skip_doc=False)
print(html)
