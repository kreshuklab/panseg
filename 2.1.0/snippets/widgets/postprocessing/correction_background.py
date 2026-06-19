import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from panseg.viewer_napari.widgets.postprocessing import Postprocessing_Tab

tab = Postprocessing_Tab()
w = tab.widget_remove_false_positives_by_foreground

html = render_widget(w, skip_name=True, skip_doc=False)
print(html)
