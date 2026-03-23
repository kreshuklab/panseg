import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from plantseg.viewer_napari.widgets.preprocessing import Preprocessing_Tab, RescaleModes

tab = Preprocessing_Tab()
widget_rescaling = tab.widget_rescaling
widget_rescaling.mode.value = RescaleModes.FROM_FACTOR

html = render_widget(widget_rescaling, skip_name=True)
print(html)
