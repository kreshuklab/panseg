import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from panseg.viewer_napari.widgets.preprocessing import Preprocessing_Tab, RescaleModes

tab = Preprocessing_Tab()
widget_rescaling = tab.widget_rescaling
widget_rescaling.mode.value = RescaleModes.TO_MODEL_VOXEL_SIZE

html = render_widget(widget_rescaling, skip_name=True)
print(html)
