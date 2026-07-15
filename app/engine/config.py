"""Headless construction of the `control` and `default_parameters` dicts.

The desktop UI builds these dicts as a side effect of creating widgets
(`app/ui/widgets/actions/layout_actions.py:add_widgets_to_tab_layout`). The
engine only ever consumes the plain dicts, so this module reproduces the same
value coercion straight from the layout-data dicts with zero Qt / zero widgets.

Keep the type coercion here in sync with `add_widgets_to_tab_layout`:
widget-name suffix decides the type, checked in this order:
    'Toggle'        -> bool   (default as-is)
    'Selection'     -> value  (call default if callable)
    'DecimalSlider' -> float  (checked before 'Slider')
    'Slider'        -> int
    'Text'          -> value as-is
"""
from typing import Dict, Any

from app.helpers.miscellaneous import ParametersDict
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA

# data_type mapping mirrors main_ui.py:186-189
#   parameter: COMMON, SWAPPER, FACE_EDITOR
#   control:   SETTINGS
PARAMETER_LAYOUTS = (COMMON_LAYOUT_DATA, SWAPPER_LAYOUT_DATA, FACE_EDITOR_LAYOUT_DATA)
CONTROL_LAYOUTS = (SETTINGS_LAYOUT_DATA,)

# Recognition model options double as the arcface model names used to key every
# face embedding_store (see card_actions.find_target_faces + arcface_mapping).
RECOGNITION_OPTIONS = list(
    SETTINGS_LAYOUT_DATA['Face Recognition']['RecognitionModelSelection']['options']
)


def _coerce_default(widget_name: str, widget_data: dict) -> Any:
    """Return the default value with the same coercion the UI applies."""
    default = widget_data['default']
    if 'Toggle' in widget_name:
        return default
    if 'Selection' in widget_name:
        return default() if callable(default) else default
    if 'DecimalSlider' in widget_name:  # must precede 'Slider'
        return float(default)
    if 'Slider' in widget_name:
        return int(default)
    # 'Text' and anything else: as-is
    return default


def _flatten(layouts) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for layout in layouts:
        for _category, widgets in layout.items():
            for widget_name, widget_data in widgets.items():
                out[widget_name] = _coerce_default(widget_name, widget_data)
    return out


def build_default_parameters() -> Dict[str, Any]:
    """Plain dict of default face parameters (COMMON + SWAPPER + FACE_EDITOR)."""
    return _flatten(PARAMETER_LAYOUTS)


def build_default_control() -> Dict[str, Any]:
    """Plain dict of default control settings (SETTINGS)."""
    return _flatten(CONTROL_LAYOUTS)


def new_parameters(overrides: Dict[str, Any] | None = None) -> ParametersDict:
    """A per-face ParametersDict seeded from defaults, with optional overrides.

    Falls back to defaults for any key not supplied (same semantics the desktop
    app relies on when loading older workspaces)."""
    defaults = build_default_parameters()
    params = ParametersDict(dict(defaults), defaults)
    if overrides:
        for key, value in overrides.items():
            params[key] = value
    return params


def new_control(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """A control dict seeded from defaults, with optional overrides."""
    control = build_default_control()
    if overrides:
        control.update(overrides)
    return control
