"""Operator labels for two visible surfaces; model classes remain task-specific."""
SURFACE_SCHEMA='SIX_COLOR_SURFACES_V1'
COLOR_NAMES={1:'红色',2:'黄色',3:'蓝色',4:'绿色',5:'黑色',6:'浅蓝色'}
LABEL_MAP={name+suffix:{'color_id':color,'surface':surface}
           for color,name in COLOR_NAMES.items()
           for surface,suffix in (('top','顶面'),('side','侧面'))}
SURFACE_LABELS=tuple(LABEL_MAP)


def decode_surface_label(label):
    if label not in LABEL_MAP:raise ValueError('Unknown color/surface annotation label')
    return dict(LABEL_MAP[label])


def label_for(color,surface):
    if type(color) is not int or color not in COLOR_NAMES or surface not in ('top','side'):
        raise ValueError('Explicit known color and surface required')
    return COLOR_NAMES[color]+('顶面' if surface=='top' else '侧面')
