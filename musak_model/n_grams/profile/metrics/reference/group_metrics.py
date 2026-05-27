from dataclasses import dataclass


@dataclass(frozen=True)
class FigureReferenceGroupMetrics:
    identity_total_variation_distance: float
    common_figure_mass: float
    rare_figure_mass: float
    novel_figure_mass: float
    property_total_variation_distance: float
    contour_total_variation_distance: float
    duration_shape_total_variation_distance: float
