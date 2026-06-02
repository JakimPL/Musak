from musak_model.n_grams.figure.schema import FigureDegree, FigureNGram, FigureOnset

_INDEL_COST: float = 1.0


def figure_edit_distance(left: FigureNGram, right: FigureNGram) -> float:
    source = left.onsets
    target = right.onsets
    previous = [float(column) for column in range(len(target) + 1)]
    for row in range(1, len(source) + 1):
        current = [float(row)] + [0.0] * len(target)
        for column in range(1, len(target) + 1):
            substitution = previous[column - 1] + _onset_distance(source[row - 1], target[column - 1])
            deletion = previous[column] + _INDEL_COST
            insertion = current[column - 1] + _INDEL_COST
            current[column] = min(substitution, deletion, insertion)

        previous = current

    return previous[len(target)]


def _onset_distance(left: FigureOnset, right: FigureOnset) -> float:
    left_degrees, left_duration = left
    right_degrees, right_duration = right
    duration_cost = 0.0 if left_duration == right_duration else 1.0
    return _degree_distance(left_degrees, right_degrees) + duration_cost


def _degree_distance(left: tuple[FigureDegree, ...], right: tuple[FigureDegree, ...]) -> float:
    if len(left) == 1 and len(right) == 1:
        left_position, left_accidental = left[0]
        right_position, right_accidental = right[0]
        return float(abs(left_position - right_position) + abs(left_accidental - right_accidental))

    return float(len(set(left) ^ set(right)))
