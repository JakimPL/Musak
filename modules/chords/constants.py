LOWEST_NOTE = 40
HIGHEST_NOTE = 90
KEYS = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B'
}

CHORDS = {
    '': (0, 4, 7),
    'm': (0, 3, 7),
    'dim': (0, 3, 6),
    'aug': (0, 4, 8),
    'sus2': (0, 2, 7),
    'sus4': (0, 5, 7),
    '7': (0, 4, 7, 10),
    'maj7': (0, 4, 7, 11),
    'm7': (0, 3, 7, 10),
    'm(maj7)': (0, 3, 7, 11)
}

INTERVAL_NAMES = {
    0: 'prime',
    1: 'minor_second',
    2: 'major_second',
    3: 'minor_third',
    4: 'major_third',
    5: 'perfect_fourth',
    6: 'tritone',
    7: 'perfect_fifth',
    8: 'minor_sixth',
    9: 'major_sixth',
    10: 'minor_seventh',
    11: 'major_seventh',
    12: 'octave',
    13: 'minor_ninth',
    14: 'major_ninth',
    15: 'minor_tenth',
    16: 'major_tenth',
    17: 'perfect_eleventh',
    18: 'octave_and_tritone',
    19: 'perfect_twelfth',
    20: 'minor_thirteenth',
    21: 'major_thirteenth',
    22: 'minor_fourteenth',
    23: 'major_fourteenth',
    24: 'double_octave'
}
