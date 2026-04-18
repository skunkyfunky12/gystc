"""Brain region definitions and community-to-region mapping.

Maps 77 graphify communities to 12 anatomical brain regions,
each with a name, RGB color, and 3D position.
"""

REGIONS = [
    {"name": "Praefrontaler Cortex",  "color": (0.204, 0.596, 0.859), "position": (0.0, 420.0, -540.0)},
    {"name": "Motorischer Cortex",    "color": (0.906, 0.298, 0.235), "position": (-210.0, 470.0, -260.0)},
    {"name": "Sensorischer Cortex",   "color": (0.180, 0.800, 0.443), "position": (210.0, 470.0, -130.0)},
    {"name": "Hippocampus",           "color": (0.953, 0.612, 0.071), "position": (-340.0, 0.0, 130.0)},
    {"name": "Kleinhirn",             "color": (0.608, 0.349, 0.714), "position": (280.0, -150.0, 470.0)},
    {"name": "Nucleus Accumbens",     "color": (0.102, 0.737, 0.612), "position": (75.0, 225.0, -300.0)},
    {"name": "Broca-Areal",           "color": (0.902, 0.494, 0.133), "position": (-260.0, 210.0, -340.0)},
    {"name": "Visueller Cortex",      "color": (0.557, 0.267, 0.678), "position": (210.0, 130.0, 470.0)},
    {"name": "Thalamus",              "color": (0.086, 0.627, 0.522), "position": (0.0, 260.0, 75.0)},
    {"name": "Stammhirn",             "color": (0.584, 0.647, 0.651), "position": (0.0, -420.0, 340.0)},
    {"name": "Basalganglien",         "color": (0.827, 0.329, 0.000), "position": (-225.0, 150.0, -120.0)},
    {"name": "Amygdala",              "color": (0.753, 0.224, 0.169), "position": (-150.0, -95.0, -210.0)},
]

COMMUNITY_TO_REGION = {
    0: 0,   1: 1,   2: 2,   3: 3,   4: 4,   5: 5,   6: 6,   7: 7,
    8: 7,   9: 7,  10: 10, 11: 10, 12: 8,  13: 8,  14: 10, 15: 11,
    16: 5,  17: 10, 18: 10, 19: 11, 20: 7,  21: 11, 22: 9,  23: 8,
    24: 10, 25: 10, 26: 10, 27: 11, 28: 7,  29: 3,  30: 3,  31: 3,
    32: 9,  33: 0,  34: 0,  35: 11, 36: 11, 37: 9,  38: 7,  39: 7,
    40: 9,  41: 7,  42: 7,  43: 9,  44: 3,  45: 3,  46: 3,  47: 7,
    48: 10, 49: 5,  50: 7,  51: 9,  52: 9,  53: 9,  54: 9,  55: 9,
    56: 9,  57: 9,  58: 9,  59: 9,  60: 9,  61: 0,  62: 9,  63: 8,
    64: 9,  65: 9,  66: 9,  67: 0,  68: 8,  69: 7,  70: 3,  71: 7,
    72: 7,  73: 3,  74: 0,  75: 10, 76: 8,
}
