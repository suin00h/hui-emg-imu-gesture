"""The 16-class benchmark vocabulary."""

CLASSES = [
    "count_1", "count_2", "count_3", "count_4", "count_5",
    "go_forward", "go_backward", "go_left", "go_right",
    "raise_height", "lower_height", "speed_up", "speed_down",
    "turn_counter_clockwise", "turn_clockwise", "stop_movement",
]

# The two gesture groups the paper analyses separately: held finger postures, read from which
# electrode is loaded, and gross arm movements, read from temporal shape.
COUNTING = list(range(0, 5))
ARM = list(range(5, 16))

REST = -1
N_CLASSES = len(CLASSES)


def group_of(label):
    return "counting" if label in COUNTING else "arm"
