"""The 16-class benchmark vocabulary."""

CLASSES = [
    "count_1",
    "count_2",
    "count_3",
    "count_4",
    "count_5",
    "push_forward",
    "pull_backward",
    "push_left",
    "push_right",
    "raise_hand",
    "lower_hand",
    "thumbs_up",
    "thumbs_down",
    "rotate_counter_clockwise",
    "rotate_clockwise",
    "raise_fist",
]

# The two gesture groups the paper analyses separately. Both involve movement; what differs is where
# the class identity sits -- in which fingers are extended, or in the trajectory of the arm.
COUNTING = list(range(0, 5))
ARM = list(range(5, 16))

REST = -1
N_CLASSES = len(CLASSES)


def group_of(label):
    return "counting" if label in COUNTING else "arm"
