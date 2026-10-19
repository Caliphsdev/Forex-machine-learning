"""Walk-forward train/test splitting by time. Never random."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """Row positions. Ends are exclusive, like Python slices."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def test(self) -> slice:
        return slice(self.test_start, self.test_end)


def walk_forward_splits(n_samples: int, train_size: int, test_size: int,
                        gap: int, expanding: bool = True) -> list[Fold]:
    """Chronological folds. Test windows tile the timeline without overlap.

    `gap` rows separate train from test; set it to the label horizon so no
    training label is computed from prices inside the test window.
    `expanding=True` grows the training set; False slides a fixed-size window.
    """
    if train_size <= 0 or test_size <= 0 or gap < 0:
        raise ValueError("train_size and test_size must be > 0 and gap >= 0")

    folds: list[Fold] = []
    train_end = train_size
    while train_end + gap + test_size <= n_samples:
        train_start = 0 if expanding else train_end - train_size
        test_start = train_end + gap
        folds.append(Fold(train_start, train_end, test_start, test_start + test_size))
        train_end += test_size

    if not folds:
        raise ValueError(
            f"Not enough samples ({n_samples}) for train={train_size}, "
            f"gap={gap}, test={test_size}"
        )
    return folds

def split_validation(start: int, end: int, val_size: int,
                     gap: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Carve the last `val_size` rows of [start, end) off as validation.

    `gap` rows between train and validation are left unused so no training label
    is computed from prices inside the validation window.
    Returns ((train_start, train_end), (val_start, val_end)).
    """
    val_start = end - val_size
    train_end = val_start - gap
    if val_size <= 0 or gap < 0 or train_end <= start:
        raise ValueError(
            f"Cannot fit val_size={val_size} and gap={gap} inside rows [{start}, {end})"
        )
    return (start, train_end), (val_start, end)
