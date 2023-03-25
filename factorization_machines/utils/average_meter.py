class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        """Constructor method.

        Args:
            name (str): Name of the meter
            fmt (str, optional): Format of the meter. Defaults to ":f".
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset the meter."""
        self.value = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: float, n: int = 1) -> None:
        """Update the meter.

        Args:
            value (float): Value to update the meter
            n (int, optional): Number of samples. Defaults to 1.
        """
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {value" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
