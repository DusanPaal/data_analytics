import time

class Timer:
  """A simple timer class to measure elapsed time."""

  def __init__(self):
    """Initializes the timer object with a start time set to None."""

    self.start_time = None

  def start(self):
    """Starts the timer by recording the current time."""

    self.start_time = time.time()

  def stop(self, print_time: bool = True) -> float:
    """Stops the timer and optionally prints the elapsed time.

    Parameters:
    -----------
    print_time (bool): 
    If True, prints the elapsed time (default).

    Returns:
    --------
    float: The elapsed time in seconds.
    """

    elapsed_time = time.time() - self.start_time

    if print_time:
      print("Elapsed time:", elapsed_time)

    return elapsed_time
  
  def reset(self):
    """Resets the timer."""
    self.start_time = None