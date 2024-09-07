from time import sleep

def retry_request(func, retries: int = 3, wait_time : float = 0.5):
    """Retries a function call a specified number of times with a specified wait time between retries.
    TODO: Add support for exponential backoff.

    Args:
        func (_type_): The function to call.
        retries (int, optional): Number of retries to perform. Defaults to 3.
        wait_time (float, optional): Time to wait between retries. Defaults to 0.5.

    Raises:
        RuntimeError: If the function call fails after the specified number of retries.
    """
    print("!")
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"API request failed after {retries} retries: {e}")
            sleep(wait_time)