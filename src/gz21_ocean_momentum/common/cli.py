import os
import sys

from typing import Optional

def path_is_nonexist_or_empty_dir(path) -> bool:
    """Is the given path either nonexistent or an empty directory?"""
    if os.path.exists(path):
        # path exists
        if os.path.isdir(path):
            # path is directory: check contents
            with os.scandir(path) as it:
                if any(it):
                    # path is non-empty directory: fail
                    return False
                else:
                    # path is empty directory: all good
                    return True
        else:
            # path is non-directory: fail
            return False
    else:
        # path does not exist: all good
        return True

def fail(err_code: int, msg: str, hint: Optional[str] = None):
    """Exit the program with the given message and error code.

    Also prints a hint (extra message) afterwards if provided.
    """
    print(f"ERROR: {msg}")
    if hint is not None:
        print(f"hint: {hint}")
    sys.exit(err_code)
