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

def fail_if_path_is_nonempty_dir(err_code: int, msg_pre: str, path):
    msg = f"{msg_pre}: path should not exist, or be empty directory"
    if os.path.exists(path):
        # path exists
        if os.path.isdir(path):
            # path is directory: check contents
            with os.scandir(path) as it:
                if any(it):
                    # path is non-empty directory: fail
                    fail(err_code, msg, "path is non-empty directory")
                # else path is empty directory: all good, do nothing
        else:
            # path is non-directory: fail
            fail(err_code, msg, "path is not a directory")
    # else path does not exist: all good, do nothing

def fail(err_code: int, msg: str, hint: Optional[str] = None):
    """Exit the program with the given message and error code.

    Also prints a hint (extra message) afterwards if provided.
    """
    print(f"ERROR: {msg}")
    if hint is not None:
        print(f"hint: {hint}")
    sys.exit(err_code)
