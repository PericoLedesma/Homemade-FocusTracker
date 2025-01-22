
import os
import numpy as np
import time
import subprocess  # Needed to call 'osascript' on macOS


# ------------------------------------------------------------------
#  FILE CHECK: verify the existence of YOLO-related files
# ------------------------------------------------------------------
def verify_files_exist(file_paths):
    """
    Checks whether all files in 'file_paths' exist.
    Raises FileNotFoundError if any are missing.
    """
    missing_files = [f for f in file_paths if not os.path.isfile(f)]
    if missing_files:
        missing = ', '.join(missing_files)
        raise FileNotFoundError(
            f"Could not find these YOLO files: {missing}.\n"
            f"Please verify the paths and ensure the files are available."
        )

# ------------------------------------------------------------------
# HELPER FUNCTIONS FOR FORMATTING AND DISTANCE CALCULATION
# ------------------------------------------------------------------
def format_time(seconds: float) -> str:
    """
    Given a time in seconds, returns a string in the HH:MM:SS format.
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def distance(pt1, pt2):
    """
    Computes Euclidean distance between two points (x1, y1) and (x2, y2).
    """
    x1, y1 = pt1
    x2, y2 = pt2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ------------------------------------------------------------------
#  CREATE A MAC CALENDAR EVENT (FOCUS SESSION)
# ------------------------------------------------------------------


def create_calendar_event(title: str, start_ts: float, end_ts: float, current_focus: float):
    """
    Creates an event in the Mac Calendar app using AppleScript (`osascript`).
    The event is placed in a calendar named "Work" (adjust as needed).
    'start_ts' and 'end_ts' are Unix timestamps.
    'current_focus' is the total focus time in seconds.
    """
    # Format times so AppleScript can parse them as date "MM/DD/YYYY HH:MM:SS"
    start_str = time.strftime("%m/%d/%Y %H:%M:%S", time.localtime(start_ts))
    end_str = time.strftime("%m/%d/%Y %H:%M:%S", time.localtime(end_ts))
    focus_str = format_time(current_focus)  # "HH:MM:SS"

    # Build AppleScript code as a *single string*, with no trailing comma in the properties
    ascript = f'''
    tell application "Calendar"
        tell calendar "Work"
            make new event at end with properties {{
                summary: "{title} / Effective: {focus_str}",
                start date: date "{start_str}",
                end date: date "{end_str}"
            }}
        end tell
    end tell
    '''

    try:
        # Strip leading/trailing newlines/spaces to avoid AppleScript parser issues
        subprocess.run(["osascript", "-e", ascript.strip()], check=True)
        print(f"Calendar event '{title}' created from {start_str} to {end_str}.")
    except subprocess.CalledProcessError as e:
        print("Error creating the Calendar event:", e)
