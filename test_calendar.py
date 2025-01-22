#!/usr/bin/env python3

import subprocess
from datetime import datetime

# Map Python's 1-12 month numbers to AppleScript's month constants.
# AppleScript uses English month names as constants, even on non-English macOS.
MONTH_CONSTANTS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def create_calendar_event(calendar_name, title, start_time, end_time, location="", notes=""):
    """
    Create a new event in the specified macOS Calendar using AppleScript.
    This version does not rely on AppleScript date-parsing, so it works well
    even on Spanish or other non-English macOS systems.
    """

    # Convert Python's datetime objects into numeric values for AppleScript
    # (year, month name, day, total seconds from midnight).
    start_year = start_time.year
    start_month = MONTH_CONSTANTS[start_time.month - 1]  # e.g., "January"
    start_day = start_time.day
    start_total_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second

    end_year = end_time.year
    end_month = MONTH_CONSTANTS[end_time.month - 1]
    end_day = end_time.day
    end_total_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second

    # Create an AppleScript that:
    # 1. Gets the current date into two variables (dStart, dEnd).
    # 2. Sets the year, month, day, and time of each variable.
    # 3. Creates a new event with dStart and dEnd as the start and end date.
    #
    # Notice we do not pass "date \"...\"" strings for AppleScript to parse.
    # Instead, we do `set year of dStart to 2025`, etc.
    apple_script = f'''
        tell application "Calendar"
            activate

            -- Build start date
            set dStart to current date
            set year of dStart to {start_year}
            set month of dStart to {start_month}
            set day of dStart to {start_day}
            set time of dStart to {start_total_seconds}

            -- Build end date
            set dEnd to current date
            set year of dEnd to {end_year}
            set month of dEnd to {end_month}
            set day of dEnd to {end_day}
            set time of dEnd to {end_total_seconds}

            -- Create the new event
            make new event at end of events of calendar "{calendar_name}" with properties {{
                summary: "{title}",
                start date: dStart,
                end date: dEnd,
                location: "{location}",
                description: "{notes}"
            }}
        end tell
    '''

    try:
        subprocess.run(["osascript", "-e", apple_script], check=True)
        print(f"Successfully created event '{title}' in calendar '{calendar_name}'.")
    except subprocess.CalledProcessError as err:
        print("An error occurred while creating the event:")
        print(err)

if __name__ == "__main__":
    # Example usage:
    calendar_name = "Home"  # Make sure this matches an existing calendar’s name
    title = "Evento creado por Python"

    # Create an event on January 22, 2025 from 10:00 PM to 11:00 PM
    # (22:00 to 23:00 in 24-hour time)
    start_time = datetime(2025, 1, 22, 22, 0)
    end_time   = datetime(2025, 1, 22, 23, 0)

    location = "Oficina"
    notes = "Llevar los documentos importantes"

    create_calendar_event(calendar_name, title, start_time, end_time, location, notes)
