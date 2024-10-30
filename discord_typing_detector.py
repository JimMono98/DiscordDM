import cv2
import numpy as np
import mss
import time
from datetime import datetime  # For timestamping
import os

# Load the image template for the typing indicator
typing_template = cv2.imread('Online.png')

# Check if the template is loaded correctly
if typing_template is None:
    print("Error: Template image not found. Check the file path.")
    exit(1)

# Function to resize the template
def resize_template(template, scale_percent):
    width = int(template.shape[1] * scale_percent / 100)
    height = int(template.shape[0] * scale_percent / 100)
    return cv2.resize(template, (width, height))

def save_screenshot(screen):
    directory = "Good Stuff"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(directory, f"screenshot_{timestamp}.png")
    cv2.imwrite(filename, screen)

# Use MSS to list all connected monitors
with mss.mss() as sct:
    # Define the region of the second monitor (adjust according to your setup)
    monitor_2 = sct.monitors[1]  # Adjust the index if necessary

    # Resize the template to 30% of its original size for detection
    typing_template_small = resize_template(typing_template, 31)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    

    while True:
        # Capture the entire second monitor
        screen = np.array(sct.grab(monitor_2))

        # Convert captured screen to BGR format (from BGRA)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

        # Check for the typing indicator in the captured screen
        res_typing = cv2.matchTemplate(screen, typing_template_small, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc_typing = np.where(res_typing >= threshold)

        # If a match is found, it means the typing indicator appeared
        if len(loc_typing[0]) > 0:
            print(f"Typing indicator detected at {current_time}! Someone might be typing.")
            save_screenshot(screen)




# Release resources
cv2.destroyAllWindows()