import cv2
import numpy as np
import mss
import time
from datetime import datetime 
import os

template_files = ['Online.png', 'Idle.png', 'DND.png', 'Invisible.png']
templates = []

for template_file in template_files:
    template = cv2.imread(template_file)
    if template is None:
        print(f"Error: Template image {template_file} not found. Check the file path.")
        exit(1)
    templates.append(template)

def resize_template(template, scale_percent):
    width = int(template.shape[1] * scale_percent / 100)
    height = int(template.shape[0] * scale_percent / 100)
    return cv2.resize(template, (width, height))

def save_screenshot(screen):
    directory = "saves"
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(directory, f"screenshot_{timestamp}.png")
    cv2.imwrite(filename, screen)

with mss.mss() as sct:
    monitor_2 = sct.monitors[1]  

    resized_templates = [resize_template(template, 15) for template in templates]

    while True:
        screen = np.array(sct.grab(monitor_2))

        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

        for i, template in enumerate(resized_templates):
            res_typing = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc_typing = np.where(res_typing >= threshold)

            if len(loc_typing[0]) > 0:
                template_name = template_files[i]
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{template_name} detected at {current_time}! Someone might be typing.")
                save_screenshot(screen)
                break  

cv2.destroyAllWindows()
