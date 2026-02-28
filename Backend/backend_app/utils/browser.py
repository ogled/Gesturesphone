import os
import shutil
import subprocess


def open_in_browser(url):
    if "DISPLAY" not in os.environ:
        print("[WARN] No DISPLAY found, GUI not available")
        return False

    browsers = [
        (
            "chromium-browser",
            [
                "--disable-infobars",
                "--noerrdialogs",
                "--window-focus",
                "--start-fullscreen",
                "--password-store=basic",
                "--user-data-dir=/tmp/chrome_temp_profile",
            ],
        ),
        (
            "chromium",
            [
                "--disable-infobars",
                "--noerrdialogs",
                "--window-focus",
                "--start-fullscreen",
                "--password-store=basic",
                "--user-data-dir=/tmp/chrome_temp_profile",
            ],
        ),
    ]

    for browser, args in browsers:
        path = shutil.which(browser)
        if path:
            subprocess.Popen([path, *args, url], env=os.environ)
            print(f"[INFO] Opened in {browser}")
            return True

    print("[WARN] Chromium not found")
    return False
