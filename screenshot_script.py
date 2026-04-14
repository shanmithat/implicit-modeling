import os
import time
import json
from playwright.sync_api import sync_playwright

os.makedirs('presentation_assets', exist_ok=True)

metrics = {}

def extract_metrics(page, lattice_type):
    # Retrieve text from the elements. Streamlit metric values have data-testid="stMetricValue"
    # Wait for completion - "STRUCTURE VALIDATED | READY", Streamlit status changes dynamically.
    page.wait_for_selector('text=STRUCTURE VALIDATED | READY', timeout=60000)
    # wait a bit for render
    time.sleep(4)
    # Capture screenshot
    page.screenshot(path=f'presentation_assets/{lattice_type.lower()}_dashboard.png')

    # Get metrics
    metric_values = page.locator('[data-testid="stMetricValue"]').all_text_contents()
    # first is vol fraction, second is mass
    if len(metric_values) >= 2:
        metrics[lattice_type] = {
            "Volume Fraction": metric_values[0],
            "Estimated Mass": metric_values[1]
        }

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto('http://localhost:8501', timeout=60000)
    time.sleep(10) # wait for streamlit initial load

    # 1. We are already at default configuration (Gyroid, 0.8mm cell size).
    # Click generate
    page.locator('button:has-text("GENERATE STRUCTURE")').click()
    extract_metrics(page, "Gyroid")
    time.sleep(2)

    # 2. Change to Diamond
    # The selectbox is "[data-testid='stSelectbox']" or similar.
    # We can use text locators:
    # the label is LATTICE TYPE
    page.get_by_text("Gyroid", exact=True).nth(0).click()
    # click Diamond
    page.get_by_role("option", name="Diamond").click()
    time.sleep(2)

    page.locator('button:has-text("GENERATE STRUCTURE")').click()
    extract_metrics(page, "Diamond")

    with open("analytics_results.json", "w") as f:
        json.dump(metrics, f)

    browser.close()
