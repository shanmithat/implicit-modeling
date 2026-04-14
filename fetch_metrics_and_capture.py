import os
import json
from playwright.sync_api import sync_playwright

os.makedirs('presentation_assets', exist_ok=True)
metrics = {}

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('http://localhost:8501')
    page.wait_for_timeout(5000)
    
    # GYROID
    page.locator('button:has-text("GENERATE STRUCTURE")').click()
    page.wait_for_timeout(25000) # Wait for processing
    page.screenshot(path='presentation_assets/gyroid_dashboard.png')
    
    vals = page.locator('[data-testid="stMetricValue"]').all_text_contents()
    if len(vals) >= 2:
        metrics['Gyroid'] = {"Volume Fraction": vals[0], "Estimated Mass": vals[1]}

    # DIAMOND
    page.get_by_text("Gyroid", exact=True).nth(0).click()
    page.get_by_role("option", name="Diamond").click()
    page.wait_for_timeout(2000)
    
    page.locator('button:has-text("GENERATE STRUCTURE")').click()
    page.wait_for_timeout(25000)
    page.screenshot(path='presentation_assets/diamond_dashboard.png')
    
    vals = page.locator('[data-testid="stMetricValue"]').all_text_contents()
    if len(vals) >= 2:
        metrics['Diamond'] = {"Volume Fraction": vals[0], "Estimated Mass": vals[1]}

    with open("analytics_reports.json", "w") as f:
        json.dump(metrics, f)
        
    browser.close()
