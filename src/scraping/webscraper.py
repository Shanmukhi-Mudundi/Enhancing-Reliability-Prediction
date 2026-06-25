"""Web scraping for Amazon reviews using Selenium."""

import time
import csv
import re
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from typing import Optional

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def scrape_amazon_reviews(
    output_file: str = "data/raw/amazon_reviews.csv",
    search_query: str = "chair",
    target_reviews: int = 1000,
    batch_delay: float = 1.5
) -> None:
    """
    Scrape Amazon reviews using Selenium.
    
    Args:
        output_file: Path to save CSV file
        search_query: Product to search for
        target_reviews: Number of reviews to collect
        batch_delay: Delay between requests in seconds
    """
    driver = None
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialize webdriver
        driver = webdriver.Chrome()
        driver.implicitly_wait(5)
        logger.info("WebDriver initialized")
        
        # Navigate to Amazon
        driver.get("https://www.amazon.in/")
        time.sleep(2)
        logger.info("Navigated to Amazon.in")
        
        # Search for product
        search = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Search Amazon.in']"))
        )
        search.send_keys(search_query)
        search.send_keys(Keys.RETURN)
        time.sleep(2)
        logger.info(f"Searched for: {search_query}")
        
        # Click first product
        product = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//img[@class='s-image']"))
        )
        product.click()
        time.sleep(3)
        
        # Switch to product window
        original_window = driver.current_window_handle
        WebDriverWait(driver, 5).until(EC.number_of_windows_to_be(2))
        for window_handle in driver.window_handles:
            if window_handle != original_window:
                driver.switch_to.window(window_handle)
                break
        
        # Navigate to reviews
        try:
            review_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//a[@id='acrCustomerReviewLink']"))
            )
            review_button.click()
            time.sleep(2)
            logger.info("Navigated to reviews")
        except TimeoutException:
            logger.warning("Review button not found. Proceeding with available reviews...")
        
        # Manual login prompt
        input("Please log in manually, then press Enter to continue...")
        logger.info("User logged in, continuing with scraping...")
        
        # Scrape reviews loop
        _scrape_reviews_loop(driver, output_file, target_reviews, batch_delay)
        logger.info(f"✅ Scraping completed. Data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")
        raise
    finally:
        if driver:
            driver.quit()
            logger.info("WebDriver closed")


def _scrape_reviews_loop(
    driver,
    output_file: str,
    target_reviews: int,
    batch_delay: float
) -> None:
    """
    Main loop for scraping reviews.
    
    Args:
        driver: Selenium WebDriver instance
        output_file: Path to save CSV
        target_reviews: Target number of reviews
        batch_delay: Delay between batches
    """
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Review Text", "Review Date", "Helpful Votes", "Verified Purchase"])
        
        collected_reviews = 0
        
        while collected_reviews < target_reviews:
            try:
                driver.execute_script("window.scrollBy(0, 500);")
                time.sleep(2)
                
                see_more_buttons = driver.find_elements(By.XPATH, "//span[contains(text(), 'See more')]")
                for btn in see_more_buttons:
                    try:
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(1)
                    except:
                        continue
                
                reviews = driver.find_elements(By.XPATH, "//span[@data-hook='review-body']")
                dates = driver.find_elements(By.XPATH, "//span[@data-hook='review-date']")
                helpful_votes = driver.find_elements(By.XPATH, "//span[@data-hook='helpful-vote-statement']")
                verified_purchases = driver.find_elements(By.XPATH, "//span[@data-hook='avp-badge']")
                
                review_texts = [r.text.strip() for r in reviews]
                review_dates = [d.text.strip() for d in dates]
                review_helpful_votes = [
                    re.search(r"(\d+)", h.text).group(1) if re.search(r"(\d+)", h.text) else "0"
                    for h in helpful_votes
                ]
                verified_purchases_list = [
                    "Yes" if "Verified Purchase" in v.text else "No" for v in verified_purchases
                ]
                
                max_length = max(
                    len(review_texts), len(review_dates),
                    len(review_helpful_votes), len(verified_purchases_list)
                )
                review_texts.extend(["N/A"] * (max_length - len(review_texts)))
                review_dates.extend(["N/A"] * (max_length - len(review_dates)))
                review_helpful_votes.extend(["0"] * (max_length - len(review_helpful_votes)))
                verified_purchases_list.extend(["No"] * (max_length - len(verified_purchases_list)))
                
                for text, date, votes, verified in zip(
                    review_texts, review_dates, review_helpful_votes, verified_purchases_list
                ):
                    writer.writerow([text, date, votes, verified])
                    collected_reviews += 1
                    if collected_reviews >= target_reviews:
                        break
                
                logger.info(f"Collected: {collected_reviews}/{target_reviews} reviews")
                
                try:
                    next_button = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, "//li[@class='a-last']/a"))
                    )
                    driver.execute_script("arguments[0].scrollIntoView();", next_button)
                    next_button.click()
                    time.sleep(batch_delay)
                except TimeoutException:
                    logger.info("No more review pages found")
                    break
                    
            except Exception as e:
                logger.error(f"Error in scraping loop: {str(e)}")
                continue
