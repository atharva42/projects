import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException

driver_path = 'C:\Chrome_driver_me\chromedriver.exe'
link = 'https://www.linkedin.com/jobs/search/?currentJobId=2552659332&f_AL=true&keywords=python%20intern'
email = //login_email_here
password = //LinkedIn_password
mobile = //Registered_mobile_number
months = '3'

driver = webdriver.Chrome(driver_path)
driver.get(link)


def find(drivers):
    elements = drivers.find_element_by_xpath('/html/body/main/div/div/form[2]/section/p/button')
    if elements:
        return elements
    else:
        return False


try:  # This is to login to the page mentioned in the photo
    find(driver)
    element = WebDriverWait(driver, 5).until(find)
    element.click()
    email_box = driver.find_element_by_name('session_key')
    email_box.send_keys(email)
    password_box = driver.find_element_by_name('session_password')
    password_box.send_keys(password)
    sign_in = driver.find_element_by_id('login-submit')
    sign_in.click()
except NoSuchElementException:  # If another login page loads this is the code which works completely fine
    change_method = driver.find_element_by_link_text('Sign in')
    change_method.click()
    name = driver.find_element_by_id('username')
    passkey = driver.find_element_by_id('password')
    sign_in = driver.find_element_by_class_name('btn__primary--large')
    name.send_keys(email)
    passkey.send_keys(password)
    sign_in.click()
link_of_jobs = driver.find_elements_by_class_name("jobs-search-results__list-item")
for i, job in enumerate(link_of_jobs):
    job.click()
    time.sleep(2)
    try:
        apply_button = driver.find_element_by_class_name('jobs-apply-button')
        apply_button.click()
        time.sleep(2)

        # If phone field is empty, then fill your phone number.
        phone = driver.find_element_by_class_name("fb-single-line-text__input")
        if phone.text == "":
            phone.send_keys(mobile)

        submit_button = driver.find_element_by_css_selector("footer button")

        # If the submit_button is a "Next" button, then this is a multi-step application, so skip.
        if submit_button.get_attribute("data-control-name") == "continue_unify":
            close_button = driver.find_element_by_class_name("artdeco-modal__dismiss")
            close_button.click()
            time.sleep(1)
            discard_button = driver.find_elements_by_class_name("artdeco-modal__confirm-dialog-btn")[1]
            discard_button.click()
            print("Complex application, skipped.")
            continue
        else:
            submit_button.click()

        # Once application completed, close the pop-up window.
        time.sleep(2)
        close_button = driver.find_element_by_class_name("artdeco-modal__dismiss")
        close_button.click()

    # If already applied to job or job is no longer accepting applications, then skip.
    except NoSuchElementException:
        print("No application button, skipped.")
        continue
