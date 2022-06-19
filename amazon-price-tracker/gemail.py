import smtplib


def send_email(product_name, product_price):
    TO = //Receivers_email
    FROM = //senders_email
    PASSKEY = //Email_login_password
    connection = smtplib.SMTP('smtp.gmail.com', port=587)
    connection.starttls()
    connection.login(user=FROM, password=PASSKEY)
    connection.sendmail(from_addr=FROM, to_addrs=TO, msg=f'Subject: Price DRop!!!!\n\nYour product {product_name} has a price drop to ${product_price}. Hurry Now!')
    print('Email sent!')
    connection.close()
