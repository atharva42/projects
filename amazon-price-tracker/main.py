import gemail
from web_requests import Web


link = 'https://camelcamelcamel.com/product/B08KVMHH6L?context=search'
soup = Web(link)
price = float(soup.scrapper())
with open('product.txt') as file:
    lines = file.readlines()
    for line in lines:
        drop = line.split(',')
        print(drop[1])
        if price < float(drop[1]):
            print('sending an email!')
            gemail.send_email(drop[0], drop[1])
