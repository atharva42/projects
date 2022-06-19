import requests
from bs4 import BeautifulSoup


class Web:
    def __init__(self, link):
        self.response = requests.get(url=link,  headers={"User-Agent": "Defined"}).text

    def scrapper(self):
        soup = BeautifulSoup(self.response, 'html.parser')
        raw_price = soup.find_all('span', class_='green')[0].getText()
        if '$' in raw_price:
            price = raw_price.split('$')
            return price[1]


