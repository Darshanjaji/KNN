import requests
import pandas as pd
from bs4 import BeautifulSoup
url = "https://vignanam.org/kannada.htm#&panel1-2"
res = requests.get(url)
print(res.content)