import requests
import os
from bs4 import BeautifulSoup
from PIL import Image

import shutil
import random
from imutils import paths
from numpy import asarray
from csv import DictWriter

def scraping(soup, direktori, header):
    fulltag = soup.find_all('div', class_='thumbnail d-flex justify-content-center')
    i=0
    
    for tag in fulltag:
        href = tag.find('a')['href'].strip('.')
        missimg = tag.find('div', class_='missingcover text-center mt-2')
        
        if missimg is None:
            page = 'http://www.fictiondb.com'+href

            pageReq = requests.get(page, headers=header)

            pageSoup = BeautifulSoup(pageReq.text, 'html.parser')
            title = pageSoup.find('img', class_='img-fluid')['alt']
            cover = pageSoup.find('img', class_='img-fluid')['src']
            isbn = pageSoup.find('meta', attrs={'property':'og:isbn'})['content']

            i+=1
            if isbn=="":
                isbn = 'null_'+str(i)
                
            pathisbn = isbn+'.jpg'
            
            print(i, title, cover, isbn)

            if pathisbn in os.listdir(direktori):
                pass
            else:
                img_req = requests.get(cover, stream=True)
                img = Image.open(img_req.raw)
                img = img.resize((256, 256), Image.BICUBIC)
                img.convert('RGB').save(direktori+pathisbn, quality=100)
        
        else:
            pass
        
    print('\n-----------DONE------------')

# Function to create cvs file for the dataset label 
def create_csv(csv, isbn, f, g, h, m, r):
    fields = ['isbn', 'fantasy', 'general', 'horror', 'mystery', 'romance']
    newdict = {'isbn':isbn, 'fantasy':f, 'general':g, 'horror':h, 'mystery':m, 'romance':r}

    if not os.path.exists(csv):
        with open(csv, 'a+', encoding='utf-8', newline='') as csvfile:
            writer = DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerow(newdict)
    else:
        with open(csv, 'a', encoding='utf-8', newline='') as csvfile:
            writer = DictWriter(csvfile, fieldnames=fields)
            writer.writerow(newdict)
            
# Transform scraped test label to csv and integrate cover files into one folder
def set_test_data(source_folder, test_folder):
    label = []
    
    for file in source_folder:
        shutil.copy(file, test_folder)
        
        lb = file.split(os.path.sep)[1].split('_')
        label.append(lb)
        
        isbn = file.split(os.path.sep)[2]
        f = g = h = m = r = 0
        
        for genre in lb:
            if genre=='fantasy':
                f=1
            elif genre=='general':
                g=1
            elif genre=='horror':
                h=1
            elif genre=='mystery':
                m=1
            elif genre=='romance':
                r=1
        create_csv('test_label.csv', isbn, f, g, h, m, r)
    labelset = asarray(label)
    
    return labelset
    
    
    
# Main starts from here   
dir_address = 'raw-test/fantasy_general/'

header={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Accept-Encoding': '*',
    'Connection': 'keep-alive'
}

year = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

for i in year:
    html = 'https://www.fictiondb.com/search/searchresults.php?author=&title=&series=&isbn=&datepublished='+i+'&synopsis=&rating=-&anthology=&imprint=0&pubgroup=0&srchtxt=multi&styp=6&GN9001N=on&GN2044N=on&GN2004N=on&GN2042N=on&GN2008Y=on&GN2009N=on&GN2040N=on&GN2014N=on&GN2015N=on&GN2016N=on&GN2019Y=on&GN2022N=on&GN2023N=on&GN2006N=on&GN3129N=on&GN1007N=on&GN1006N=on&GN1004N=on&GN1008N=on&ltyp=1'
    request = requests.get(html, headers=header)
    webpage = request.text
    
    bs = BeautifulSoup(webpage, 'html.parser')
    scraping(bs, dir_address, header)
    
 
# 
list_dir = sorted(list(paths.list_images('raw-test')))
random.shuffle(list_dir)

y = set_test_data(list_dir, 'test/')
