# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:47:03 2025

@author: leor7
"""

"""

"""
from bs4 import BeautifulSoup
import requests 
from urllib.request import urlopen, Request
import json


header = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"}

url = "https://www.centennialcollege.ca/programs-courses/full-time/artificial-intelligence-online/"

url = url.replace(" ","")


req = Request(url=url, headers=header)
html = urlopen(req).read()

soup = BeautifulSoup(html, "html.parser")

#3 a) finding the title
title = soup.find("title")
title_text = title.get_text(strip = True)

print(title_text)

#3b find the program highlighs
#find the inital div containing all of the side bar options
pdf_button_div = soup.find("div", {"data-component": "programDetailsPdfButton","data-type": "layer"})

#Extracting the data-json attribute
data_json_str = pdf_button_div.get("data-json", "")
   
#Convert JSON string to Python dict
parsed_data = json.loads(data_json_str)
    
#Grab the list of content blocks
content_list = parsed_data.get("PdbTabContent", [])
    
#Find the block for the "Career Options and Education Pathways" tab
career_block = None

for item in content_list:
    if item.get("title") == "Career Options and Education Pathways":
        career_block = item
        break

    
#Parse the HTML in career_block["content"] to find Program Highlights
block_html = career_block.get("content", "")
block_soup = BeautifulSoup(block_html, "html.parser")
    
#Finding the program highlights using <h3>Program Highlights</h3>, then <ul>
h3 = block_soup.find("h3", string="Program Highlights")

    
highlights_ul = h3.find_next_sibling("ul")
    
#Extract text from each <li>
highlights_list = [li.get_text(strip=True) for li in highlights_ul.find_all("li")]
    

#printing the results of the program highlights
print(highlights_list)

for i in range(len(highlights_list)):
    print(highlights_list[i])

#3c the first 2 paragraphs of the program overview
overview = soup.find('div', {'data-component': 'pdbTabContent', 'data-type': 'layer'})

overview_json_str = overview.get('data-json', None)

data = json.loads(overview_json_str)

program_overview_html = None
for item in data:
    if item.get('type') == 'content' and item.get('title') == 'Program Overview':
        program_overview_html = item.get('content')
        break
    
content_soup = BeautifulSoup(program_overview_html, 'html.parser')
paragraphs = content_soup.find_all('p')
first_two_paragraphs = paragraphs[:2]

#printing the first 2 paragraphs
print(first_two_paragraphs)


#4 export the data
filename = "leor_my_future.txt"

with open(filename, "w", encoding="utf-8") as f:
    # Write the page title
    f.write("=== Page Title ===\n")
    f.write(f"{title_text}\n\n")
    
    # Write the program details
    f.write("=== Program Highlights ===\n")
    for i in range(len(highlights_list)):
        f.write(f"{highlights_list[i]}\n\n")

    
    
    # Write the paragraphs
    f.write("=== First Two Paragraphs ===\n")
    f.write(f"{first_two_paragraphs}")
    f.write("\n")
    
print(f"Data written to {filename}")