import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import spacy 
import re
from argparse import ArgumentParser
from pathlib import Path
import os
 
def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--output",
        type=Path
    )
    return args.parse_args()

def scrape_text(url):
    '''
    Scrape the text from the webpage
    :param url: str, url of the webpage
    '''
    response = requests.get(url)
    return_string  = ""
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        for paragraph in paragraphs:
            return_string += paragraph.get_text()
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return_string = None
    return return_string

def get_geo_info(url):
    '''
    Split scraped text into the sentences and filter out those that doo not contain geographical information
    :param url: str, url of the webpage
    '''
    scraped_text = scrape_text(url)
    if scraped_text is None:
        return None
    scraped_text = scraped_text.replace("\n", " ")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(scraped_text)
    sentences = [sent.text for sent in doc.sents]
    geo_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        has_ner_tag = False
        if any(ent.label_ in {"GPE", "LOC", "NORP"} for ent in doc.ents):
            geo_sentences.append(sentence)
    
    print(f"Before filtering: {len(sentences)} sentences")
    print(f"After filtering: {len(geo_sentences)} sentences")
    return geo_sentences

def preprocess_countries(input_text):
    '''
    Preprocess the countries from the raw text with available countries
    :param input_text: str, text containing the countries
    '''
    country_pattern = re.compile(r'([A-Za-z &]+) [a-z]{2} \d{2}\.\d{2}\.\d{4}')
    matches = country_pattern.findall(input_text)
    formatted_countries = ['-'.join(match.split()) if len(match.split()) > 1 else match for match in matches]
    formatted_countries = [country.lower() for country in formatted_countries]
    return formatted_countries

if __name__ == "__main__":
    args = parse_args()
    # geotpis
    continents = ["africa", "asia", "europe", "north-america", "oceania", "south-america"]
    geotips_text = []
    for continent in continents:
        url_to_scrape = f"https://geotips.net/{continent}/"
        geotips_text += get_geo_info(url_to_scrape)
    pd.DataFrame(geotips_text).to_csv(os.path.join(args.output,"geotips.csv"), index=False)
    # plonkit
    # raw text with avaialbel countries in plonkit
    countries_raw = " Africa       Flag Name  \t\t\t\t\t\tCountry Code \t\t\t\t\t Last updated        Botswana bw 23.01.2023      Egypt eg 06.07.2023      Eswatini sz 07.11.2022      Ghana gh 20.06.2023      Kenya ke 26.03.2023      Lesotho ls 03.05.2023      Madagascar mg 10.05.2023      Mali ml 06.07.2023      Réunion re 06.07.2023      Rwanda rw 23.12.2023     Senegal sn 27.07.2023      South Africa za 06.03.2023      Tunisia tn 23.01.2023      Uganda ug 07.03.2023                              Asia       Flag Name            Country Code          Last updated        Bangladesh bd 23.01.2023      Bhutan bt 06.03.2023      British Indian Ocean Territory io 16.01.2024      Cambodia kh 06.03.2023      China cn 07.02.2024      Hong Kong hk 15.10.2023      Indonesia id 07.04.2023      Iraq iq 07.02.2024      Israel & the West Bank il 08.12.2023      Jordan jo 23.01.2023      Kyrgyzstan kg 20.07.2023      Laos la 05.01.2023      Lebanon lb 07.02.2024      Macau mo 07.02.2024      Malaysia my 06.03.2023      Nepal np 07.02.2024      Pakistan pk 07.02.2024      Qatar qa 24.05.2023      Singapore sg 06.03.2023      South Korea kr 05.10.2023      Sri Lanka lk 19.04.2023      Taiwan tw 04.08.2023      Thailand th 24.12.2022      Turkey tr 16.01.2024      U.A.E ae 23.01.2023                              Europe       Flag Name                Country Code              Last updated        Albania al 14.07.2023      Andorra ad 06.03.2023      Austria at 18.03.2023      Belarus by 13.06.2023      Belgium be 19.04.2023      Croatia hr 18.03.2023      Czechia cz 26.04.2023      Denmark dk 07.06.2023      Estonia ee 31.05.2023      Faroe Islands fo 07.06.2023      Finland fi 29.12.2023      Germany de 07.12.2023      Gibraltar gi 13.06.2023      Greece gr 02.09.2023      Hungary hu 03.04.2023      Iceland is 18.03.2023      Ireland ie 31.07.2023      Italy it 11.09.2023      Isle of Man im 31.07.2023     Jersey je 06.03.2023      Latvia lv 31.05.2023      Lithuania lt 31.05.2023      Luxembourg lu 10.03.2023      Malta mt 13.06.2023      Monaco mc 14.10.2023      Montenegro me 13.06.2023      Netherlands nl 06.03.2023      North Macedonia mk 12.04.2023      Norway no 17.11.2023      Poland pl 15.10.2023      Romania ro 12.08.2023      Russia ru 10.01.2024      San Marino sm 07.11.2022      Serbia rs 15.10.2023      Slovakia sk 26.04.2023      Slovenia si 06.07.2023      Spain es 06.03.2023      Svalbard sj 17.11.2023      Sweden se 13.06.2023      Switzerland ch 28.02.2023      Turkey tr 16.01.2024      Ukraine ua 20.06.2023      United Kingdom uk 06.03.2023                             North America       Flag Name            Country Code          Last updated        Alaska us 12.10.2023      Bermuda bm 01.11.2023      Canada ca 18.03.2023      Dominican Republic do 23.01.2023      Greenland gl 20.03.2023      Guatemala gt 03.11.2023      Hawaii us 12.10.2023      Martinique mq 16.01.2024      Puerto Rico pr 12.10.2023      Saint Pierre and Miquelon pm 16.01.2024      United States us 12.10.2023      US Minor Outlying Islands um 12.10.2023      US Virgin Islands vi 12.10.2023                              Oceania       Flag Name            Country Code          Last updated        American Samoa as 12.10.2023      Australia au 06.03.2023      Christmas Island cx 24.12.2022      Cocos Islands cc 01.11.2023      Guam gu 15.10.2023      New Zealand nz 10.06.2023      Northern Mariana Islands mp 15.10.2023      Pitcairn Islands pn 16.01.2024      Vanuatu vu 16.01.2024                             South America       Flag Name            Country Code          Last updated        Argentina ar 07.06.2023      Brazil br 16.05.2023      Chile cl 17.05.2023      Colombia co 06.03.2023      Curaçao cw 01.11.2023      Ecuador ec 04.07.2023      Falkland Islands fk 01.11.2023      Panama pa 27.01.2024      South Georgia and the South Sandwich Islands gs 01.11.2023      Uruguay uy 06.04.2023"
    countries_list = preprocess_countries(countries_raw)
    transformed_countries = []
    for string in countries_list:
        if "last-updated" in string:
            parts = string.split('last-updated')
            last_part = parts[-1]
        else:
            last_part = string
        transformed_countries.append(last_part.lstrip("-").strip())
    plonkit_text = []
    for country in transformed_countries:
        url_to_scrape = f"https://www.plonkit.net/{country}"
        print(url_to_scrape)
        text_info = get_geo_info(url_to_scrape)
        if text_info != None:
            plonkit_text += text_info
    url_to_scrape = f"https://www.plonkit.net/israel-west-bank"
    text_info = get_geo_info(url_to_scrape)
    if text_info != None:
        plonkit_text += text_info
    url_to_scrape = f"https://www.plonkit.net/south-georgia-sandwich-islands"
    text_info = get_geo_info(url_to_scrape)
    if text_info != None:
        plonkit_text += text_info
    pd.DataFrame(plonkit_text).to_csv(os.path.join(args.output,'plonkit.csv'), index=False)
    merged_text = plonkit_text + geotips_text
    # longest item is random joined parts of the website
    longest_item = max(merged_text, key=len)
    merged_text.remove(longest_item)
    pd.DataFrame(merged_text).to_csv(os.path.join(args.output,'merged.csv'), index=False)