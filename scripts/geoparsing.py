import pandas as pd
import json
import spacy
import requests
from bs4 import BeautifulSoup
import os
from argparse import ArgumentParser
from pathlib import Path

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--output_guidebook",
        type=Path
    )
    args.add_argument(
        "--output_pseudo_labels",
        type=Path
    )
    args.add_argument(
        "--data_path",
        type=Path
    )
    args.add_argument(
        "--geo_sentences_csv",
        type=Path
    )
    args.add_argument(
        "--countries_csv",
        type=Path
    )
    return args.parse_args()


def create_country_dict(dataframe):
    '''
    Create a dictionary that maps country names, adjectivals and demonyms to the country name
    :param dataframe: pd.DataFrame, dataframe containing the country names, adjectivals and demonyms
    '''
    country_dict = {}
    for index, row in dataframe.iterrows():
        country_name = row['Country/entity name']
        adjectivals = [x.strip() for x in str(row['Adjectivals']).split(',')]
        demonyms = [x.strip() for x in str(row['Demonyms']).split(',')]
        country_dict[country_name] = country_name
        for adj in adjectivals:
            country_dict[adj] = country_name
        for dem in demonyms:
            country_dict[dem] = country_name
    return country_dict

def parse_clues(clues, demonym_to_country):
    '''
    Parse the clues and countries
    :param clues: list, list of clues
    :param demonym_to_country: dict, dictionary that maps country names, adjectivals and demonyms to the country name
    '''
    nlp = spacy.load("en_core_web_sm")
    geoparsed_clues = []
    for clue_id, clue in enumerate(clues):
        doc = nlp(clue)
        geoparsed_info = []
        for ent in doc.ents:
            country_name = demonym_to_country.get(ent.text, None)
            if country_name:  
                geoparsed_info.append({
                    "text": ent.text,
                    "Country": country_name,
                    "start": ent.start,
                    "end": ent.end
                })
        geoparsed_clues.append({
            "CLUE_ID": clue_id,
            "text": clue,
            "geoparsed": geoparsed_info
        })
    return geoparsed_clues

if __name__ == "__main__":
    args = parse_args()
    clues = pd.read_csv(args.geo_sentences_csv)['0'].tolist()
    countries  = pd.read_csv(args.countries_csv)
    country_denonyms_dict = create_country_dict(countries)
    geopal_clues = parse_clues(clues, country_denonyms_dict)
    with open(args.output_guidebook, "w") as outfile: 
        json.dump(geopal_clues, outfile)
    pseudo_labels = {}
    splits = ["train", "val", "test"]
    for split in splits:
        with open(os.path.join(args.data_path,rf"{split}\label_mapping\countries_names.json"),"r") as f:
            files_to_country= json.load(f)
        for key, value in files_to_country.items():
            pseudo_labels[key] = [[]]
            for clue in geopal_clues:
                for country_dict in clue['geoparsed']:
                    if country_dict["Country"] == value[0]:
                        pseudo_labels[key][0].append(clue['CLUE_ID'])
    with open(args.output_pseudo_labels, "w") as outfile: 
        json.dump(pseudo_labels, outfile)
