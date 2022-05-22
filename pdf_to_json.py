import re
import os

import json
from PyPDF2 import PdfFileReader

from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import *

json_keys = ["",
             "CITY ID",
             "CONTINENT",
             "City",
             "DAY",
             "DAY WITH",
             "EPRTRAnnexIMainActivityCode",
             "EPRTRAnnexIMainActivityLabel",
             "EPRTRSectorCode",
             "FacilityInspireID",
             "MONTH",
             "REPORTER NAME",
             "avg_temp",
             "avg_wind_speed",
             "countryName",
             "eprtrSectorName",
             "facilityName",
             "max_temp",
             "max_wind_speed",
             "min_temp",
             "min_wind_speed",
             "pollutant",
             "reportingYear",
             "targetRelease"]

"""
[('REPORT  CONTAMINACIÓN', 0), ('nº:', 1), ('81597', 2), 
('FACILITYNAME:', 3), ('MillerhillRecycling&  EnergyRecovery  Centre', 4), 
('FacilityInspireID:', 5), ('UK.SEPA/200002651.Facility', 6), 
('COUNTRY:', 7), ('UnitedKingdom', 8), 
('CONTINENT:', 9), ('EUROPE', 10), 
('CITY:', 11), ('Millerhill,Dalkeith', 12), 
('EPRTRSectorCode:', 13), ('5', 14), 
('eprtrSectorName:', 15), ('Waste  andwastewatermanagement', 16), 
('MainActivityCode:', 17), ('5(b)', 18), 
('targetRealase:', 19), ('AIR', 20), 
('pollutant:', 21), ('Nitrogenoxides(NOX)', 22), 
('emissions:', 23), ('175000', 24), 
('DAY:', 25), ('12', 26), 
('MONTH:', 27), ('6', 28), 
('YEAR:', 29), ('2019', 30), 
('METEOROLOGICAL  CONDITIONS', 31), 
('max_wind_speed:', 32), ('1,79E+15', 33), 
('min_wind_speed:', 34), ('2,2E+16', 35), 
('avg_wind_speed:', 36), ('2,04E+15', 37), 
('max_temp:', 38), ('1,51E+16', 39), 
('min_temp:', 40), ('1,82E+15', 41), 
('avg_temp:', 42), ('1,71E+16', 43), 
('DAYSFOG:', 44), ('10', 45),
('REPORTER  NAME:', 46), ('WilliamNelson', 47), 
('CITY_ID', 48), ('c662b4b4d859a9c224b5ac0acf221748', 49), 
('', 50)]
"""
pdf_val_pos = [1,
               49,
               10,
               12,
               26,
               45,
               18,
               16,
               14,
               6,
               28,
               47,
               43,
               37,
               8,
               16,
               4,
               39,
               33,
               41,
               35,
               22,
               30,
               20]


# json_schema =

def read_pdf_to_text_list(pdfPath):  # returns a string with the contents of the pdf
    reader = PdfFileReader(pdfPath)
    text = reader.getPage(0).extractText("")
    res_str = re.sub(r'(?<! ) (?! )', '', text)
    return res_str.split('\n')


def text_list_to_json(text_list, keys, index_list):
    if len(keys) == len(index_list):
        dict = {}
        for ind, key in enumerate(keys):
            dict[key] = text_list[index_list[ind]+1]
        return json.dumps(dict)
    else:
        print(f"key list length: {len(keys)} and index list length: {len(index_list)} must match")

def write_json(json, file):
    with open(file, "w") as outfile:
        outfile.write(json)

def parse_PDF_to_json(pdf_dir = "./test/src/data/train6",output_file = "./train6.json"):

    with open(output_file, "w") as outfile:
        outfile.write("[\n")
        for filename in os.listdir(pdf_dir):
            if filename != "pdfs-1.pdf":
                filepath = pdf_dir + "/" + filename
                text = read_pdf_to_text_list(filepath)
                json_obj = text_list_to_json(text,json_keys,pdf_val_pos)
                print(json_obj)
                outfile.write(json_obj)
                outfile.write(",\n")
        outfile.write("]")
        outfile.close()
####


if __name__ == '__main__':
    parse_PDF_to_json()
