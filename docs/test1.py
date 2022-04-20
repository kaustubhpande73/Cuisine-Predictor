import urllib
import requests
import PyPDF2
import tempfile
import sqlite3
import re

import os

def extract():
    path = os.getcwd()
    path = path + '\\' +'incident_raw_data.txt'
    f = open(path, "rb")
    incident_data = open("C:\\Users\\Kaustubh\\Desktop\\test\\incident_raw_data.txt",'rb')
    
    #Extracts the data downloaded from the pdf to a temp file
    tfile = tempfile.TemporaryFile()

    # Write the pdf data to a temp file
    tfile.write(incident_data.read())

    # Set the curser to the begining
    tfile.seek(0)
    
    incident_rows, incident_list,j= [],[],[]

    #report_data = open('Sample.pdf.', 'rb') #opens the pdf file
    
    #Read the PDF
    pdfReader = PyPDF2.pdf.PdfFileReader(tfile)
    pagecount = pdfReader.getNumPages()
    incident_rows = []
    for a in range(1,pagecount):
        # Get page
        page = pdfReader.getPage(a).extractText()
        page = re.sub('Incident ORI','Incident ORI#',page)
        page = re.sub('EMSSTAT','EMSSTAT#',page)
        page = re.sub('OK0140200','OK0140200#',page)
        page = re.sub('14005','14005#',page)
        page = re.sub('14009','14009#',page)
        
        page_list = page.split('#')
        page_list = page_list[:-1]
        print(page_list)
        for j in page_list:
            j = j.split('\n')
            for k in j:
                if '' in j:
                    j.pop(0)
                   # print(j)
            incident_rows.append(j)
            for _ in range(len(incident_rows)):
                if len(incident_rows[_])<5:
                    incident_rows[_].insert(2,'NA')
                    incident_rows[_].insert(3,'NA')
           # print(incident_rows)
        return incident_rows

# fetch()
ans = extract()