# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:05:20 2019

@author: Tanner
"""

from endnoteParser import genDF
import pickle
import pandas as pd


atp_df = genDF('AtpRelease/data/initialAtp.xml', 'AtpRelease/data/finalAtp.xml')
astro_df = genDF('Astronauts/data/initialAstronaut.xml', 'Astronauts/data/finalAstronaut.xml')

atp_df.to_pickle('AtpRelease/data/atp_df_no_paper.pkl')
astro_df.to_pickle('Astronauts/data/astro_df_no_paper.pkl')
print("Done")