import pandas as pd
import numpy as np
import re
import datetime
import streamlit as st
from datetime import datetime


#file = "/Users/jmechach/Desktop/LP_Dashboard/Meta.csv"
#file = "/Users/jmechach/Desktop/LP_Dashboard/Taboola.csv"
file = "Taboola.csv"



@st.cache_data

#---- Funzioni Varie ----

def get_week(value):
    week_number = value.isocalendar()[1]
    return(week_number)

def extract_country(value): 
    if isinstance(value, str):
        lista = value.split('_')
        country = str(lista[1])
        return country
    return

def extract_mci(value): 
    if isinstance(value, str):
        match = re.findall(r'(.+)\?mci=(\w+)', value)
        return str(match[0][1]) if match else None
    return 

def extract_date(value): 
    if isinstance(value, str):
        if ' - ' in value:
            match = re.split(" - ", value)
            return str(match[1]) if match else np.nan
    return np.nan 

def extract_url(value): 
    if isinstance(value, str):
        match = re.findall(r'(.+)\?mci=(\w+)', value)
        return str(match[0][0]) if match else np.nan
    return np.nan 

#---- CLEANING ----

def load_data():
    df = pd.read_csv(file)
    df = df.rename(columns={'Url': 'URL'})
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Week'] = df['Date'].apply(get_week)
    df['MCI'] = df['URL'].apply(extract_mci)
    df['URL'] = df['URL'].apply(extract_url)
    df['Country'] = df['Campaign Name'].apply(extract_country)

    return df.dropna(subset=["Campaign Name", "URL"])

df = load_data()

#---- DASHBOARD ----

import folium
import plotly.express as px
import pydeck as pdk
import openpyxl

st.title(f'🗞️ NATIVE: Landing Page Dashboard')

show_all = st.sidebar.checkbox("Sidebar", value=False)

if show_all:
    filtered_df = df
else:
    st.sidebar.header('Filters')


##----- SLIDER FILTRI -----

    #----- URL -----
    url = df['URL'].unique()
    
    url_all = st.sidebar.checkbox("ALL_URLs", value=True)

    campaign = st.sidebar.multiselect(
    'Select URL',
    url, 
    disabled=url_all, 
    )
    #----- Country -----
    outcome = df['Country'].unique()
    all_country = st.sidebar.checkbox("Country", value=True)
    
    country = st.sidebar.multiselect(
    'Country',
    outcome, 
    disabled=all_country
    )
    #----- MCI -----
    mci = df['MCI'].unique()
    
    mci_all = st.sidebar.checkbox("ALL_MCIs", value=True)
 
    mcis = st.sidebar.multiselect(
    'Select MCI',
    mci, 
    disabled=mci_all, 
    )
    #----- WEEK -----
    week = df['Week'].unique()
    
    week_all = st.sidebar.checkbox("ALL_WEEKs", value=True)
 
    weeks = st.sidebar.multiselect(
    'Select Week',
    week, 
    disabled=week_all, 
    )
    #----- Date -----
    date_range = st.sidebar.date_input('Select Date Range', value=["2024-01-01","2026-01-01"])

    filtered_df = df
    
    if not url_all:
        filtered_df = filtered_df[filtered_df['URL'].isin(campaign)]
    
    if not all_country:
        filtered_df = filtered_df[filtered_df['Country'].isin(country)]
    
    if not date_range:
         filtered_df = filtered_df[(filtered_df['date'] == date_range)]  

    if not week_all:
        filtered_df = filtered_df[filtered_df['Week'].isin(weeks)]

    if not mci_all:
        filtered_df = filtered_df[filtered_df['MCI'].isin(mcis)]

st.write(f"🌐 **{len(list(filtered_df['URL'].unique()))}** URLs Found")
st.write(f"🆔 **{len(list(filtered_df['MCI'].unique()))}** MCIs Found",divider=True)

##----- TABELLA -----

#zip_states = filtered_df_yes.groupby('State_y').size().reset_index(name='Appt')
#df_recap = filtered_df.groupby('URL')['CPA'].sum().reset_index(name='Leads')

def cpa_aggregation(group_df):
    total_spent = group_df['Spent'].sum()
    total_conversions = group_df['Conversions'].sum()
    return pd.Series({
        'Spent': total_spent,
        'Conversions': total_conversions,
        'CPA': total_spent / total_conversions if total_conversions > 0 else float('nan')
    })

pivot_with_cpa = filtered_df.groupby('URL').apply(cpa_aggregation).reset_index()

selected_rows = st.data_editor(
    pivot_with_cpa
    .sort_values(by="Conversions", ascending=False)
    .reset_index(drop=True),
    use_container_width=True,
    height=400,
    width=1500,
    num_rows="dynamic",
    hide_index=True,
    column_config={"Link": st.column_config.LinkColumn()},
    key="table_selection"
)

st.divider()


##----- TREND CPL per URL per WEEK -----

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import plotly.express as px
from numpy.random import default_rng as rng

st.subheader("💸 Trend CPL by Week")

pivot1 = filtered_df.pivot_table(columns='URL', index='Date', values='CPA', aggfunc="mean").reset_index().fillna(0)

fig = px.line(pivot1, x='Date', y=pivot1.columns[1:], title="Cost per result by URL")
fig.update_layout(legend=dict(yanchor="top", y=-0.5, xanchor="left", x=0.2),showlegend=True)

st.plotly_chart(fig, use_container_width=True)

st.divider()
##----- + Tavola
grouped = filtered_df.groupby(['URL', 'Week']).agg({
    'Spent': 'sum',
    'Conversions': 'sum'
}).reset_index()

grouped['CPA'] = (grouped['Spent'] / grouped['Conversions']).map("{:,.1f}".format)
pivot_week_url = grouped.pivot(index='URL', columns='Week', values='CPA')

st.markdown('**URL performance by week**')
st.write(pivot_week_url)

st.divider()

##----- TREND CPL per MCI per WEEK -----

st.subheader("💸 Trend MCI by Week")

pivot2 = filtered_df.pivot_table(columns='MCI', index='Date', values='CPA',aggfunc="mean").reset_index().fillna(0)

fig = px.line(pivot2, x='Date', y=pivot2.columns[1:], title="Cost per result by MCI")

st.plotly_chart(fig, use_container_width=True)


st.divider()

##----- + Tavola
grouped = filtered_df.groupby(['MCI', 'Week']).agg({
    'Spent': 'sum',
    'Conversions': 'sum'
}).reset_index()

grouped['CPA'] = (grouped['Spent'] / grouped['Conversions']).map("{:,.1f}".format)
pivot_week_url = grouped.pivot(index='MCI', columns='Week', values='CPA')

st.markdown('**MCI performance by week**')
st.write(pivot_week_url)

st.divider()



