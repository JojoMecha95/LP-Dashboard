import pandas as pd
import numpy as np
import re
import datetime
import streamlit as st

#file = "/Users/jmechach/Desktop/LP_Dashboard/Meta.csv"
#file = "/Users/jmechach/Desktop/LP_Dashboard/META_ALL COUNTRIES.csv"
file = "META_ALL COUNTRIES.csv"


@st.cache_data

#---- Funzioni Varie ----

def get_country(value):
    if isinstance(value, str):
        return str(value[:2]).upper()
    return np.nan

def get_mci(value):
    if isinstance(value, str):
        return str(value[len(value)-15:len(value)]).upper()
    return np.nan

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
    df = df.rename(columns={'Link (ad settings)': 'URL','Amount spent (EUR)':'Spent'})
    df['Country'] = df['Ad name'].apply(get_country)
    df['MCI'] = df['Ad name'].apply(get_mci)
    df['date'] = pd.to_datetime(df['Week'].apply(extract_date),errors="coerce")
    df['URL'] = df['URL'].apply(extract_url)

    df = df[~df['URL'].isin(["http://fb.me/"])]

    return df.dropna(subset=["Results", "URL"])

df = load_data()



#---- DASHBOARD ----

import folium
import plotly.express as px
import pydeck as pdk
import openpyxl

st.title(f'📱 Landing Page Dashboard')

show_all = st.sidebar.checkbox("Sidebar", value=False)

if show_all:
    filtered_df = df
else:
    st.sidebar.header('Filters')

##----- SLIDER FILTRI -----
    #----- Leads -----
    
    min_leads = int(min(df["Results"]))
    max_leads = int(max(df["Results"]))

    selected_attempts = st.sidebar.slider("Number of Leads", min_leads, max_leads, (1, 50))
    
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
    #----- Date -----
    date_range = st.sidebar.date_input('Select Date Range', value=["2023-01-01","2026-01-01"])

    filtered_df = df[
       (df["Results"] >= min_leads) & (df["Results"] <= max_leads)
    ]

    if not url_all:
        filtered_df = filtered_df[filtered_df['URL'].isin(campaign)]
    
    if not all_country:
        filtered_df = filtered_df[filtered_df['Country'].isin(country)]
    
    if not date_range:
         filtered_df = filtered_df[(filtered_df['date'] == date_range)]
    
    if not mci_all:
        filtered_df = filtered_df[filtered_df['MCI'].isin(mcis)]

st.write(f"🌐 **{len(list(filtered_df['URL'].unique()))}** URLs Found")
st.write(f"🆔 **{len(list(filtered_df['MCI'].unique()))}** MCIs Found",divider=True)


##----- TABELLA -----

def cpa_aggregation(group_df):
    total_spent = group_df['Spent'].sum()
    total_conversions = group_df['Results'].sum()
    return pd.Series({
        'Spent': total_spent,
        'Conversions': total_conversions,
        'CPA': total_spent / total_conversions if total_conversions > 0 else float('nan')
    })

pivot_with_cpa = filtered_df.groupby('URL').apply(cpa_aggregation).reset_index()
df = pivot_with_cpa.style.format('{:.2f$}')
#df_recap = filtered_df.groupby('URL')['Cost per result'].mean().reset_index(name='Leads')

selected_rows = st.data_editor(
    pivot_with_cpa
    .sort_values(by="Spent", ascending=False)
    .reset_index(drop=True),
    use_container_width=False,
    height=400,
    width=1000,
    num_rows="dynamic",
    hide_index=True,
    column_config={"Link": st.column_config.LinkColumn()},
    key="table_selection"
)

st.divider()

#df.to_csv('/Users/jmechach/Desktop/LP_Dashboard/Meta_modificato.csv')

##----- TREND CPL per URL per WEEK -----

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import plotly.express as px
from numpy.random import default_rng as rng

st.subheader("💸 Trend CPL by Week")

pivot1 = filtered_df.pivot_table(columns='URL', index='date', values='Cost per result', aggfunc="mean").reset_index()

fig = px.line(pivot1, x='date', y=pivot1.columns[1:], title="Cost per result by URL")
fig.update_layout(legend=dict(yanchor="top", y=-0.5, xanchor="left", x=0.2),showlegend=True)

st.plotly_chart(fig, use_container_width=True)

st.divider()


##----- TREND CPL per MCI per WEEK -----

st.subheader("💸 Trend MCI by Week")

pivot2 = filtered_df.pivot_table(columns='MCI', index='date', values='Cost per result',aggfunc="mean").reset_index()

fig = px.line(pivot2, x='date', y=pivot2.columns[1:], title="Cost per result by MCI")

st.plotly_chart(fig, use_container_width=True)

#df.plot(x="Month", y="Sales", kind="line", title="Monthly Sales", legend=True,color='red')

st.divider()

"""""""""
