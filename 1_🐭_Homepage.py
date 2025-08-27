import streamlit as st

st.set_page_config(
    page_title = "LP App",
    page_icon = "🐭"
)

st.title("Benvenuto!🐿️")
st.sidebar.success("Seleziona il Canale qui sopra")

st.header("Istruzioni per l'uso")

st.write("*Per rendere leggibile la dashboard seleziona i filtri che ti interessano a destra. Scegli Country e Periodo (WEEK). Nota: il filtro date range non funziona per ora. Puoi selezionare anche gli MCI. La selezione può essere multipla.* \n\n *ATTENZIONE: se la dashboard ti da errore, ti basterà ri-modificare i filtri e ri-apparirà di nuovo* \n\n **Tabelle:** puoi renderle più leggibili ridimensionando le colonne a mano. \n\n **Grafici:** sono interattivi. Clicca su un valore della legenda per escluderlo. Se passi il mouse sul grafico puoi vedere più dettagli sui valori")
st.image("Filtri.png", caption="Filtri",width=300)

