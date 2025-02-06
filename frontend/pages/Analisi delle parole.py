from backend import SparkBuilder
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import numpy as np

st.set_page_config(page_title="Analisi delle Parole", layout="wide")

@st.cache_data #dalla doc: serve per calcolare solo una volta il risultato di questa funzione
def wsa(n,min_frequency):
    positive_word_scores,negative_word_scores= spark.queryManager.words_score_analysis(n=n, min_frequency=min_frequency)
    positive_df = positive_word_scores.toPandas()
    negative_df = negative_word_scores.toPandas()
    return positive_df, negative_df

@st.cache_resource
def getSpark(appName):
    return SparkBuilder(appName)


def create_score_length_plot():
    df = spark.queryManager.review_length_analysis().toPandas()
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Assumiamo che ogni valore di 'Reviewer_Score' sia unico e già ordinato
    scores = df['Reviewer_Score']
    avg_positive_lengths = df['avg_positive_length']
    avg_negative_lengths = df['avg_negative_length']
    
    # Posizioni delle barre
    x = np.arange(len(scores))

    # Larghezza delle barre
    width = 0.35
    
    # Grafico delle barre
    ax.bar(x - width/2, avg_positive_lengths, width, label='Avg Positive Length', color='green')
    ax.bar(x + width/2, avg_negative_lengths, width, label='Avg Negative Length', color='red')

    # Aggiungiamo etichette e titolo
    ax.set_xlabel('Reviewer Score')
    ax.set_ylabel('Average Length')
    ax.set_title('Average Positive and Negative Lengths by Reviewer Score')
    ax.set_xticks(x)
    ax.set_xticklabels(scores, rotation=90)
    ax.set_xticklabels(scores)
    ax.legend()

    return plt

spark = getSpark("BigData_App")

st.title("📊 Analisi delle Parole nelle Recensioni")
st.write("La pagina mostra le top parole positive e negative nelle recensioni, è possibile impostare il numero di parole da visualizzare e la frequenza minima necessaria per includere quella parola nell'analisi.")

st.sidebar.title("🔍 Navigazione")
st.sidebar.markdown("### Sezioni disponibili:")
st.sidebar.markdown("- 🏠 **Home**")
st.sidebar.markdown("- 📍 **Mappa Hotel**")
st.sidebar.markdown("- 📊 **Trend & Analisi**")
st.sidebar.markdown("- 🔍 **Anomaly Detection**")
st.sidebar.markdown("- 📝 **Word Cloud**")

st.markdown("## Imposta i parametri di analisi:")
n = st.selectbox("Numero di parole da visualizzare", [20, 50, 100])
min_frequency = st.selectbox("Frequenza minima della parola", [100, 1000, 5000])

with st.spinner("⚙️ Elaborazione dati..."):
    positive_df, negative_df = wsa(n=n, min_frequency=min_frequency)

@st.cache_data
def plot_wordcloud(word_scores, title):
    words_freq = dict(zip(word_scores["word"], word_scores["word_count"]))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(words_freq)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

col1, col2 = st.columns(2)

if(n<=50):
    with col1:
        with st.spinner("attendere..."):
            st.subheader("📊 Top parole positive")
            st.dataframe(positive_df)
            st.subheader("📈 Distribuzione parole positive")
            fig, ax = plt.subplots(figsize=(32, 16))
            ax.barh(positive_df["word"][:n], positive_df["avg_score"][:n], color="green")
            ax.set_xlabel("Punteggio medio", fontsize=20)
            ax.set_ylabel("Parola", fontsize=20)
            ax.set_title("Aggettivi con punteggio più alto",fontsize=40)
            ax.tick_params(axis='y', labelsize=30)  # Aumenta la dimensione del testo sulle etichette Y
            st.pyplot(fig)
        with st.spinner("generazione delle word cloud..."):
            st.subheader("☁️ Word Cloud delle parole positive")
            plot_wordcloud(positive_df, "Parole Positive")

    with col2:
        with st.spinner("attendere..."):
            st.subheader("📊 Top parole negative")
            st.dataframe(negative_df)
            st.subheader("📈 Distribuzione parole negative")
            fig, ax = plt.subplots(figsize=(32, 16))
            ax.barh(negative_df["word"][:n], negative_df["avg_score"][:n], color="red")
            ax.set_xlabel("Punteggio medio", fontsize=20)
            ax.set_ylabel("Parola", fontsize=20)
            ax.set_title("Aggettivi con punteggio più basso", fontsize=40)
            ax.tick_params(axis='y', labelsize=30)  
            st.pyplot(fig)
        with st.spinner("generazione delle word cloud..."):
            st.subheader("☁️ Word Cloud delle parole negative")
            plot_wordcloud(negative_df, "Parole Negative")
        
else:
    
    with col1:
        with st.spinner("attendere..."):
            st.subheader("📊 Top parole positive")
            st.dataframe(positive_df)
            st.subheader("📈 Distribuzione parole positive")
            fig, ax = plt.subplots(figsize=(32, 16))
            ax.barh(positive_df["word"][:n], positive_df["avg_score"][:n], color="green")
            ax.set_xlabel("Punteggio medio", fontsize=20)
            ax.set_ylabel("Parola", fontsize=20)
            ax.set_title("Aggettivi con punteggio più alto",fontsize=40)
            ax.tick_params(axis='y', labelsize=10)  
            st.pyplot(fig)
        with st.spinner("generazione delle word cloud..."):
            st.subheader("☁️ Word Cloud delle parole positive")
            plot_wordcloud(positive_df, "Parole Positive")

    with col2:
        with st.spinner("attendere..."):
            st.subheader("📊 Top parole negative")
            st.dataframe(negative_df)
            st.subheader("📈 Distribuzione parole negative")
            fig, ax = plt.subplots(figsize=(32, 16))
            ax.barh(negative_df["word"][:n], negative_df["avg_score"][:n], color="red")
            ax.set_xlabel("Punteggio medio", fontsize=20)
            ax.set_ylabel("Parola", fontsize=20)
            ax.set_title("Aggettivi con punteggio più basso", fontsize=40)
            ax.tick_params(axis='y', labelsize=10)  
            st.pyplot(fig)
        with st.spinner("generazione delle word cloud..."):
            st.subheader("☁️ Word Cloud delle parole negative")
            plot_wordcloud(negative_df, "Parole Negative")

st.download_button(
    "📥 Scarica dati in CSV",
    positive_df.to_csv(index=False) + "\n" + negative_df.to_csv(index=False),
    "word_scores.csv",
    "text/csv",
)

st.subheader("Analisi lunghezza recensioni in corrispondenza dei punteggi assegnati")
with st.spinner("Calcolando lunghezza media delle recensioni per ciascun score..."):
    st.pyplot(create_score_length_plot())
    