import requests
import streamlit as st

API_KEY = st.secrets["TMDB_API_KEY"]

url = "https://api.themoviedb.org/3/search/movie"

params = {
    "api_key": API_KEY,
    "query": "The Dark Knight",
    "year": 2008
}

response = requests.get(url, params=params)

print("Status:", response.status_code)

if response.status_code == 200:
    data = response.json()

    print("Movie:", data["results"][0]["title"])
    print("Poster:", data["results"][0]["poster_path"])

else:
    print(response.text)