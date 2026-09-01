import os
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # path oberoende av vartifrån programmet startas

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

@st.cache_data(ttl=3300)
def get_spotify_token():
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        timeout=10
    )

    response.raise_for_status()

    return response.json()["access_token"]


def get_track_info(track_id, token):
    response = requests.get(
        f"https://api.spotify.com/v1/tracks/{track_id}",
        headers={"Authorization": f"Bearer {token}"},
        params={"market": "SE"},
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    title = data["name"]
    artist = ", ".join(artist["name"] for artist in data["artists"])

    release_date = data["album"]["release_date"]
    year = int(release_date[:4])

    is_playable = data.get("is_playable", True)

    return artist, title, year, is_playable

def get_tracks_info(track_ids, token):
    tracks = []

    for track_id in track_ids:
        response = requests.get(
            f"https://api.spotify.com/v1/tracks/{track_id}",
            headers={"Authorization": f"Bearer {token}"},
            params={"market": "SE"},
            timeout=10
        )

        if response.status_code == 200:
            tracks.append(response.json())
        else:
            tracks.append(None)

    return tracks

st.set_page_config(
    page_title="Moodify",
    page_icon=os.path.join(BASE_DIR, "assets", "moodify_logo_transparent.png"),
    layout="centered"
)

logo_col, title_col = st.columns(
    [1, 4],
    vertical_alignment="bottom"
)

with logo_col:
    st.image(
        os.path.join(BASE_DIR, "assets", "moodify_logo_transparent.png"),
        width=1100
    )

with title_col:
    st.title("M O O D I F Y")

st.write("Choose your mood and get Spotify recommendations.")

df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "278k_labelled_uri.csv"
    )
)

model = XGBClassifier()
model.load_model(
    os.path.join(BASE_DIR, "models", "xgboost_model.json")
)

feature_columns = [
    "duration (ms)",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

df["predicted_label"] = model.predict(df[feature_columns])

label_map = {
    "Melancholic": 0,
    "Upbeat": 1,
    "Energetic": 2,
    "Calm": 3
}

era_map = {
    "Any": None,
    "1970s": (1970, 1979),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2029)
}

with st.form("mood_form"):
    mood = st.radio(
        "How are you feeling?",
        list(label_map.keys()),
        horizontal=True
    )

    era = st.selectbox(
        "Choose an era",
        list(era_map.keys()),
        width=135
    )

    submitted = st.form_submit_button("Recommend songs")


if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "recommended_mood" not in st.session_state:
    st.session_state.recommended_mood = None


if submitted:
    label = label_map[mood]

    matching_songs = df[
        (df["predicted_label"] == label) &
        (df["speechiness"] < 0.66) &
        (df["duration (ms)"] < 600000)
    ]

    candidates = matching_songs.sample(
        n=min(50, len(matching_songs))
    )

    valid_tracks = []

    try:
        token = get_spotify_token()
        year_range = era_map[era]

        track_ids = [
            uri.split(":")[-1]
            for uri in candidates["uri"]
        ]

        spotify_tracks = get_tracks_info(track_ids, token)

        for (_, row), track in zip(candidates.iterrows(), spotify_tracks):
            if track is None:
                continue

            if not track.get("is_playable", True):
                continue

            release_date = track["album"]["release_date"]

            if not release_date:
                continue

            year = int(release_date[:4])

            if year_range is not None:
                start_year, end_year = year_range

                if not start_year <= year <= end_year:
                    continue

            valid_tracks.append(row)

            if len(valid_tracks) == 5:
                break

        st.session_state.recommendations = pd.DataFrame(valid_tracks)

        st.session_state.recommended_mood = mood

    except requests.HTTPError as e:
        if e.response.status_code == 429:
            retry_after = e.response.headers.get("Retry-After", "unknown")
            print(f"Spotify rate limit. Retry-After: {retry_after} seconds")
            st.warning("Spotify is temporarily limiting requests. Please try again later.")
        else:
            st.error(
                f"Spotify error {e.response.status_code}: {e.response.text}"
            )

    except requests.RequestException:
        st.error("Could not connect to Spotify.")


if st.session_state.recommendations is not None:

    st.subheader(
        f"Songs for a {st.session_state.recommended_mood.lower()} mood"
    )

    try:
        token = get_spotify_token()

        tracks = []

        # Get artist and song information for each recommendation
        for i, (_, row) in enumerate(
            st.session_state.recommendations.iterrows(),
            start=1
        ):
            track_id = row["uri"].split(":")[-1]

            try:
                artist, title, year, is_playable = get_track_info(track_id, token)

                tracks.append({
                    "track_id": track_id,
                    "artist": artist,
                    "title": title
                })

            except requests.RequestException:
                continue

        # Create Spotify players
        if tracks:
            players_html = ""

            for i, track in enumerate(tracks):
                players_html += f"""
                    <div style="margin-bottom: 10px;">
                        <strong>
                            {i + 1}. {track["artist"]} – {track["title"]}
                        </strong>

                        <div
                            id="spotify-player-{i}"
                            style="margin-top: 5px;">
                        </div>
                    </div>
                """

            player_data = ",".join(
                f'"spotify:track:{track["track_id"]}"'
                for track in tracks
            )

            html = f"""
                <script
                    src="https://open.spotify.com/embed/iframe-api/v1"
                    async>
                </script>

                {players_html}

                <script>
                    const trackUris = [{player_data}];
                    const controllers = [];

                    window.onSpotifyIframeApiReady = (IFrameAPI) => {{

                        trackUris.forEach((uri, index) => {{

                            const element =
                                document.getElementById(
                                    `spotify-player-${{index}}`
                                );

                            const options = {{
                                uri: uri,
                                width: "100%",
                                height: 80
                            }};

                            IFrameAPI.createController(
                                element,
                                options,
                                (controller) => {{

                                    controllers[index] = controller;

                                    controller.addListener(
                                        "playback_started",
                                        () => {{

                                            controllers.forEach(
                                                (
                                                    otherController,
                                                    otherIndex
                                                ) => {{

                                                    if (
                                                        otherController &&
                                                        otherIndex !== index
                                                    ) {{
                                                        otherController.pause();
                                                    }}
                                                }}
                                            );
                                        }}
                                    );
                                }}
                            );
                        }});
                    }};
                </script>
            """

            components.html(
                html,
                height=len(tracks) * 115
            )

        else:
            st.warning("No Spotify previews available.")

    except requests.RequestException:
        st.error("Could not connect to Spotify.")