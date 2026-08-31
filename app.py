import os
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

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
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    title = data["name"]
    artist = ", ".join(artist["name"] for artist in data["artists"])

    return artist, title

st.set_page_config(
    page_title="Moodify",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Moodify")
st.write("Choose your mood and get Spotify recommendations.")

df = pd.read_csv("data/raw/278k_labelled_uri.csv")

label_map = {
    "Melancholic": 0,
    "Upbeat": 1,
    "Energetic": 2,
    "Calm": 3
}

with st.form("mood_form"):
    mood = st.radio(
        "How are you feeling?",
        list(label_map.keys()),
        horizontal=True
    )

    submitted = st.form_submit_button("Recommend songs")


if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "recommended_mood" not in st.session_state:
    st.session_state.recommended_mood = None


if submitted:
    label = label_map[mood]

    matching_songs = df[
        (df["labels"] == label) &
        (df["speechiness"] < 0.66) &
        (df["duration (ms)"] < 600000)
    ]

    candidates = matching_songs.sample(
        n=min(20, len(matching_songs))
    )

    valid_tracks = []

    try:
        token = get_spotify_token()

        for _, row in candidates.iterrows():
            track_id = row["uri"].split(":")[-1]

            try:
                get_track_info(track_id, token)

                valid_tracks.append(row)

                if len(valid_tracks) == 5:
                    break

            except requests.RequestException:
                continue

    except requests.RequestException:
        st.error("Could not connect to Spotify.")

    st.session_state.recommendations = pd.DataFrame(valid_tracks)

    st.session_state.recommended_mood = mood


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
                artist, title = get_track_info(track_id, token)

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