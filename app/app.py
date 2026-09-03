import os
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # path oberoende av vartifrån programmet startas

load_dotenv(override=True)


def get_secret(key):
    """Read a credential from the environment (.env locally) or from
    Streamlit's secrets store (used on Community Cloud, where .env files
    don't exist)."""
    value = os.getenv(key)

    if value:
        return value

    try:
        return st.secrets.get(key)
    except Exception:
        return None


SPOTIFY_CLIENT_ID = get_secret("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = get_secret("SPOTIFY_CLIENT_SECRET")

print(
    "[moodify debug] using SPOTIFY_CLIENT_ID ending in ..."
    + (SPOTIFY_CLIENT_ID[-4:] if SPOTIFY_CLIENT_ID else "MISSING")
)

MAX_CHECKS = 30  # candidates to check when no era filter is set
MAX_CHECKS_WITH_ERA = 60  # a specific decade narrows matches a lot, so check more


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


@st.cache_data(ttl=3600, show_spinner=False)
def get_track_info(track_id, token):
    """Look up a single track. Cached so that the same track isn't looked
    up again for the rest of the hour — this is what actually keeps the
    number of Spotify requests down, without depending on the multi-id
    batch endpoint (which returned 403 for this app)."""

    response = requests.get(
        f"https://api.spotify.com/v1/tracks/{track_id}",
        headers={"Authorization": f"Bearer {token}"},
        params={"market": "SE"},
        timeout=10
    )

    response.raise_for_status()

    return response.json()


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(
        os.path.join(BASE_DIR, "models", "xgboost_model.json")
    )
    return model


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


@st.cache_data
def load_data_with_predictions():
    """Load the dataset and run model predictions once per app instance
    instead of on every single Streamlit rerun (previously this ran on
    every widget interaction, not just on submit)."""
    df = pd.read_csv(
        os.path.join(
            BASE_DIR,
            "data",
            "raw",
            "278k_labelled_uri.csv"
        )
    )

    df["predicted_label"] = load_model().predict(df[feature_columns])

    return df


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

df = load_data_with_predictions()

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


class SpotifyRateLimited(Exception):
    """Raised (never cached) when Spotify returns 429 mid-search, so a
    rate-limited moment never gets baked into the cached result below."""

    def __init__(self, retry_after):
        self.retry_after = retry_after
        super().__init__(f"Spotify rate limited, retry after {retry_after}s")


@st.cache_data(ttl=3600, show_spinner="Finding recommendations...")
def find_recommendations(label, year_range):
    """Find up to 5 recommended tracks for a given mood label + era, and
    cache the whole result for an hour.

    This is the main defense against rate limiting when the app is shared:
    without it, every single click on "Recommend songs" — even for a mood/
    era combo someone already searched a minute ago — would draw a fresh
    random sample and re-check it against Spotify from scratch. With only
    ~28 possible mood/era combinations total, caching the full result per
    combination means Spotify only gets hit once per combination per hour,
    no matter how many people click around in the meantime.
    """
    max_checks = MAX_CHECKS_WITH_ERA if year_range is not None else MAX_CHECKS

    matching_songs = df[
        (df["predicted_label"] == label) &
        (df["speechiness"] < 0.66) &
        (df["duration (ms)"] < 600000)
    ]

    sample_size = min(len(matching_songs), max_checks)
    candidates = matching_songs.sample(n=sample_size) if sample_size > 0 else matching_songs
    candidate_ids = candidates["uri"].str.split(":").str[-1].tolist()

    token = get_spotify_token()

    valid_tracks = []
    checked = 0
    not_playable_count = 0
    no_release_date_count = 0
    year_mismatch_count = 0

    for track_id in candidate_ids:
        if checked >= max_checks or len(valid_tracks) >= 5:
            break

        checked += 1

        try:
            track = get_track_info(track_id, token)
        except requests.HTTPError as track_err:
            if track_err.response is not None and track_err.response.status_code == 429:
                retry_after = track_err.response.headers.get("Retry-After", "unknown")
                print(
                    f"[moodify debug] rate limited after {checked} checks "
                    f"(Retry-After: {retry_after}s), stopping early"
                )
                raise SpotifyRateLimited(retry_after) from track_err
            raise

        if not track.get("is_playable", True):
            not_playable_count += 1
            continue

        release_date = track.get("album", {}).get("release_date")

        if not release_date:
            no_release_date_count += 1
            continue

        year = int(release_date[:4])

        if year_range is not None:
            start_year, end_year = year_range

            if not start_year <= year <= end_year:
                year_mismatch_count += 1
                continue

        valid_tracks.append({
            "track_id": track_id,
            "artist": ", ".join(artist["name"] for artist in track["artists"]),
            "title": track["name"]
        })

    debug_info = {
        "checked": checked,
        "not_playable": not_playable_count,
        "no_release_date": no_release_date_count,
        "year_mismatch": year_mismatch_count,
        "valid_found": len(valid_tracks)
    }

    return valid_tracks, debug_info


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
    year_range = era_map[era]

    try:
        valid_tracks, debug_info = find_recommendations(label, year_range)

        st.session_state.recommendations = valid_tracks
        st.session_state.recommended_mood = mood
        st.session_state.debug_info = debug_info

    except SpotifyRateLimited as e:
        st.warning(
            f"Spotify is temporarily limiting requests. Please try again "
            f"in about {e.retry_after} seconds."
        )

    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        body = e.response.text[:500] if e.response is not None else str(e)
        print(f"[moodify debug] Spotify HTTP error {status}: {body}")
        st.error(f"Spotify error {status}: {body}")

    except requests.RequestException:
        st.error("Could not connect to Spotify.")


if st.session_state.recommendations is not None:

    st.subheader(
        f"Songs for a {st.session_state.recommended_mood.lower()} mood"
    )

    tracks = st.session_state.recommendations

    # Note: no Spotify Web API calls happen below. Track names/artists were
    # already captured during the search above, and the players themselves
    # are loaded client-side by the visitor's browser via Spotify's embed
    # widget — so just interacting with the page (without hitting
    # "Recommend songs" again) no longer burns any of the app's API quota.
    if len(tracks) > 0:
        if len(tracks) < 5:
            st.info(
                f"Found {len(tracks)} matching track(s) for this mood/era "
                "combination — narrower decades naturally have fewer "
                "matches in the dataset. Try \"Any\" era for more results."
            )

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
        debug_info = st.session_state.get("debug_info") or {}
        checked_count = debug_info.get("checked", 0)
        mismatch_count = debug_info.get("year_mismatch", 0)

        if checked_count > 0 and mismatch_count / checked_count >= 0.8:
            st.warning(
                f"No {st.session_state.recommended_mood.lower()} tracks from "
                f"the {era} era turned up — this dataset looks like it has "
                "very few (or none) for that combination. Try a different "
                "era, or \"Any\"."
            )
        else:
            st.warning("No Spotify previews available.")

        if st.session_state.get("debug_info"):
            with st.expander("Debug info (why no tracks were found)"):
                st.json(st.session_state.debug_info)
                st.caption(
                    "Check the terminal running 'streamlit run' for full "
                    "[moodify debug] log lines."
                )
