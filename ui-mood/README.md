# Mood-Based UI (`ui-mood/`)

A consumer-facing variant of the Streamlit app in `ui/`, built on the exact
same trained model and preprocessing pipeline. Where `ui/app.py` is a
technical demo (feature sliders, batch CSV scoring, raw probabilities) aimed
at showing the model working, this app is aimed at an end user who doesn't
know or care what "danceability" means.

## Why two UI folders?

They serve two different audiences on purpose, not by accident:

| | `ui/` | `ui-mood/` |
|---|---|---|
| Audience | Technical reviewer | End user |
| Input | 15 audio feature sliders | Free-text mood description or a quick-mood button |
| Output | Genre label + full probability table | A styled genre "vibe" card + Spotify playlist recommendations |
| Purpose | Prove the model works end-to-end | Show the model can sit behind an actual product |

## Running the App

From the project root:

```bash
streamlit run ui-mood/app.py
```

## How it works

1. The user types a mood ("chill", "need to focus", "hype me up") or picks a
   quick-mood button.
2. Mood keywords are mapped to one of the 6 trained genre categories.
3. The app displays a styled result card (emoji + colour per genre) and a
   short list of matching Spotify playlist searches.
4. A secondary "Audio features" tab exposes the same slider-based prediction
   as `ui/app.py`, for anyone who wants to see the underlying model directly.

## Notes

- Loads the same `models/final_model.pkl`, `models/preprocessor.pkl`, and
  `models/label_encoder.pkl` as `ui/app.py` — there is only one trained model,
  shared by both interfaces.
- Playlist links are static Spotify search queries, not live API calls — no
  Spotify credentials are required to run this app.
