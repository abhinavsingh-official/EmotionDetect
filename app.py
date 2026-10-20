"""Streamlit frontend for EmotionDetect - record a few seconds of speech
and see three different approaches to the same question side by side:
AudioEmo.py's original nearest-centroid baseline, the classical model
classifier.py found to work best, and a real pretrained wav2vec2 model
(pretrained.py), so the comparison isn't just a results table buried in a
README but something you can actually try against your own voice."""

import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from streamlit_mic_recorder import mic_recorder

from AudioEmo import AudioEmotionRecognizer

st.set_page_config(page_title="EmotionDetect", layout="wide")

STYLE = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }
h1 { font-weight: 700; letter-spacing: -0.02em; color: #43b047; }
.ed-card {
    background: #13213d; border: 1px solid #2a5ca8; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.ed-feature-card {
    background: #13213d; border: 1px solid #2a5ca8; border-radius: 12px;
    padding: 1.3rem 1.4rem; height: 100%;
}
.ed-feature-card h4 { margin-top: 0; color: #5b9bf0; }
.ed-feature-card ul { margin: 0.4rem 0 0 0; padding-left: 1.2rem; color: #c9d1d9; }
.ed-feature-card li { margin-bottom: 0.35rem; }
.ed-stat { font-size: 2rem; font-weight: 700; color: #43b047; }
.ed-stat-label { color: #8b949e; font-size: 0.9rem; }
.ed-caveat {
    background: #2d2410; border: 1px solid #6b5420; border-radius: 10px;
    padding: 0.9rem 1.1rem; color: #e6d4a8; font-size: 0.92rem;
}
</style>
"""

OUTPUTS = Path("outputs")


@st.cache_resource
def load_baseline() -> AudioEmotionRecognizer | None:
    if not Path("Audio_db.pkl").exists():
        return None
    return AudioEmotionRecognizer("Audio_db.pkl", sample_rate=16000)


@st.cache_resource
def load_classical_model():
    path = Path("models/emotion_classifier.joblib")
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_pretrained_model():
    from pretrained import PretrainedEmotionModel
    return PretrainedEmotionModel()


def load_json(name: str) -> dict | None:
    path = OUTPUTS / name
    return json.loads(path.read_text()) if path.exists() else None


def decode_recording(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
    import io
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
    return data.astype(np.float32)


def home_tab():
    classifier_results = load_json("classifier_results.json")
    pretrained_results = load_json("pretrained_results.json")

    st.markdown(
        '<div class="ed-card" style="padding:2rem 2.2rem;">'
        "<h2 style='margin-top:0;'>Three guesses at the same emotion</h2>"
        "<p style='color:#c9d1d9; font-size:1.05rem; max-width:760px; line-height:1.6;'>"
        "Say something into a mic and this app guesses the emotion behind it three different "
        "ways, shown side by side instead of picking one and hiding the rest."
        "</p>"
        "<ul>"
        "<li><b>Baseline</b> - cosine similarity to each emotion's average feature vector. "
        "No training, the weakest signal.</li>"
        "<li><b>Classical ML</b> - a model trained and cross-validated on the same features, "
        "picked by actual accuracy.</li>"
        "<li><b>Pretrained deep model</b> - a wav2vec2 model fine-tuned for speech emotion, "
        "used here just for inference.</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Where things stand right now")
    col1, col2, col3 = st.columns(3)
    with col1:
        best = classifier_results["best_model"] if classifier_results else "-"
        acc = classifier_results["results"][best]["accuracy_mean"] if classifier_results else 0
        st.markdown(
            f'<div class="ed-feature-card"><div class="ed-stat">{acc:.0%}</div>'
            f'<div class="ed-stat-label">Best classical model accuracy ({best})</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        baseline_acc = classifier_results["baseline_nearest_centroid"]["accuracy_mean"] if classifier_results else 0
        st.markdown(
            f'<div class="ed-feature-card"><div class="ed-stat">{baseline_acc:.0%}</div>'
            '<div class="ed-stat-label">Nearest-centroid baseline accuracy</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        pretrained_acc = pretrained_results["accuracy"] if pretrained_results else 0
        st.markdown(
            f'<div class="ed-feature-card"><div class="ed-stat">{pretrained_acc:.0%}</div>'
            '<div class="ed-stat-label">Pretrained wav2vec2 accuracy (caveated)</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### What each tab does")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="ed-feature-card"><h4>Live Analysis</h4>'
            "<ul>"
            "<li>Record a few seconds of speech in the browser</li>"
            "<li>See all three models' guesses as a bar chart</li>"
            "<li>Runs locally - no audio leaves your computer</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="ed-feature-card"><h4>Model Comparison</h4>'
            "<ul>"
            "<li>The full cross-validated results table</li>"
            "<li>A confusion matrix for the best model - what it confuses, not just its score</li>"
            "<li>Which audio features the model actually relies on</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="color:#8b949e; margin-top:1.6rem;">Try <b>Live Analysis</b>, or check '
        "<b>Model Comparison</b> for the full numbers.</p>",
        unsafe_allow_html=True,
    )


def _bar_chart(title: str, probs: dict[str, float]):
    top_label = max(probs, key=probs.get)
    st.markdown(
        f'<p style="margin-bottom:0.3rem;">{title}: '
        f'<span style="color:#43b047; font-weight:700;">{top_label}</span> '
        f'<span style="color:#8b949e;">({probs[top_label]:.0%})</span></p>',
        unsafe_allow_html=True,
    )
    labels = list(probs.keys())
    values = list(probs.values())
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#43b047"))
    fig.update_layout(
        yaxis_range=[0, 1], template="plotly_dark",
        paper_bgcolor="#13213d", plot_bgcolor="#13213d", height=280,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def live_analysis_tab():
    st.markdown(
        '<p style="color:#8b949e;">Record a short sentence. Try the same line in two '
        "different tones (flat vs. excited) and see how much the predictions shift.</p>",
        unsafe_allow_html=True,
    )

    audio_data = mic_recorder(
        start_prompt="Start recording", stop_prompt="Stop recording", format="wav", key="recorder"
    )

    if audio_data is None:
        st.markdown('<div class="ed-card">Press "Start recording" above to begin.</div>', unsafe_allow_html=True)
        return

    st.audio(audio_data["bytes"], format="audio/wav")

    with st.spinner("Decoding audio..."):
        audio = decode_recording(audio_data["bytes"], audio_data["sample_rate"])

    if np.max(np.abs(audio)) < 1e-4:
        st.warning("That recording looks silent - check your microphone and try again.")
        return

    baseline = load_baseline()
    classical = load_classical_model()

    col1, col2 = st.columns(2)
    with col1:
        if baseline is not None:
            with st.spinner("Running baseline (nearest-centroid)..."):
                probs = baseline.predict(audio, sr=16000)
            total = sum(probs.values()) or 1.0
            _bar_chart("Baseline (nearest-centroid)", {k: v / total for k, v in probs.items()})
        else:
            st.info("Baseline unavailable - Audio_db.pkl not found. Run Generatedb.py first.")

        if classical is not None:
            with st.spinner("Running classical model..."):
                feature_cols = [f"f{i}" for i in range(180)]
                features = _extract_feature_row(audio)
                proba = classical.predict_proba(pd.DataFrame([features], columns=feature_cols))[0]
                labels = classical.classes_
            _bar_chart("Classical model", dict(zip(labels, proba)))
        else:
            st.info("Classical model unavailable - run classifier.py first.")

    with col2:
        with st.spinner("Running pretrained wav2vec2 (first run downloads the model)..."):
            pretrained = load_pretrained_model()
            probs = pretrained.predict(audio, sr=16000)
        _bar_chart("Pretrained wav2vec2", probs)
        st.markdown(
            '<div class="ed-caveat">Heads up: this model was fine-tuned on the RAVDESS '
            "dataset, so its confidence here reflects how well it learned that dataset more "
            "than how well it generalizes to a new voice.</div>",
            unsafe_allow_html=True,
        )


def _extract_feature_row(audio: np.ndarray) -> np.ndarray:
    import librosa as lr
    mel = lr.feature.melspectrogram(y=audio, sr=16000)
    chroma = lr.feature.chroma_stft(y=audio, sr=16000)
    mfcc = lr.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    combined = np.concatenate([mel, mfcc, chroma], axis=0)
    return combined.mean(axis=1)


def model_comparison_tab():
    classifier_results = load_json("classifier_results.json")
    pretrained_results = load_json("pretrained_results.json")
    explainability_results = load_json("explainability_results.json")

    if classifier_results is None:
        st.info("No results yet - run classifier.py first.")
        return

    st.markdown("#### Accuracy by approach (5-fold cross-validation, except where noted)")
    rows = [{"Approach": "Baseline (nearest-centroid)",
              "Accuracy": classifier_results["baseline_nearest_centroid"]["accuracy_mean"],
              "Std dev": classifier_results["baseline_nearest_centroid"]["accuracy_std"]}]
    for name, scores in classifier_results["results"].items():
        rows.append({"Approach": name, "Accuracy": scores["accuracy_mean"], "Std dev": scores["accuracy_std"]})
    if pretrained_results:
        rows.append({"Approach": "Pretrained wav2vec2 (caveated - see below)",
                      "Accuracy": pretrained_results["accuracy"], "Std dev": None})
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if pretrained_results:
        st.markdown(
            f'<div class="ed-caveat">{pretrained_results["caveat"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"#### Confusion matrix - {classifier_results['best_model']} (best classical model)")
    cm = np.array(classifier_results["confusion_matrix"])
    labels = classifier_results["labels"]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Greens"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#13213d", plot_bgcolor="#13213d",
        xaxis_title="Predicted", yaxis_title="Actual", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    if explainability_results:
        st.markdown(f"#### What the {explainability_results['model']} model looks at")
        st.markdown(
            '<p style="color:#8b949e;">Shuffle one feature, see how much accuracy drops. '
            "A feature the model relies on hurts a lot when scrambled; one it ignores barely "
            "moves the needle.</p>",
            unsafe_allow_html=True,
        )
        feat_df = pd.DataFrame(explainability_results["top_features"])
        st.bar_chart(feat_df.set_index("feature")["importance_mean"])
        summary = explainability_results["family_summary"]
        st.markdown(
            f'<p style="color:#8b949e;">Of the top 15 features, {summary["mfcc"]} are MFCCs, '
            f'{summary["mel"]} are Mel bands, {summary["chroma"]} are chroma - MFCCs dominating '
            "matches what's expected in speech-emotion work.</p>",
            unsafe_allow_html=True,
        )


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("EmotionDetect")
    st.caption("Speech emotion recognition, compared across three different approaches.")
    tab0, tab1, tab2 = st.tabs(["Home", "Live Analysis", "Model Comparison"])
    with tab0:
        home_tab()
    with tab1:
        live_analysis_tab()
    with tab2:
        model_comparison_tab()


if __name__ == "__main__":
    main()
