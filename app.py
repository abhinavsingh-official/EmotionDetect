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
h1 { font-weight: 700; letter-spacing: -0.02em; }
.ed-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.ed-feature-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 1.3rem 1.4rem; height: 100%;
}
.ed-feature-card h4 { margin-top: 0; }
.ed-feature-card ul { margin: 0.4rem 0 0 0; padding-left: 1.2rem; color: #c9d1d9; }
.ed-feature-card li { margin-bottom: 0.35rem; }
.ed-stat { font-size: 2rem; font-weight: 700; color: #ff4b4b; }
.ed-stat-label { color: #8b949e; font-size: 0.9rem; }
.ed-caveat {
    background: #2d1b1b; border: 1px solid #5c2b2b; border-radius: 10px;
    padding: 0.9rem 1.1rem; color: #e6b8b8; font-size: 0.92rem;
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
        "<h2 style='margin-top:0;'>Three different answers to the same question</h2>"
        "<p style='color:#c9d1d9; font-size:1.05rem; max-width:760px; line-height:1.6;'>"
        "Say something into a microphone and this app guesses the emotion behind it three "
        "different ways: a simple nearest-centroid lookup, a classical ML model trained from "
        "scratch on hand-crafted audio features, and a large pretrained speech model. The "
        "point isn't to pick a winner and hide the rest - it's to show what each approach "
        "actually buys you, honestly, including where the comparison itself has caveats."
        "</p>"
        "<ul>"
        "<li><b>Baseline</b> - cosine similarity between your voice's features and each "
        "emotion's average feature vector. Zero training cost, the weakest signal.</li>"
        "<li><b>Classical ML</b> - a model actually trained and cross-validated on the same "
        "features, picked from several candidates by real accuracy, not by assumption.</li>"
        "<li><b>Pretrained deep model</b> - a wav2vec2 model fine-tuned specifically for "
        "speech emotion, used here purely for inference.</li>"
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

    st.markdown("#### What each tab actually does")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="ed-feature-card"><h4>Live Analysis</h4>'
            "<ul>"
            "<li>Record a few seconds of speech directly in the browser</li>"
            "<li>See all three models' top emotion guesses side by side as a bar chart, "
            "not just a single number</li>"
            "<li>Runs entirely on this machine - no audio leaves your computer</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="ed-feature-card"><h4>Model Comparison</h4>'
            "<ul>"
            "<li>The full cross-validated results table behind the numbers above</li>"
            "<li>A confusion matrix for the best classical model, so you can see exactly "
            "which emotions it confuses, not just its overall score</li>"
            "<li>A permutation-importance breakdown of which audio features the model "
            "actually relies on</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="color:#8b949e; margin-top:1.6rem;">Head to <b>Live Analysis</b> to try it '
        "yourself, or <b>Model Comparison</b> for the full numbers behind these results.</p>",
        unsafe_allow_html=True,
    )


def _bar_chart(title: str, probs: dict[str, float]):
    labels = list(probs.keys())
    values = list(probs.values())
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#ff4b4b"))
    fig.update_layout(
        title=title, yaxis_range=[0, 1], template="plotly_dark",
        paper_bgcolor="#161b22", plot_bgcolor="#161b22", height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def live_analysis_tab():
    st.markdown(
        '<p style="color:#8b949e;">Record a short sentence - a few seconds is enough. Try '
        "saying the same sentence in an obviously different tone (flat vs. excited) to see "
        "how much the predictions shift.</p>",
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
            '<div class="ed-caveat">Heads up: this pretrained model was itself fine-tuned on '
            "the RAVDESS dataset, so it has almost certainly heard voices and sentences like "
            "this before. Its confidence here says more about how well it learned RAVDESS "
            "specifically than how well it generalizes to an arbitrary new voice.</div>",
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
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Reds"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        xaxis_title="Predicted", yaxis_title="Actual", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    if explainability_results:
        st.markdown(f"#### What the {explainability_results['model']} model actually looks at")
        st.markdown(
            '<p style="color:#8b949e;">Permutation importance: shuffle one feature across all '
            "samples and see how much accuracy drops. A feature the model relies on hurts a lot "
            "when scrambled; one it ignores barely moves the needle.</p>",
            unsafe_allow_html=True,
        )
        feat_df = pd.DataFrame(explainability_results["top_features"])
        st.bar_chart(feat_df.set_index("feature")["importance_mean"])
        summary = explainability_results["family_summary"]
        st.markdown(
            f'<p style="color:#8b949e;">Of the top 15 features, {summary["mfcc"]} are MFCCs, '
            f'{summary["mel"]} are Mel-spectrogram bands, and {summary["chroma"]} are chroma bins - '
            "MFCCs dominating matches the established speech-emotion literature, where MFCCs "
            "capture the spectral envelope shape tied to vocal tract and prosody.</p>",
            unsafe_allow_html=True,
        )


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("EmotionDetect")
    st.caption(
        "Speech emotion recognition, compared honestly across three different approaches - "
        "not just the one that scores highest."
    )
    tab0, tab1, tab2 = st.tabs(["Home", "Live Analysis", "Model Comparison"])
    with tab0:
        home_tab()
    with tab1:
        live_analysis_tab()
    with tab2:
        model_comparison_tab()


if __name__ == "__main__":
    main()
