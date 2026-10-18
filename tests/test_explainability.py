from explainability import N_CHROMA, N_MEL, N_MFCC, family_summary, feature_family


def test_feature_family_boundaries():
    assert feature_family(0) == "mel[0]"
    assert feature_family(N_MEL - 1) == f"mel[{N_MEL - 1}]"
    assert feature_family(N_MEL) == "mfcc[0]"
    assert feature_family(N_MEL + N_MFCC - 1) == f"mfcc[{N_MFCC - 1}]"
    assert feature_family(N_MEL + N_MFCC) == "chroma[0]"
    assert feature_family(N_MEL + N_MFCC + N_CHROMA - 1) == f"chroma[{N_CHROMA - 1}]"


def test_family_summary_counts_by_prefix():
    ranked = [{"feature": "mfcc[0]"}, {"feature": "mfcc[1]"}, {"feature": "mel[3]"}, {"feature": "chroma[2]"}]
    summary = family_summary(ranked)
    assert summary == {"mel": 1, "mfcc": 2, "chroma": 1}
