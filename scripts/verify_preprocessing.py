import pickle, json, math
from pathlib import Path
from collections import Counter

PROCESSED_PATH = Path("data/processed/patient_records.pkl")


def verify():
    print("═" * 60)
    print("PREPROCESSING VERIFICATION")
    print("═" * 60)

    if not PROCESSED_PATH.exists():
        print("ERROR: patient_records.pkl not found. Run pipeline.py first.")
        return

    with open(PROCESSED_PATH, "rb") as f:
        records = pickle.load(f)

    print(f"Total records:        {len(records)}")

    outcomes = Counter(r.outcome_binary for r in records)
    print(f"Discharged (0):       {outcomes[0]} ({outcomes[0]/len(records):.1%})")
    print(f"Admitted   (1):       {outcomes[1]} ({outcomes[1]/len(records):.1%})")

    n_windows = [len(r.time_windows) for r in records]
    print(f"Avg windows/patient:  {sum(n_windows)/len(n_windows):.1f}")
    print(f"Max windows:          {max(n_windows)}")
    print(f"Min windows:          {min(n_windows)}")

    acuities = Counter(r.acuity for r in records)
    print(f"Acuity distribution:  {dict(sorted(acuities.items()))}")

    n_sepsis = sum(r.sepsis_flag for r in records)
    print(f"Sepsis flagged:       {n_sepsis} ({n_sepsis/len(records):.1%})")

    from src.utils.schema import VITAL_FEATURES

    feature_density = {f: 0.0 for f in VITAL_FEATURES}
    total_windows = 0
    for r in records:
        for w in r.time_windows:
            for f in VITAL_FEATURES:
                feature_density[f] += w.missingness_mask.get(f, 0)
            total_windows += 1
    print(f"\nFeature observation rates across {total_windows} windows:")
    for f, count in feature_density.items():
        rate = count / total_windows
        flag = " ⚠" if rate < 0.5 else ""
        print(f"  {f:15s}: {rate:.1%}{flag}")

    nan_triage = sum(
        1
        for r in records
        if any(math.isnan(v) for v in r.triage_vitals.values() if isinstance(v, float))
    )
    print(
        f"\nRecords with NaN triage vitals: {nan_triage} ({nan_triage/len(records):.1%})"
    )
    print("(NaNs in triage are expected — handled by context layer)")

    print("\n── Sample Record ──────────────────────────────")
    r = records[0]
    print(f"  stay_id:       {r.stay_id}")
    print(f"  acuity:        {r.acuity}")
    print(f"  outcome:       {r.outcome_label} ({r.outcome_binary})")
    print(f"  n_windows:     {len(r.time_windows)}")
    print(f"  sepsis_flag:   {r.sepsis_flag}")
    print(f"  n_meds:        {r.n_medications}")
    w0 = r.time_windows[0]
    print(f"  Window 0 density: {w0.observation_density:.2f}")
    print(f"  Window 0 mask:    {w0.missingness_mask}")
    print("═" * 60)
    print("VERIFICATION COMPLETE — Ready for Phase 3 (Context Construction)")
    print("═" * 60)


if __name__ == "__main__":
    verify()
