
"""
End-to-end demo using generated synthetic data.

Run this after installing requirements to verify the full pipeline.
"""

from pathlib import Path
from judge.utils.demo import generate_demo_dataset
from judge.core.session import AnalysisSession
from judge.reporting.generator import generate_report


def main():
    print("Generating synthetic demo dataset with injected anomalies...")
    files = generate_demo_dataset("demo_data", duration_s=42)
    print("Dataset:", files)

    print("\nInitializing Judge session...")
    session = AnalysisSession(sensitivity=0.5, min_event_duration=0.06, cross_modal_window=1.6)

    session.add_file(files["video"])
    session.add_file(files["audio"])
    session.add_file(files["sensor"])

    print("Running detectors + fusion...")
    result = session.run()

    print(f"\nDetected {len(result.events)} candidate events.")
    for ev in sorted(result.events, key=lambda e: -e.score)[:8]:
        print(f"  [{ev.modality.value}] {ev.pretty_time()} score={ev.score:.2f}  {ev.description[:65]}...")

    out = Path("reports")
    out.mkdir(exist_ok=True)
    report = generate_report(result, out / "demo_report", format="all")
    print(f"\nReport generated: {report}")

    print("\nDemo finished successfully. Open the PDF or Markdown for full technical detail.")


if __name__ == "__main__":
    main()
