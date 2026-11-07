from pathlib import Path
import joblib

MODEL = Path(__file__).resolve().parent / "fake_news_model.joblib"

def main():
    if not MODEL.exists():
        print("Model not found. Run: python train_model.py")
        return
    model = joblib.load(MODEL)
    print("AI-Based Fake News Detection — educational demo")
    print("This model cannot verify facts. Check trustworthy sources.")
    while True:
        text = input("\nEnter news text (or type exit): ").strip()
        if text.lower() == "exit":
            break
        if not text:
            print("Enter some text.")
            continue
        label = model.predict([text])[0]
        probs = model.predict_proba([text])[0]
        confidence = max(probs) * 100
        print("Model label:", label)
        print(f"Model confidence: {confidence:.1f}% (not a truth probability)")

if __name__ == "__main__":
    main()
