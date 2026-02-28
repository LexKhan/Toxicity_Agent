import pandas as pd
import re
import os
from html import unescape

class ToxicCommentPreprocessor:
    def __init__(self, input_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"'{input_path}' not found.\n"
                "Download the Jigsaw dataset from Kaggle and place train.csv in data/."
            )
        self.df = pd.read_csv(input_path)

        label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        for col in label_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0).astype(int)

        print(f" Loaded {len(self.df)} comments")

    def preprocess(self):
        self._basic_cleaning()          # ← fixed
        self._create_classification()   # ← fixed
        self._balance_dataset()         # ← fixed
        self._generate_explanations()   # ← fixed
        self._generate_messages()       # ← fixed
        self._validate()                # ← fixed
        return self.df

    def _basic_cleaning(self):
        before = len(self.df)
        self.df = self.df.dropna(subset=["comment_text"])
        self.df["comment_text"] = self.df["comment_text"].apply(self._clean_text)
        self.df = self.df[self.df["comment_text"].str.len() >= 10]
        self.df = self.df.drop_duplicates(subset=["comment_text"])
        self.df = self.df.reset_index(drop=True)
        print(f"   Removed {before - len(self.df)} rows  |  Remaining: {len(self.df)}")

    @staticmethod
    def _clean_text(text: str) -> str:
        text = str(text)
        text = unescape(text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\[\[.*?\]\]", "", text)
        text = re.sub(r"{{.*?}}", "", text)
        text = re.sub(r"([!?.]){4,}", r"\1\1\1", text)
        text = text.strip('"').strip("'").strip()
        return text

    def _create_classification(self):
        def classify(row) -> str:
            if row["severe_toxic"] or row["threat"] or row["identity_hate"]:
                return "TOXIC"
            if row["insult"] or row["obscene"] or row["toxic"]:
                return "TOXIC"
            good_words = [
                "thank", "great", "good", "awesome", "excellent",
                "appreciate", "congratulations", "well done", "amazing",
                "helpful", "wonderful", "fantastic", "brilliant",
            ]
            if any(w in row["comment_text"].lower() for w in good_words):
                return "GOOD"
            return "NEUTRAL"

        self.df["classification"] = self.df.apply(classify, axis=1)
        print("\n   Distribution:")
        for label, count in self.df["classification"].value_counts().items():
            print(f"   {label}: {count:,}")

    def _balance_dataset(self, samples_per_class: int = 1000):
        parts = []
        for cls in ("TOXIC", "NEUTRAL", "GOOD"):
            subset = self.df[self.df["classification"] == cls]
            n = min(samples_per_class, len(subset))
            if n == 0:
                print(f"   ⚠ No examples for class {cls} — skipping")
                continue
            parts.append(subset.sample(n=n, random_state=42))

        if not parts:
            raise ValueError("All classes empty after balancing. Check your dataset.")

        self.df = (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        print(f"   Balanced total: {len(self.df)}")

    def _generate_explanations(self):
        print("\n[4/6] Generating Explanations...")

        def explain(row) -> str:
            reasons = []
            if row["severe_toxic"]:  reasons.append("severely toxic language")
            if row["threat"]:        reasons.append("threatening content")
            if row["identity_hate"]: reasons.append("identity-based hate speech")
            if row["insult"]:        reasons.append("insulting language")
            if row["obscene"]:       reasons.append("obscene content")
            if row["toxic"] and not reasons:
                reasons.append("toxic language")
            if reasons:
                return f"Contains {', '.join(reasons)}."
            if row["classification"] == "GOOD":
                return "Positive and constructive communication."
            return "Neutral, factual communication without toxicity."

        self.df["explanation"] = self.df.apply(explain, axis=1)
        print("   ✓ Done")   # ← added

    def _generate_messages(self):
        print("\n[5/6] Generating Author Messages …")

        def message(row) -> str:
            if row["classification"] != "TOXIC":
                return "N/A"
            if row["severe_toxic"]:
                return "This content violates community guidelines and may cause serious harm. Please reconsider your language."
            if row["threat"]:
                return "Threatening language is not acceptable and may have legal consequences. Please express concerns without threats."
            if row["identity_hate"]:
                return "This content contains discriminatory language. Please treat all individuals with respect regardless of their background."
            if row["insult"]:
                return "Personal attacks are not constructive. Please focus on ideas rather than attacking individuals."
            if row["obscene"]:
                return "Please keep language appropriate and respectful for all audiences."
            return "This comment may be perceived as harmful. Consider rephrasing more constructively."

        self.df["message_to_author"] = self.df.apply(message, axis=1)
        print("   ✓ Done")

    def _validate(self):
        print("\n[6/6] Validating …")
        required = ["comment_text", "classification", "explanation", "message_to_author"]
        null_counts = self.df[required].isnull().sum()

        if null_counts.sum() == 0:
            print("   ✓ No missing values")
        else:
            print("   Filling missing values:")
            print(null_counts[null_counts > 0])
            self.df["explanation"]       = self.df["explanation"].fillna("No explanation available.")
            self.df["message_to_author"] = self.df["message_to_author"].fillna("N/A")

        lengths = self.df["comment_text"].str.len()
        print(f"   Text length — min:{lengths.min()}  max:{lengths.max()}  avg:{lengths.mean():.0f}")

    def save(self, output_path: str = "data/toxicity_examples.csv") -> pd.DataFrame:
        print(f"\n SAVING PROCESSED DATA")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        out = self.df[["classification", "comment_text", "explanation", "message_to_author"]].copy()
        out.columns = ["classification", "content", "explanation", "message_to_author"]
        out.to_csv(output_path, index=False, quoting=1)
        print(f" Saved {len(out):,} rows → {output_path}")

        print("\n Sample per class:")
        for cls in ("TOXIC", "NEUTRAL", "GOOD"):
            row = out[out["classification"] == cls]
            if len(row):
                s = row.iloc[0]
                print(f"\n  {cls}:")
                print(f"    Content:     {s['content'][:80]} …")
                print(f"    Explanation: {s['explanation']}")

        return out   # ← fixed


if __name__ == "__main__":
    print("\n Starting Data Preprocessing …\n")
    os.makedirs("data", exist_ok=True)
    preprocessor = ToxicCommentPreprocessor("data/train.csv")
    preprocessor.preprocess()
    preprocessor.save()
    print("\n Preprocessing Complete!\n")