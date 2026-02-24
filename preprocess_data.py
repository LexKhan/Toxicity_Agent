import pandas as pd
import re
from html import unescape

class ToxicCommentPreprocessor:
    def __init__(self, input_path):
        """Initialize with Kaggle toxic comment dataset"""
        self.df = pd.read_csv(input_path)
        print(f"✅ Loaded {len(self.df)} comments")
    
    def preprocess(self):
        """Clean and prepare the dataset"""
        print("\n" + "="*60)
        print("PREPROCESSING KAGGLE TOXIC COMMENT DATASET")
        print("="*60)
        
        self.basic_cleaning()
        self.create_classification()
        self.balance_dataset()
        self.generate_explanations()
        self.generate_messages()
        self.validate()
        
        return self.df
    
    def basic_cleaning(self):
        """Remove problematic content"""
        print("\n[1/6] Basic Cleaning...")
        initial_count = len(self.df)

        self.df = self.df.dropna(subset=['comment_text'])
        self.df['comment_text'] = self.df['comment_text'].apply(self.clean_text)
        self.df = self.df[self.df['comment_text'].str.len() >= 10]
        self.df = self.df.drop_duplicates(subset=['comment_text'])
        
        print(f"   Removed {initial_count - len(self.df)} problematic rows")
        print(f"   Remaining: {len(self.df)} comments")
    
    def clean_text(self, text):
        """Clean individual comment"""
        text = str(text)
        text = unescape(text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'{{.*?}}', '', text)
        text = re.sub(r'([!?.]){4,}', r'\1\1\1', text)
        return text.strip()
    
    def create_classification(self):
        """Convert multi-label to TOXIC/NEUTRAL/GOOD"""
        print("\n[2/6] Creating Classifications...")
        
        def classify(row):
            # Priority-based classification
            if row['severe_toxic'] == 1 or row['threat'] == 1 or row['identity_hate'] == 1:
                return 'TOXIC'
            elif row['insult'] == 1 or row['obscene'] == 1 or row['toxic'] == 1:
                return 'TOXIC'
            else:
                # Check for positive indicators
                text_lower = row['comment_text'].lower()
                good_words = ['thank', 'great', 'good', 'awesome', 'excellent', 
                             'appreciate', 'congratulations', 'well done', 'hero']
                
                if any(word in text_lower for word in good_words):
                    return 'GOOD'
                else:
                    return 'NEUTRAL'
        
        self.df['classification'] = self.df.apply(classify, axis=1)
        
        print("\n   Distribution:")
        for label, count in self.df['classification'].value_counts().items():
            print(f"   {label}: {count}")
    
    def balance_dataset(self, samples_per_class=1000):
        """Balance to avoid bias"""
        print(f"\n[3/6] Balancing Dataset ({samples_per_class} per class)...")
        
        toxic = self.df[self.df['classification'] == 'TOXIC'].sample(
            n=min(samples_per_class, len(self.df[self.df['classification'] == 'TOXIC'])),
            random_state=42
        )
        
        neutral = self.df[self.df['classification'] == 'NEUTRAL'].sample(
            n=min(samples_per_class, len(self.df[self.df['classification'] == 'NEUTRAL'])),
            random_state=42
        )
        
        good = self.df[self.df['classification'] == 'GOOD'].sample(
            n=min(samples_per_class, len(self.df[self.df['classification'] == 'GOOD'])),
            random_state=42
        )
        
        self.df = pd.concat([toxic, neutral, good], ignore_index=True)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   Balanced to {len(self.df)} total comments")
    
    def generate_explanations(self):
        """Generate explanation based on toxicity flags"""
        print("\n[4/6] Generating Explanations...")
        
        def create_explanation(row):
            reasons = []
            
            if row['severe_toxic'] == 1:
                reasons.append("severely toxic language")
            if row['threat'] == 1:
                reasons.append("threatening content")
            if row['identity_hate'] == 1:
                reasons.append("identity-based hate speech")
            if row['insult'] == 1:
                reasons.append("insulting language")
            if row['obscene'] == 1:
                reasons.append("obscene content")
            if row['toxic'] == 1 and not reasons:
                reasons.append("toxic language")
            
            if reasons:
                return f"Contains {', '.join(reasons)}."
            elif row['classification'] == 'GOOD':
                return "Positive and constructive communication."
            else:
                return "Neutral, factual communication without toxicity."
        
        self.df['explanation'] = self.df.apply(create_explanation, axis=1)
        print("   ✓ Explanations generated")
    
    def generate_messages(self):
        """Generate feedback messages for authors"""
        print("\n[5/6] Generating Author Messages...")
        
        def create_message(row):
            if row['classification'] == 'TOXIC':
                if row['severe_toxic'] == 1:
                    return "This content violates community guidelines and may cause serious harm. Please reconsider your language and approach."
                elif row['threat'] == 1:
                    return "Threatening language is not acceptable and may have legal consequences. Please express concerns without threats."
                elif row['identity_hate'] == 1:
                    return "This content contains discriminatory language targeting identity. Please treat all individuals with respect regardless of their background."
                elif row['insult'] == 1:
                    return "Personal attacks are not constructive. Please focus on discussing ideas rather than attacking individuals."
                elif row['obscene'] == 1:
                    return "Please keep language appropriate and respectful for all audiences."
                else:
                    return "This comment may be perceived as harmful or offensive. Consider rephrasing more constructively."
            else:
                return "N/A"
        
        self.df['message_to_author'] = self.df.apply(create_message, axis=1)
        print("   ✓ Author messages generated")
    
    def validate(self):
        """Final validation"""
        print("\n[6/6] Validating Data...")
        
        required_cols = ['comment_text', 'classification', 'explanation', 'message_to_author']
        missing = self.df[required_cols].isnull().sum()
        
        if missing.sum() == 0:
            print("   ✓ No missing values")
        else:
            print("   ⚠ Missing values found:")
            print(missing[missing > 0])
        
        print(f"\n   Text length stats:")
        print(f"   Min: {self.df['comment_text'].str.len().min()} chars")
        print(f"   Max: {self.df['comment_text'].str.len().max()} chars")
        print(f"   Avg: {self.df['comment_text'].str.len().mean():.0f} chars")
    
    def save(self, output_path='data/toxicity_examples.csv'):
        """Save processed data"""
        print(f"\n{'='*60}")
        print("SAVING PROCESSED DATA")
        print("="*60)
        
        # Keep only necessary columns and rename
        output_df = self.df[['classification', 'comment_text', 'explanation', 'message_to_author']].copy()
        output_df.columns = ['classification', 'content', 'explanation', 'message_to_author']
        
        output_df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(output_df)} examples to {output_path}")
        
        # Show sample
        print("\n📋 Sample Examples:")
        for classification in ['TOXIC', 'NEUTRAL', 'GOOD']:
            sample = output_df[output_df['classification'] == classification].iloc[0]
            print(f"\n{classification}:")
            print(f"  Content: {sample['content'][:80]}...")
            print(f"  Explanation: {sample['explanation']}")
        
        return output_df


if __name__ == "__main__":
    print("\n🚀 Starting Data Preprocessing...\n")
    
    import os
    if not os.path.exists('data/train.csv'):
        print("❌ Error: train.csv not found!")
        print("Please download the Kaggle Toxic Comment dataset and place train.csv in the data/ directory.")
        exit(1)
    
    preprocessor = ToxicCommentPreprocessor('data/train.csv')
    cleaned_df = preprocessor.preprocess()
    output_df = preprocessor.save()
    
    print("\n✅ Preprocessing Complete!\n")