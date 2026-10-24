import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv("data/processed/metadata.csv")
    
    print("Dataset Summary:")
    print(df['source_dataset'].value_counts())
    print("\n")
    
    # Filter for wikiart emotions
    mood_df = df[df['source_dataset'] == 'wikiart_emotions'].copy()
    
    print(f"Total mood annotations: {len(mood_df)}")
    print("\nMood class distribution:")
    print(mood_df['mood'].value_counts())
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=mood_df, x='mood')
    plt.title("Distribution of Majority Mood Labels")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/processed/mood_distribution.png")
    print("\nSaved mood distribution plot to data/processed/mood_distribution.png")
    
    # Check agreement levels
    print("\nAgreement stats:")
    print(mood_df['mood_agreement'].describe())
    
if __name__ == "__main__":
    main()
