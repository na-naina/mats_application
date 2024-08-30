# %%
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from collections import Counter
import re
import numpy as np
from tqdm.auto import tqdm

# %%

target_words = ["literally", "obviously", "practically", "totally", "essentially"]


# %%
def count_words(text):
    if isinstance(text, str):
        words = re.findall(r'\b\w+\b', text.lower())
        return Counter(words)
    return Counter()

data_dir = 'data'
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
word_counts = {}

for file in csv_files:
    print(f"Processing {file}...")
    try:
        df = pd.read_csv(file)
        
        required_columns = ['creation_date', 'Body']
        if not all(col in df.columns for col in required_columns):
            print(f"Skipping {file} due to missing columns")
            continue
        
        # Convert 'creation_date' to datetime and extract year
        df['year'] = pd.to_datetime(df['creation_date'], errors='coerce').dt.year
        
        # Filter out rows with NaT years and ensure 'Body' is string
        df = df.dropna(subset=['year', 'Body'])
        df['Body'] = df['Body'].astype(str)
        
        # Group by year and concatenate all text
        yearly_text = df.groupby('year')['Body'].apply(' '.join)
        
        # Count words for each year
        for year, text in yearly_text.items():
            if year not in word_counts:
                word_counts[year] = Counter()
            word_counts[year] += count_words(text)
        
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

# Get the top 5 words across all years
all_words = Counter()
for year_counts in word_counts.values():
    all_words += year_counts
top_5_words = [word for word, _ in all_words.most_common(5)]

# Calculate total word counts for each year
total_word_counts = {year: sum(counts.values()) for year, counts in word_counts.items()}

# Create a DataFrame with the target words and their percentages
df_percentage = pd.DataFrame({
    word: [word_counts.get(year, Counter()).get(word, 0) / total_word_counts[year] * 100 
           if total_word_counts[year] > 0 else 0
           for year in sorted(word_counts.keys())]
    for word in target_words
}, index=sorted(word_counts.keys()))

# Convert to numpy arrays for plotting
years = np.array(df_percentage.index.astype(int))

plt.figure(figsize=(12, 6))
for word in target_words:
    percentages = np.array(df_percentage[word].values)
    plt.plot(years, percentages, label=word, marker='o')

plt.title('Usage Trends of Selected Words in Reddit Comments')
plt.xlabel('Year')
plt.ylabel('Percentage of Usage (Relative to All Words)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('selected_words_usage_trends.png')
print("Selected words usage trends have been plotted and saved as 'selected_words_usage_trends.png'")

# Display the data
print(df_percentage)

print("\nSelected words occurrence summary:")
for word in target_words:
    total_count = sum(word_counts[year].get(word, 0) for year in word_counts)
    print(f"'{word}': {total_count} occurrences")

# Calculate and print the total word count across all years
total_words = sum(total_word_counts.values())
print(f"\nTotal word count across all years: {total_words}")


# %%

# Load the Twitter dataset
twitter_df = pd.read_csv('twitter_data/twitter_dataset.csv')

# Convert Timestamp to datetime
twitter_df['Timestamp'] = pd.to_datetime(twitter_df['Timestamp'])

# Group by month and count words
word_counts = {}
for name, group in twitter_df.groupby(twitter_df['Timestamp'].dt.to_period('M')):
    texts = ' '.join(group['Text'])
    word_counts[name] = count_words(texts)

# Calculate total word counts for each month
total_word_counts = {month: sum(counts.values()) for month, counts in word_counts.items()}

# Create a DataFrame with the target words and their percentages
twitter_df_percentage = pd.DataFrame({
    word: [word_counts.get(month, Counter()).get(word, 0) / total_word_counts[month] * 100 
           if total_word_counts[month] > 0 else 0
           for month in sorted(word_counts.keys())]
    for word in target_words
}, index=sorted(word_counts.keys()))

print(twitter_df_percentage)

# Convert to numpy arrays for plotting
months = np.array(twitter_df_percentage.index.astype(str))

plt.figure(figsize=(12, 6))
for word in target_words:
    percentages = np.array(twitter_df_percentage[word].values)
    plt.plot(months, percentages, label=word, marker='o')

plt.title('Usage Trends of Selected Words in Twitter Data')
plt.xlabel('Date')
plt.ylabel('Percentage of Usage (Relative to All Words)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('twitter_selected_words_usage_trends.png')
print("Selected words usage trends have been plotted and saved as 'twitter_selected_words_usage_trends.png'")

# Display the data
print(twitter_df_percentage)

print("\nSelected words occurrence summary:")
for word in target_words:
    total_count = sum(counts.get(word, 0) for counts in word_counts.values())
    print(f"'{word}': {total_count} occurrences")

# Calculate and print the total word count across all tweets
total_words = sum(total_word_counts.values())
print(f"\nTotal word count across all tweets: {total_words}")

# %%

def count_words(text):
    if isinstance(text, str):
        words = re.findall(r'\b\w+\b', text.lower())
        return Counter(words)
    return Counter()

# Load the Reddit dataset
print("Loading the Reddit dataset...")
reddit_df = pd.read_csv('reddit_data/ten-million-reddit-answers.csv', usecols=['created_utc', 'body'])

# Convert created_utc to datetime
reddit_df['created_utc'] = pd.to_datetime(reddit_df['created_utc'], unit='s')

# Function to process chunks of data
def process_chunk(chunk):
    chunk['created_utc'] = pd.to_datetime(chunk['created_utc'], unit='s')
    monthly_counts = {}
    for name, group in chunk.groupby(chunk['created_utc'].dt.month):
        texts = ' '.join(group['body'].dropna())
        counts = count_words(texts)
        monthly_counts[name] = {word: counts[word] for word in target_words}
        monthly_counts[name]['total'] = sum(counts.values())
    return monthly_counts

# Initialize variables to store results
word_counts = {}
total_word_counts = {}

# Process the data in chunks
print("Processing data in chunks...")
chunksize = 100000  # Adjust this value based on your available memory
for chunk in tqdm(pd.read_csv('reddit_data/ten-million-reddit-answers.csv', 
                         usecols=['created_utc', 'body'], 
                         chunksize=chunksize)):
    chunk_counts = process_chunk(chunk)
    for month, counts in chunk_counts.items():
        if month not in word_counts:
            word_counts[month] = Counter()
            total_word_counts[month] = 0
        word_counts[month].update(counts)
        total_word_counts[month] += counts['total']

# Calculate percentages
print("Calculating percentages...")
df_percentage = pd.DataFrame({
    word: [word_counts[month][word] / total_word_counts[month] * 100 
           if total_word_counts[year] > 0 else 0
           for year in sorted(word_counts.keys())]
    for word in target_words
}, index=sorted(word_counts.keys()))

# Convert to numpy arrays for plotting
years = np.array(df_percentage.index.astype(int))

plt.figure(figsize=(12, 6))
for word in target_words:
    percentages = np.array(df_percentage[word].values)
    plt.plot(years, percentages, label=word, marker='o')

plt.title('Usage Trends of Selected Words in Reddit Comments')
plt.xlabel('Month')
plt.ylabel('Percentage of Usage (Relative to All Words)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('reddit_selected_words_usage_trends.png')
print("Selected words usage trends have been plotted and saved as 'reddit_selected_words_usage_trends.png'")

# Display the data
print("\nPercentage of word usage by year:")
print(df_percentage)

print("\nSelected words occurrence summary:")
for word in target_words:
    total_count = sum(word_counts[year][word] for year in word_counts)
    print(f"'{word}': {total_count} occurrences")

# Calculate and print the total word count across all comments
total_words = sum(total_word_counts.values())
print(f"\nTotal word count across all Reddit comments: {total_words}")

# %%

# Load the Top Posts data
posts_df = pd.read_csv('new_reddit_data/Top_Posts.csv')
posts_df['date-time'] = pd.to_datetime(posts_df['date-time'])

# Load the Comments data
print("Loading the Comments data...")
comments_df = pd.read_csv('new_reddit_data/Top_Posts_Comments.csv')

# Merge the datasets
merged_df = pd.merge(comments_df, posts_df[['post_id', 'date-time']], on='post_id', how='left')

# Initialize variables to store results
word_counts = {}
total_word_counts = {}

# Process the data
print("Processing data...")
for year, group in merged_df.groupby(merged_df['date-time'].dt.year):
    texts = ' '.join(group['comment'].dropna())
    counts = count_words(texts)
    word_counts[year] = {word: counts[word] for word in target_words}
    total_word_counts[year] = sum(counts.values())

# Calculate percentages
print("Calculating percentages...")
df_percentage = pd.DataFrame({
    word: [word_counts[year][word] / total_word_counts[year] * 100 
           if total_word_counts[year] > 0 else 0
           for year in sorted(word_counts.keys())]
    for word in target_words
}, index=sorted(word_counts.keys()))

# Convert to numpy arrays for plotting
years = np.array(df_percentage.index.astype(int))

plt.figure(figsize=(12, 6))
for word in target_words:
    percentages = np.array(df_percentage[word].values)
    plt.plot(years, percentages, label=word, marker='o')

plt.title('Usage Trends of Selected Words in Reddit Comments')
plt.xlabel('Year')
plt.ylabel('Percentage of Usage (Relative to All Words)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('new_reddit_selected_words_usage_trends.png')
print("Selected words usage trends have been plotted and saved as 'new_reddit_selected_words_usage_trends.png'")

# Display the data
print("\nPercentage of word usage by year:")
print(df_percentage)

print("\nSelected words occurrence summary:")
for word in target_words:
    total_count = sum(word_counts[year][word] for year in word_counts)
    print(f"'{word}': {total_count} occurrences")

# Calculate and print the total word count across all comments
total_words = sum(total_word_counts.values())
print(f"\nTotal word count across all Reddit comments: {total_words}")