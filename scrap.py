import os
import csv
from googleapiclient.discovery import build

# Replace 'YOUR_API_KEY' with your actual API key
API_KEY = 'AIzaSyCCGlXOkctAd6FJaqad7yaDvxPrtG8Z2D4'

# Initialize the YouTube Data API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Replace 'YOUR_VIDEO_ID' with the video ID you want to extract comments from
video_id = 'JfYeqhTCrII'
# cD4ntiKp0mk
# JfYeqhTCrII
# TuvjYqPvebg 
# video_id = 'wLegCFsOi_E' 
try:
    # Retrieve video comments
    comments = []
    nextPageToken = None
    while True:
        results = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            pageToken=nextPageToken
        ).execute(
)
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
            comments.append({
                'comment': comment,
                'like_count': like_count
            })

        # Check if there are more comments to fetch
        nextPageToken = results.get('nextPageToken')

        if not nextPageToken:
            break

    # Save comments to a CSV file
    with open('youtube_comments.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['comment', 'like_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        # Write comment data
        for comment_info in comments:
            writer.writerow(comment_info)
    print('Comments saved to youtube_comments.csv')

except Exception as e:
    print('An error occurred:', str(e))




