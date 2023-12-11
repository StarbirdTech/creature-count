from pytube import YouTube

# Enter the YouTube video URL
video_url = "https://www.youtube.com/watch?v=I7dYd-Ra8bk"

# Create a YouTube object
yt = YouTube(video_url)

# Get the highest resolution video stream
stream = yt.streams.get_highest_resolution()

# Download the video
stream.download()
