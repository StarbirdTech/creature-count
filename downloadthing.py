from pytube import YouTube

# Enter the YouTube video URL
video_url = "https://www.youtube.com/watch?v=BRmnuwAEbiY"

# Create a YouTube object
yt = YouTube(video_url)

# Get the highest resolution video stream
stream = yt.streams.get_highest_resolution()

# Download the video
stream.download()

"ffmpeg -i demo44.mp4 -ss 00:00:30 -t 00:01:30 -c:v copy -c:a copy demo4.mp4"
