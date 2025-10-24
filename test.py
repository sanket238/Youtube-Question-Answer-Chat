from langchain_community.document_loaders import YoutubeLoader

url = "https://www.youtube.com/watch?v=-HzgcbRXUK8&t=248s"
loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
docs = loader.load()

print(docs[0].page_content)