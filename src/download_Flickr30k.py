import kagglehub

# Download latest version
path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")

print("Path to dataset files:", path)