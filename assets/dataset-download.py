# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("odins0n/ucf-crime-dataset")
#
# print("Path to dataset files:", path)

from supervision.assets import download_assets, VideoAssets

# Downloading pre-defined video datasets
download_assets(VideoAssets.MARKET_SQUARE)
download_assets(VideoAssets.GROCERY_STORE)
download_assets(VideoAssets.SUBWAY)

