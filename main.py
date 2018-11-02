from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
files = api.competition_download_files("twosigmanews")
print(files)
