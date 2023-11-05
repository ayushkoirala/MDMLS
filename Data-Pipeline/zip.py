import zipfile

def unzip_folder(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

zip_path = '/home/akoirala/Thesis/Data-Pipeline/Random_Dataset/Random_with_avg_token.zip'
destination = '/home/akoirala/Thesis/Data-Pipeline/Random_Dataset/'
unzip_folder(zip_path, destination)
