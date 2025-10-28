from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Authenticate and create the PyDrive client.
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.

drive = GoogleDrive(gauth)

# Choose the file to upload
file_path = 'path_to_your_file'
file_name = 'uploaded_file_name'

# Create and upload the file.
gfile = drive.CreateFile({'title': file_name})
gfile.SetContentFile(file_path)
gfile.Upload()

print(f"File '{file_name}' has been uploaded to Google Drive.")
