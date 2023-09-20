from google.cloud import firestore

from google.cloud import firestore
from google.oauth2 import service_account

import os
# Set the path to your service account JSON file
service_account_path = "plant-pot-qjbt-e6beedd4d5e3.json"

# Set the environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path

# Initialize Firestore client
credentials = service_account.Credentials.from_service_account_file(service_account_path)
db = firestore.Client(credentials=credentials)


def create_user_persona(user_id):

    doc_ref = db.collection('SAIG-Project').document(user_id)
    doc = doc_ref.get()
    if not doc.exists:
        doc_ref.set({
            "name": "",
            "status_type" : "",
            "status_water" : "", # low, well
            "status_light" : "", # low, well
            "q" : [{"input" : "ok"}, {"input" : "ok"}, {"input" : "ok"}],
            "a" : [{"output" : "Nok"}, {"input" : "ok"}]
        })  

    print("Done init persona")

def get_user_document(user_id):
    doc_ref = db.collection('SAIG-Project').document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        user_dict = doc.to_dict()
        return user_dict
    else:
        create_user_persona(user_id)
        doc_ref = db.collection('SAIG-Project').document(user_id)
        doc = doc_ref.get()
        user_dict = doc.to_dict()
        return user_dict

def update_user_data(user_id, data):
    if(len(data) > 0):
        doc_ref = db.collection('SAIG-Project').document(user_id)
        doc_ref.update(data)

def reset_user_persona(user_id):

    doc_ref = db.collection('SAIG-Project').document(user_id)
    doc_ref.set({
        "name": "",
    })  

    print("Done reset persona")

create_user_persona("id_02")

data = get_user_document("id_02")

print(data)