import string
import random
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CORS(app, resources={r"/*": {"origins": "*"}})

cred = credentials.Certificate("D:/DAMS/code/Service.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

def generate_user_id(name):
    random_number = ''.join(random.choices(string.digits, k=4))
    initials = name[:2].upper()
    user_id = f'u{initials}{random_number}'
    return user_id

@app.route("/")
def server():
    return "client_server API"

@app.route('/login', methods=['POST'])
def login():
    login_data = request.json
    userid = login_data.get('userid')
    password = login_data.get('password')
    user_type = login_data.get('type')
    
    if user_type == 0:
        # Reference to the accounts document
        accounts_ref = db.collection('accounts').document(userid)
        accounts_doc = accounts_ref.get()
        
        # Check if accounts document exists
        if accounts_doc.exists:
            accounts_data = accounts_doc.to_dict()
            # Check if password is correct
            if 'password' in accounts_data and accounts_data['password'] == password:
                # Reference to the userinfo document
                userinfo_ref = db.collection('users').document(userid).collection('userinfo').document(userid)
                userinfo_doc = userinfo_ref.get()
                
                # Extract userinfo data
                if userinfo_doc.exists:
                    userinfo_data = userinfo_doc.to_dict()
                    
                    # Check if rides subcollection exists
                    rides_ref = db.collection('users').document(userid).collection('rides')
                    rides_exists = len(list(rides_ref.list_documents())) > 0
                    
                    # Construct user_data_all object
                    user_data_all = {'userid': userid, 'userinfo': userinfo_data}
                    if rides_exists:
                        user_data_all['rides'] = 1
                    else:
                        user_data_all['rides'] = 0
                    
                    return jsonify({"status":True,"message":"User logged in","userdata":user_data_all}), 200
                else:
                    return jsonify({"status":False,"message":"Userinfo not found for this user"}), 200
            else:
                return jsonify({"status":False,"message":"Unauthorized: Incorrect password"}), 200
        else:
            return jsonify({"status":False,"message":"Account not found for this user"}), 200
        
    elif user_type == 1:
        # Check if a document with userid exists in the organization_codes collection
        org_code_ref = db.collection('organization_codes').document(userid)
        org_code_doc = org_code_ref.get()
        
        if org_code_doc.exists:
            org_code_data = org_code_doc.to_dict()
            # Check if passwords are the same
            if 'password' in org_code_data and org_code_data['password'] == password:
                # Find the document with id userid in the organizations collection
                org_ref = db.collection('organizations').document(userid)
                org_doc = org_ref.get()
                
                # Extract userdata data
                if org_doc.exists:
                    userdata = org_doc.to_dict()
                    return jsonify({"status":True,"message":"Organization logged in","userdata":userdata}), 200
                else:
                    return jsonify({"status":False,"message":"Organization not found"}), 200
            else:
                return jsonify({"status":False,"message":"Unauthorized: Incorrect password"}), 200
        else:
            return jsonify({"status":False,"message":"Organization code not found"}), 200
    else:
        return jsonify({"status":False,"message":"Invalid user_type"}), 200

@app.route('/register', methods=['POST'])
def register_user():
    data = request.json
    username = data.get('name')
    age = data.get('age')
    car_no = data.get('car_no')
    car_model = data.get('car_model')
    car_mileage = data.get('car_mileage')
    
    # Generate user ID
    user_id = generate_user_id(username)
    
    user_ref = db.collection('users').document(user_id)
    user_ref.set({})

    # Add account data to the accounts collection
    accounts_ref = db.collection('accounts').document(user_id)
    accounts_ref.set({
        'userid': user_id,
        'password': data.get('password')
    })
    
    # Create userinfo subcollection
    userinfo_ref = user_ref.collection('userinfo').document(user_id)

    data_todb={
        'id': user_id,
        'name': username,
        'age': age,
        'car_no': car_no,
        'car_model': car_model,
        'car_mileage': car_mileage,
        'affiliated_to':[],
        'score':0,
        'distance':0,
        'avg_speed':0
    }
    userinfo_ref.set(data_todb)
    
    return jsonify({"status":True,"message": "User registered successfully"}), 201


@app.route('/register_org', methods=['POST'])
def register_org():
    data = request.json
    org_id = data.get('org_id')
    org_name = data.get('org_name')
    password=data.get('password')
    
    # Check if the organization ID already exists in the organization_codes collection
    org_code_ref = db.collection('organization_codes').document(org_id)
    org_code_doc = org_code_ref.get()
    
    if org_code_doc.exists:
        org_code_ref.update({
            'org_id': org_id,
            'password': password
        })
        # Create a new organization in the organizations collection
        org_ref = db.collection('organizations').document(org_id)

        org_data={
            'org_id': org_id,
            'org_name': org_name,
            'affiliates':[]
        }

        org_ref.set(org_data)
        
        return jsonify({"status": True, "org_id": org_id}), 201
    else:
        return jsonify({"status": False,"message":"NO MATCHING ID"}), 200

@app.route('/invite', methods=['POST'])
def invite_user():
    data = request.json
    org_id = data.get('org_id')
    userid = data.get('userid')
    
    # Check if the user document exists
    user_ref = db.collection('users').document(userid)
    user_doc = user_ref.get()
    
    if user_doc.exists:
        # Add an invitation document to the invites subcollection
        invite_ref = user_ref.collection('invites').document(org_id)
        invite_ref.set({
            'status': False
        })
        
        return jsonify({"message": "Invitation sent successfully", "org_id": org_id}), 201
    else:
        return "User not found", 404


@app.route('/verify_invite', methods=['POST'])
def verify_invite():
    data = request.json
    userid = data.get('userid')
    org_id = data.get('org_id')
    reply = data.get('reply')
    
    # Reference to the user's invite document
    invite_ref = db.collection('users').document(userid).collection('invites').document(org_id)
    invite_doc = invite_ref.get()
    
    # Check if the invite document exists
    if invite_doc.exists:
        if reply:
            invite_ref.delete()

            org_ref = db.collection('organizations').document(org_id)
            org_doc = org_ref.get()
            
            if org_doc.exists:
                
                org_ref.update({'affiliates': firestore.ArrayUnion([userid])})
                
                # Append org_id to the affiliated_to array in userinfo subcollection
                userinfo_ref = db.collection('users').document(userid).collection('userinfo').document(userid)
                userinfo_ref.update({'affiliated_to': firestore.ArrayUnion([org_id])})
                
                return jsonify({"message": "Invite verified successfully", "org_id": org_id}), 200
            else:
                return "Organization not found", 404
        else:
            # If reply is false, delete the invite document
            invite_ref.delete()
            return jsonify({"message": "Invite declined and removed successfully", "org_id": org_id}), 200
    else:
        return "Invite not found", 404

@app.route('/get_rides_info', methods=['POST'])
def get_rides_info():
    data = request.json
    userid = data.get('userid')
    
    # Reference to the user document
    user_ref = db.collection('users').document(userid)
    user_doc = user_ref.get()
    
    # Check if the user document exists
    if user_doc.exists:
        # Reference to the user's rides subcollection
        rides_ref = user_ref.collection('rides')
        rides_docs = rides_ref.stream()
        
        ride_info = []
        # Iterate over each ride document and extract its data
        for ride_doc in rides_docs:
            ride_data = ride_doc.to_dict()
            ride_info.append(ride_data)
        
        return jsonify({"status":True,"rides":ride_info}), 200
    else:
        return jsonify({"status":False,"message":"User not found"}), 200


@app.route("/get_all_users", methods=['POST'])
def get_all_users():
    data = request.json
    user_ids = data.get("user_ids")

    if user_ids:
        all_users_data = {}
        for userid in user_ids:
            user_ref = db.collection('users').document(userid)
            user_doc = user_ref.get()

            if user_doc.exists:
                user_data = user_doc.to_dict()
                userinfo_ref = user_ref.collection('userinfo').document(userid)
                userinfo_doc = userinfo_ref.get()

                if userinfo_doc.exists:
                    userinfo_data = userinfo_doc.to_dict()
                    rides_ref = user_ref.collection('rides')
                    rides_docs = rides_ref.stream()

                    rides_data = []
                    for ride_doc in rides_docs:
                        ride_data = ride_doc.to_dict()
                        rides_data.append(ride_data)

                    all_users_data[userid] = {'user_data': user_data, 'userinfo_data': userinfo_data, 'rides_data': rides_data}
                else:
                    all_users_data[userid] = {'user_data': user_data, 'userinfo_data': {}, 'rides_data': []}
            else:
                all_users_data[userid] = {'user_data': {}, 'userinfo_data': {}, 'rides_data': []}

        return jsonify({"status":True,"users_data":all_users_data}), 200
    else:
        return "User IDs not provided in the request", 400

if __name__ == '__main__':
    app.run(debug=True)


# python -m flask --app .\server.py run  