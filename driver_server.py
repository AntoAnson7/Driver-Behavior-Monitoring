from flask import Flask, jsonify,request
import detection_vars as dv
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
app = Flask(__name__)

# cred = credentials.Certificate("D:/DAMS/code/Service.json")
# firebase_admin.initialize_app(cred)

db=firestore.client()

@app.route("/get", methods=['POST'])
def get_all():
    data = request.json
    userid = data.get("userid")
    time=data.get("time")

    if userid:
        user_ref = db.collection('users').document(userid)
        user_doc = user_ref.get()

        if user_doc.exists:
            userinfo_ref = user_ref.collection('userinfo').document(userid)
            userinfo_doc = userinfo_ref.get()

            if userinfo_doc.exists:
                userinfo_data = userinfo_doc.to_dict()
                rides_ref = user_ref.collection('rides')
                rides_count = len(list(rides_ref.list_documents()))

                new_ride_id = f'ride{rides_count + 1}'
                new_ride_data = dv.get_all({"userid":userid,"time":time})

                rides_ref.document(new_ride_id).set(new_ride_data)
                
                # Calculate total distance and duration
                # rides_docs = rides_ref.stream()
                # total_distance = 0
                # total_duration = 0
                # average_speed=0
                # for ride_doc in rides_docs:
                #     ride_data = ride_doc.to_dict()
                #     total_distance += ride_data.get('distance', 0)
                #     # total_duration += ride_data.get('ride_duration', 0)
                #     average_speed += ride_data.get('avg_speed', 0)
                
                # # Calculate average speed
                # average_speed = total_distance / len(rides_docs) if len(rides_docs) > 0 else 0

                # # Update the userinfo document with the calculated values
                # userinfo_ref.update({'distance': total_distance, 'average_speed': average_speed})

                # _id = userinfo_data["id"]
                return jsonify({"status": f"db updated with ride {rides_count + 1} for user: ",
                                "data": new_ride_data})
            else:
                return "Userinfo not found for this user", 404
        else:
            return "User not found", 404
    else:
        return "Userid not provided in the request", 400


if __name__ == '__main__':
    app.run(debug=True)


#  python -m flask --app .\driver_server.py run