from flask import Flask, jsonify,request
import detection_vars as dv
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

                if (rides_count+1)<10:
                    rides_no=f"00{rides_count+1}"
                elif (rides_count+1)>=10 and (rides_count+1)<100:
                    rides_no=f"0{rides_count+1}"
                else:
                    rides_no=rides_count+1

                new_ride_id = f'ride{rides_no}'
                new_ride_data = dv.get_all({"userid":userid,"time":time})

                rides_ref.document(new_ride_id).set(new_ride_data)
                
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

#!Received data
# formatted_output={
#         "date":datetime.datetime.now().strftime('%d/%m/%Y'),
#         "ride_duration":f"{(e_time-s_time)/60} minutes",
#         "start_time":start,
#         "end_time":end,
#         "drowsiness_status":drowsy_data[0],
#         "score":score,
#         "inattention":info["inattention"],
#         "cellphone_det":info["cell_det"],
#         "cell_time":info["cell_time"],
#         "links":access_link if access_link!=0 else "no images",
#         "avg_speed":avg_speed,
#         "distance":float(f"{avg_speed*((e_time-s_time)/3600)}"),
#         "pose_unsafe_perc":info["pose_unsafe_perc"]
#     }
