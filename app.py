from flask import Flask, render_template, Response,request
import cv2
import face_recognition
import numpy as np
# from PIL import Image
import urllib.request

# fetching data      
import requests

def fetch_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Unable to fetch data. Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
# url = "http://localhost:5500/Students/65a94f6a0413794bb21542a7"
url ="https://cms-uov8.onrender.com/Students/65c3a72bcebbd735bd652ef0"
data = fetch_data(url)
if data:
    # print(f"Data fetched successfully:\n{data}")
    
    for i in range(len(data)):
       name=data[i]['name']
       print(f"\n{data[i]['name']}")
       
       
       


app=Flask(__name__)
camera = cv2.VideoCapture(0)
# Load a sample picture and learn how to recognize it.
krish_image = face_recognition.load_image_file("Krish/krish.jpg")
krish_face_encoding = face_recognition.face_encodings(krish_image)[0]

# Load a second sample picture and learn how to recognize it.
bradley_image = face_recognition.load_image_file("Bradley/bradley.jpg")
bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]
# print(bradley_face_encoding)
# Create arrays of known face encodings and their names
known_face_encodings = [
    krish_face_encoding,
    bradley_face_encoding,
]
known_face_names = [
    "Krish",
    "Bradly"
]

for i in range(len(data)):
    img_res = urllib.request.urlopen(data[0]['photoUrl'])
    img_arr=np.array(bytearray(img_res.read()),dtype=np.uint8)
    img=cv2.imdecode(img_arr,1)
    student_face_encoding = face_recognition.face_encodings(img)[0]
    print(student_face_encoding)
    known_face_encodings.append(student_face_encoding)
    known_face_names.append(data[i]['name'])
    print(known_face_names)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


subjects=[]
dates=[]
@app.route('/get-text', methods=['GET', 'POST'])
def foo():
    subject = request.form['subject']
    date = request.form['date']
    subjects.append(subject)
    dates.append(date)
    print(subject,date)
    return render_template('index.html',data=data)

# subject = request.form['subject']
# date = request.form['date']


def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # rgb_small_frame = small_frame[:, :, ::-1]
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    print(subjects[0],dates[0])
                    
                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                if(name=="unknown"):
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
                else:
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (34,139,34), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (34,139,34), cv2.FILLED)
                
                font = cv2.FONT_HERSHEY_DUPLEX
                # font =cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name +" is Present", (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('home.html',data=data)


@app.route('/back',methods=['GET', 'POST'])
def home():
    return render_template('home.html',data=data)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(host='localhost',debug=True)