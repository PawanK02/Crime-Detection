from flask import Flask, render_template, Response,request
import cv2
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
import sqlite3

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0

upload_folder = os.path.join('static', 'uploads')
app=Flask(__name__)
app.config['UPLOAD'] = upload_folder

def capture_by_frames(): 
    #DATABASE---------------
    conn = sqlite3.connect('fimage.db')
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS mytable(id INTEGER PRIMARY KEY,name TEXT, data BLOB,age TEXT,bloodgroup TEXT)""")
    n = cursor.execute("""SELECT * from mytable""")
    global names,ages,bg
    names = ['None']
    ages = ['-']
    bg = ['-']
    for x in n:
        names += [x[1]]
        ages += [x[3]]
        bg += [x[4]]
    conn.commit()
    cursor.close()
    conn.close()
    #DATABASE-----------------
    print(names)
    global cam
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while True:
        ret, img =cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
       )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])           
            if (confidence < 100):
                x = round(100 - confidence)
                if x>30:
                    xid = names[id]
                    xage=ages[id]
                    xbg=bg[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    xid = "Not recognized..."
                    confidence = "  {0}%".format(round(100 - confidence))
            else:
                xid="Unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img,str(xid),(x+5,y-5),font,1,(255,0,0),2)
            
            cv2.putText(img,'AGE:'+str(xage),(x+5,y+h-45),font,1,(255,0,0),2)
            cv2.putText(img,'Blood:'+str(xbg),(x+5,y+h-85),font,1,(255,0,0),2)
            cv2.putText(img,'Match:'+str(confidence),(x+5,y+h),font,1,(255,0,0),2)
        ret1, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def capture_crime():
    global cam
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # For each person, enter one numeric face id
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                        str(count) + ".jpg", gray[y:y+h,x:x+w])
            if(count==15):
                 cv2.imwrite("static/uploads/C" + str(face_id)+ ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
            break
    # Do a bit of cleanup
    #print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    #cv2.destroyAllWindows()
    ret1, buffer = cv2.imencode('.jpg', img)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/loginP',methods=['GET'])
def loginP():
    if request.method == 'GET':
        global CITIZEN,POLICE
        CITIZEN=0
        POLICE=1
        pid = request.args['pid']
        pwd = request.args['pwd']
        conn = sqlite3.connect('pol')
        cursor = conn.cursor()
        #cursor.execute("""CREATE TABLE IF NOT EXISTS police(pid TEXT PRIMARY KEY, pwd TEXT)""")
       # cursor.execute("""INSERT into police(pid, pwd) VALUES(?,?)""",('11','qwer'))
        n = cursor.execute("""SELECT * from police""")
        for x in n:
            if x[0]==pid and x[1]==pwd:
                return render_template("stop.html")
        conn.commit()
        cursor.close()
        conn.close()
    context = { 'access': 'denied'}
    return render_template("login.html",**context)

@app.route('/loginC',methods=['GET'])
def loginC():
    if request.method == 'GET':
        global CITIZEN,POLICE
        CITIZEN=1
        POLICE=0
        pid = request.args['cid']
        pwd = request.args['cpwd']
        conn = sqlite3.connect('cit')
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS citizen(pid TEXT PRIMARY KEY, pwd TEXT)""")
        cursor.execute("""INSERT into citizen(pid, pwd) VALUES(?,?)""",('citizen','1234'))
        n = cursor.execute("""SELECT * from citizen""")
        for x in n:
            if x[0]==pid and x[1]==pwd:
                return render_template("citizenpage.html")
        conn.commit()
        cursor.close()
        conn.close()
    context = { 'access': 'denied'}
    return render_template("login.html",**context)
    
@app.route('/citizenpage',methods=['POST'])
def citizenpage():
    return render_template("citizenpage.html")

@app.route('/search',methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        conn = sqlite3.connect('fimage.db')
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS mytable(id INTEGER PRIMARY KEY,name TEXT, data BLOB,age TEXT,bloodgroup TEXT)""")
        n = cursor.execute("""SELECT * from mytable""")
        global criminal
        criminal = request.args['search']
        for x in n:
            if(x[1]==criminal):
                context = {
                'name': criminal,
                'age' : x[3],
                'bg' : x[4],
                'nig': ''+str(x[0])
            }
    return render_template('search.html',**context)

@app.route('/start',methods=['POST'])
def start():
    return render_template('index.html')
@app.route('/newup',methods=['POST'])
def newup():
    return render_template('newup.html')
@app.route('/stop',methods=['POST'])
def stop():
    if cam.isOpened():
        cam.release()
    return render_template('stop.html')

@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/up',methods=['POST'])
def up():
    return render_template('image_render.html')

@app.route('/gobck',methods=['POST'])
def gobck():
    if CITIZEN==0 and POLICE==1:
        return render_template('stop.html')
    else:
        return render_template('citizenpage.html')

@app.route('/newcrime')
def newcrime():
    return Response(capture_crime(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train',methods=['POST'])
def train():
    if(request.method == 'POST'):
        path = 'dataset'
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
        # function to get the images and label data
        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L') # grayscale
                img_numpy = np.array(PIL_img,'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
            return faceSamples,ids
        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml') 
        val = len(np.unique(ids))
        
        conn = sqlite3.connect('fimage.db')
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS mytable(id INTEGER PRIMARY KEY,name TEXT, data BLOB,age TEXT,bloodgroup TEXT)""")
        cursor.execute("SELECT * FROM mytable") 
        temp = len(cursor.fetchall())
        if val>temp:
            print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
            with open("dataset/User." + face_id + ".15.jpg",'rb') as f:
                data = f.read()
            m = cursor.execute("""INSERT into mytable(id,name,data,age,bloodgroup) VALUES(?,?,?,?,?)""",(face_id,cname,data,cage,cbg))
        global names,ages,bg
        names = ['none']
        ages = ['-']
        bg = ['-']
        n = cursor.execute("""SELECT * from mytable""")
        for x in n:
           names += [x[1]]
           ages += [x[3]]
           bg += [x[4]]
        conn.commit()
        cursor.close()
        conn.close()    
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    
        return render_template('stop.html')
    
@app.route('/tp',methods=['GET'])
def tp():
    if request.method == 'GET':
        global face_id
        global cname,cage,cbg
        face_id = request.args['id']
        cname = request.args['fname']
        cage = request.args['age']
        cbg = request.args['bg']
        return render_template('added.html')

@app.route('/upload_file',methods=['POST'])
def upload_file():
    if request.method == 'POST':
        #DATABASE---------------
        conn = sqlite3.connect('fimage.db')
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS mytable(id INTEGER PRIMARY KEY,name TEXT, data BLOB,age TEXT,bloodgroup TEXT)""")
        n = cursor.execute("""SELECT * from mytable""")
        global names,ages,bg
        names = ['None']
        ages = ['-']
        bg = ['-']
        for x in n:
           names += [x[1]]
           ages += [x[3]]
           bg += [x[4]]
        conn.commit()
        cursor.close()
        conn.close()
        #DATABASE-----------------
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        path2 = file.filename
        absolute_path = os.path.join(os.getcwd(), 'static', 'uploads', path2);
        img1 = cv2.imread(absolute_path)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        font = cv2.FONT_HERSHEY_SIMPLEX
        id = 0
       # while True:
        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5 )
        for(x,y,w,h) in faces:
            cv2.rectangle(img1, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w]) 
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            x=(round(100 - confidence))
            if x>=40:
                crimage = ages[id]
                crimbg = bg[id]
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            elif x<40 and x>20:
                crimage = ""
                crimbg = ""
                id = "please provide clear image"
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                crimage = ""
                crimbg = ""
                id = "None"
                confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            crimage = ""
            crimbg = ""
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    img1, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    2, 
                    (0,0,0), 
                    5
                )
        cv2.putText(
                    img1, 
                    str(confidence), 
                    (x+5,y+h), 
                    font, 
                    1, 
                    (0,0,0),
                )
        cv2.imshow("image", img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        context = {
            'img': img,
            'name': id+"   ("+confidence+"  match)",
            'cm': 'CRIMINAL',
            'age': crimage,
            'bloodgroup':crimbg }
        return render_template('image_render.html', **context)

if __name__=='__main__':
    app.run(debug=True,use_reloader=False, port=8000)