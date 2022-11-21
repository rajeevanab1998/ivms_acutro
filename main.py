import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import fileinput

from sklearn.ensemble import GradientBoostingClassifier
import random
from tkinter import *
from tkinter.ttk import *

import time
import pandas as pd

import numpy as np
import codecs

from sklearn.model_selection import train_test_split

column = "Instant_Fuel_consum Engine_cool_temp RPM Vehicle_speed Distance_Travelled Instant_Engine_Power Fuel_Used IAT Average_speed Prediction".split()


val_IFC = random.randint(10, 20)  # arr[0]
val_EngineCool = random.randint(150, 220)  # arr[1]
val_RPM = random.randint(2500, 5200)  # arr[2]
val_VS = random.randint(30, 100)  # arr[3]
val_DT = random.randint(3, 100)  # arr[4]
val_IEP = random.randint(30, 40)  # arr[5]
val_FU = random.randint(10, 20)  # arr[6]
val_IAT = random.randint(70, 110)  # arr[7]
val_AS = random.randint(30, 100)  # arr[8]


arr = np.array([val_IFC, val_EngineCool, val_RPM, val_VS, val_DT, val_IEP, val_FU, val_IAT, val_AS])

# function to generate random inputs
def random_val():
    val_IFC = random.randint(10, 20)  # arr[0]
    val_EngineCool = random.randint(90, 220)  # arr[1]
    val_RPM = random.randint(500, 5200)  # arr[2]
    val_VS = random.randint(10, 100)  # arr[3]
    val_DT = random.randint(3, 100)  # arr[4]
    val_IEP = random.randint(30, 40)  # arr[5]
    val_FU = random.randint(10, 20)  # arr[6]
    val_IAT = random.randint(30, 110)  # arr[7]
    val_AS = random.randint(10, 100)  # arr[8]

    arr = np.array([val_IFC, val_EngineCool, val_RPM, val_VS, val_DT, val_IEP, val_FU, val_IAT, val_AS])

    conditions = [(arr[1] < 195) & (arr[2] < 4000) & (arr[7] < 90) & (arr[8] < 60) & (arr[3] < 70),
                  (arr[1] >= 195) | (arr[2] >= 4000) | (arr[7] >= 90) | (arr[8] >= 60) | (arr[3] >= 70)]

    values = [0, 1]
    arr = np.append(arr, np.select(conditions, values), axis=None)
    return arr

# THIS FUNCTION TAKES THE RANDOM VALUE FROM THE ARRAY AND PUTS IT IN THE DATAFRAME
def add_value():
    df = pd.DataFrame()
    for i in range(1):

        newarr = random_val()
        # print(newarr)
        df1 = pd.DataFrame(data=[newarr[0:10]], index=None, columns=column)
        df = df.append(df1)
    # print(df)

    return df

root = Tk()
root.title('ACCIDENT PREDICTION')
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))

df2=pd.read_csv(r'C:\Users\saranya sreedev\Desktop\Acutro\IVMS\upsam_accident.csv')

df2=df2.drop(['Unnamed: 0'],axis=1)

df2.columns=column

X=df2.iloc[:,:-1]

y=df2[["Prediction"]]








X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

gbm_classifier = GradientBoostingClassifier(n_estimators=300, random_state=1, learning_rate=0.5)

model=gbm_classifier.fit(X_train, y_train)

# MAKE PREDICTIONS ON THE TESTING SET
while True:
    # dataframe = pd.DataFrame()
    dataframe1 = add_value()

    target1 = pd.DataFrame(dataframe1, columns=["Prediction"])
    Xx = pd.DataFrame(dataframe1, columns=column)

    Xx = Xx.drop(["Prediction"], axis=1)

    yy = target1["Prediction"]

    y_pred = model.predict(Xx)

    y_predicted = []

    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            y_predicted.append("Not Accident")
        else:
            y_predicted.append("Accident")
   # st.success(y_pred[0:3])
    print(y_predicted[:])


    # This function is used to
    # display time on the label

    def temp():

        for k in range(len(Xx['Engine_cool_temp'])):
            print(Xx['Engine_cool_temp'].iloc[k])
            string = Xx['Engine_cool_temp'].iloc[k]

            temp = Label(root, text="TEMPERATURE",font=('Helvetica', 10, 'bold'))
            temp.place(x=40, y=200)

            lbl1 = Label(root, font=('digital-7', 150, 'bold'),
                         background='black',
                         foreground='blue')
            lbl1.place(relx=10.0,
                       rely=1.0,
                       anchor='sw')
            # Placing clock at the centre
            lbl1.config(text=string)
    # lbl1.after(1000, time)

    # Styling the label widget so that clock
    # will look more attractive

    # of the tkinter window
            lbl1.pack(anchor='sw')
    temp()

    def rpm():
        for k in range(len(Xx['RPM'])):
            print(Xx['RPM'].iloc[k])
            string = Xx['RPM'].iloc[k]

            rpm = Label(root, text="RPM", font=('Helvetica', 10, 'bold'))
            rpm.place(x=40, y=610)


            lbl2 = Label(root, font=('digital-7', 150, 'bold'),
                         background='black',
                         foreground='blue')
            lbl2.place(relx=1.0,
                              rely=0.0,
                              anchor='ne')

            # Placing clock at the centre
            lbl2.config(text=string)
    # lbl1.after(1000, time)

    # Styling the label widget so that clock
    # will look more attractive

    # of the tkinter window
            lbl2.pack(anchor='ne')

    rpm()

    def speed():
        for k in range(len(Xx['Vehicle_speed'])):
            print(Xx['Vehicle_speed'].iloc[k])
            string = Xx['Vehicle_speed'].iloc[k]

            speed = Label(root, text="SPEED", font=('Helvetica', 10, 'bold'))
            speed.place(x=1350, y=170)


            lbl3 = Label(root, font=('digital-7', 150, 'bold'),
                         background='black',
                         foreground='blue')
            lbl3.place(relx=0.0,
                     rely=1.0,
                     anchor='sw')

            # Placing clock at the centre
            lbl3.config(text=string)
    # lbl1.after(1000, time)

    # Styling the label widget so that clock
    # will look more attractive

    # of the tkinter window
            lbl3.pack(anchor='sw')

    speed()

    def prediction():
        for k in range(len(y_predicted)):
            print(y_predicted[k])
            string = y_predicted[k]

            predict = Label(root, text="PREDICTION", font=('Helvetica', 10, 'bold') )
            predict.place(x=700, y=580)
            lbl4 = Label(root, font=('digital-7', 150, 'bold'),
                         background='black',
                         foreground='blue')
            lbl4.place(relx=0.0,
                     rely=1.0,
                     anchor='center')

            # Placing clock at the centre
            lbl4.config(text=string)
    # lbl1.after(1000, time)

    # Styling the label widget so that clock
    # will look more attractive

    # of the tkinter window
            lbl4.pack(anchor='center')

    prediction()
    mainloop()

    # importing whole module

    # creating tkinter window

    #ra.codec_html(X_test,y_pred)
  #  print(y_pred[0:5])


