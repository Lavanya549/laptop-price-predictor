import streamlit as st
import pickle
import numpy as np

pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open("df.pkl",'rb'))

st.title("Laptop Predictor")

company=st.selectbox("Brand",df['Company'].unique())

type=st.selectbox("Type",df['TypeName'].unique())

ram=st.selectbox("Ram(in GB)",[2,4,6,8,12,16,24,32,64])

weight=st.number_input("Weight of the laptop")

touch=st.selectbox("Touchscreen",['No','Yes'])

ips=st.selectbox("IPS",['Yes','No'])

#Screen size

screen_size=st.number_input("Screen Size")

#resolution

resolution=st.selectbox("Screen Resolution",[ "1920x1080", "1366x768", "2560x1440", "3840x2160", "1600x900", "1440x900", "1280x800", "3200x1800"])

#cpu brand

cpu_brand=st.selectbox("Cpu Brand",df['Cpu brand'].unique())


hdd=st.selectbox("HDD(in GB)",['0','128','256','512','1024','2048'])

sdd=st.selectbox("SDD(in GB)",['0','8','128','256','512','1024'])

gpu_brand=st.selectbox("Gpu Brand",df['Gpu brand'].unique())

os=st.selectbox("OS",df['os'].unique())

if st.button("Predict Price"):
  ppi=None
  if touch=="Yes":
    touch=1
  else:
    touch=0
  if ips=="Yes":
    ips=1
  else:
    ips=0
  X_res=int(resolution.split('x')[0])
  Y_res=int(resolution.split('x')[1])
  if X_res is not None and Y_res is not None and screen_size:
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
  else:
    print("Resolution or screen size values are missing or invalid.")

  query=np.array([company,type,ram,weight,touch,ips,ppi,cpu_brand,hdd,sdd,gpu_brand,os])
  query=query.reshape(1,12)
  st.title(np.exp(pipe.predict(query)))

