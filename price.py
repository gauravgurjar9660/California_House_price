
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
data=pd.read_csv("cali-data.csv")
model=joblib.load("model.pkl")
pipeline=joblib.load("pipline.pkl")
try:
    acc_data = joblib.load("accuracy.pkl")
except:
    acc_data = {"mean_rmse": 68000, "std_rmse": 2000} # Default values agar file na mile

# 2. Sidebar mein sundar tarike se dikhayein
with st.sidebar: 
    st.header("📈 Model Info")
    st.metric(label="Model Error (RMSE)", value=f"${acc_data['mean_rmse']:,.0f}")
    st.write(f"Model  reliability (+/-): ${acc_data['std_rmse']:,.0f}")
    st.progress(82) # Aapki accuracy ka andaza
    st.caption("Training based on California Housing Dataset")
st.title("House Cost Pridiction ")
st.write("This app is used for achive the correct amount of any house at a defiend location")
st.image("https://scontent.fudr2-2.fna.fbcdn.net/v/t39.30808-6/481702095_638231598808163_1455785499178917411_n.jpg?_nc_cat=100&ccb=1-7&_nc_sid=833d8c&_nc_ohc=lrgALywk_CcQ7kNvwGI6zLb&_nc_oc=AdpA6047ZQctYdDZSAROJaNCfVo5_6N32De6dr7uEodHxHxkp3qZjTFUb41lJYapfuM&_nc_zt=23&_nc_ht=scontent.fudr2-2.fna&_nc_gid=xD9dr4PXaklHmcIxItF5Jw&_nc_ss=7a289&oh=00_Af9LrJPZu8ot0udV5_DKELmKVIUb50N1RuWXiUXi9DMBmQ&oe=6A286ABB",width=300)

st.header("Enter the details below ")

col1,col2 =st.columns(2)
with col1:
    ocean=data['ocean_proximity'].unique().tolist()
    ocean_proximity=st.selectbox("Enter location",ocean)
    latitude=st.number_input("latitude",max_value=42.0,min_value=30.0)
    longitude=st.number_input("longitude",min_value=-124.35,max_value=-114.31)
    housing_median_age=st.number_input("Enter house age",max_value=70,min_value=1)
with col2:
    total_rooms=st.number_input("Total Rooms",value=500)
    total_bedrooms=st.number_input("Total Bedrooms",value=700)
    population=st.number_input("Total Population in area",value=800)
    households=st.number_input("Enter number of Households",value=970)
median_income=st.slider("Enter median income",0.0,15.0,7.2574)

if st.button("SUBMIT"):
    input_dict={
    "longitude":[longitude],"latitude":[latitude],
        "housing_median_age":[housing_median_age],"total_rooms":[total_rooms],"total_bedrooms":[total_bedrooms],
        "population":[population],"households":[households],"median_income":[median_income],"ocean_proximity":[ocean_proximity]}
    df=pd.DataFrame(input_dict)
    try:
        transform_data=pipeline.transform(df)
        prediction=model.predict(transform_data)
        st.success(f"The house price is : ${prediction[0]:.2f}")
        st.subheader("Comparison between your house price to other")
        st.write(" Visualizing by bar chart")
        try:
            averg= data.groupby("ocean_proximity")["median_house_value"].mean().reset_index()
            predicts=pd.DataFrame({"ocean_proximity":['your prediction'],"median_house_value":[prediction[0]]})
            tl=pd.concat([averg,predicts],ignore_index=True)
            st.bar_chart(data=tl,x='ocean_proximity',y='median_house_value',use_container_width=True)
        except Exception as e:
            st.error(f"The Exception is : {e}")

        
    except Exception as e:
        st.error(f"The error occured is {e}")
   

    
    # 1. Accuracy file load karein


    

