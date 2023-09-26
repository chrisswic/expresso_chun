import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.linear_model import LinearRegression
import seaborn as sns
data = pd.read_csv('Expresso_churn_dataset (1).csv')

#.......load the model.................
model = joblib.load(open('Expresso_Churn (1).pkl', 'rb'))

#....................Streamlit Development Starts.............
# st.markdown("hi style = 'text-align: right; color: FF3FA4"> EXPRESSO CHURN PREDICTION CHALLENGE<h1>, unsafe_allow_html = True)

# st.title('EXPRESSO CHURN PREDICTION CHALLENGE')
st.write('Built for yellow beast class')

st.markdown("<h1 style = ' color: #176B87'>EXPRESSO CHURN PREDICTION CHALLENGE</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'top-margin: 0rem; color: F8FF95'>Built By Christopher swiz. </h1>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)

# import base64
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(f"""<style>.stApp {{background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#         background-size: cover}}</style>""", unsafe_allow_html=True)
# add_bg_from_local('expresso png.png')


# st.write('Pls Enter your username')
username = st.text_input('Please enter Username:')
if st.button('Submit name'):
    st.success(f"Welcome {username}. Pls enjoy your usage")

st.markdown("<br> <br>", unsafe_allow_html= True)
st.markdown("<br> <br>", unsafe_allow_html= True)
st.markdown("<h2 style = 'top-margin: 0rem;text-align: center; color: #A2C579'>Project Introduction</h1>", unsafe_allow_html = True)

st.markdown("<p style = 'top margin: 0rem; 'text align: justify; color: #AED2FF'>Expresso is an African telecommunications services company that provides telecommunication services in two African markets: Mauritania and Senegal.Expresso is known for its commitment to expanding and improving telecommunications infrastructure and connecting people in underserved areas to reliable communication services. The data describes 2.5 million Expresso clients with more than 15 behaviour variables in order to predict the clients' churn probability.</p>", unsafe_allow_html = True)

heat_map = plt.figure(figsize = (14,7))#........create a heatmap plot
correlation_data = data[['REGULARITY',	'REVENUE',	'FREQUENCE_RECH', 'CHURN']]#.........select data for correlation
sns.heatmap(correlation_data.corr(), annot = True, cmap = 'BuPu')

st.write(heat_map)
data.drop('user_id', axis = 1, inplace = True)
st.write(data.sample(5))

st.sidebar.image('logo expresso.png', width = 100, caption= f"Welcome {username}", use_column_width= True)

st.markdown("<br>", unsafe_allow_html= True)

# picture = st.camera_input('take a picture')
# if picture:
#     st.sidebar.image(picture, use_column_width= True, caption = f"welcome {username}")

st.sidebar.write('Pls decide your variable input type')
input_style = st.sidebar.selectbox('Pick Your Preferred input', ['Slider Input', 'Number Input'])

if input_style == 'Slider Input':
    regularity = st.sidebar.slider('REGULARITY', data['REGULARITY'].min(), data['REGULARITY'].max())
    revenue = st.sidebar.slider('REVENUE', data['REVENUE'].min(), data['REVENUE'].max())
    frequencerech = st.sidebar.slider('FREQUENCE_RECH', data['FREQUENCE_RECH'].min(),  data['FREQUENCE_RECH'].max())
    frequence = st.sidebar.slider('FREQUENCE', data['FREQUENCE'].min(),  data['FREQUENCE'].max())
    top_pac = data['TOP_PACK'].unique()
    top_pac = st.sidebar.select_slider('Select Your Top Pac', top_pac)
    region = data['REGION'].unique()
    region = st.sidebar.select_slider('Select Your Region', region)


else:
    regularity = st.sidebar.number_input('REGULARITY', data['REGULARITY'].min(), data['REGULARITY'].max())
    revenue = st.sidebar.number_input('REVENUE', data['REVENUE'].min(), data['REVENUE'].max())
    frequencerech = st.sidebar.number_input('FREQUENCE_RECH', data['FREQUENCE_RECH'].min(),  data['FREQUENCE_RECH'].max())
    frequence = st.sidebar.number_input('FREQUENCE', data['FREQUENCE'].min(),  data['FREQUENCE'].max())
    top_pac = data['TOP_PACK'].unique()
    top_pac = st.sidebar.select_slider('Select Your Top Pac', top_pac)
    region = data['REGION'].unique()
    region = st.sidebar.select_slider('Select Your Region', region)

st.subheader("Your Inputted Data")
input_var = pd.DataFrame([{'REGULARITY': regularity, 'REVENUE': revenue, 'FREQUENCE_RECH': frequencerech, 'FREQUENCE': frequence, 'REGION': region, 'TOP_PACK': top_pac}])

st.write(input_var)

from sklearn.preprocessing import LabelEncoder, StandardScaler
lb = LabelEncoder()
scaler = StandardScaler()

# def transformer(dataframe):
#     # scale the numerical columns
#     for i in dataframe.columns: # ---------------------------------------------- Iterate through the dataframe columns
#         if i in dataframe.select_dtypes(include = 'number').columns: # --------- Select only the numerical columns
#             dataframe[[i]] = scaler.fit_transform(dataframe[[i]]) # ------------ Scale all the numericals

#     # label encode the categorical columns
#     for i in dataframe.columns:  # --------------------------------------------- Iterate through the dataframe columns
#         if i in dataframe.select_dtypes(include = ['object', 'category']).columns: #-- Select all categorical columns
#             dataframe[i] = lb.fit_transform(dataframe[i]) # -------------------- Label encode selected categorical columns
#     return dataframe

for i in input_var.columns: # ---------------------------------------------- Iterate through the dataframe columns
    if i in input_var.select_dtypes(include = 'number').columns: # --------- Select only the numerical columns
        input_var[[i]] = scaler.fit_transform(input_var[[i]]) # ------------ Scale all the numericals

for i in input_var.columns:  # --------------------------------------------- Iterate through the dataframe columns
    if i in input_var.select_dtypes(include = ['object', 'category']).columns: #-- Select all categorical columns
        input_var[i] = lb.fit_transform(input_var[i]) # -------------------- Label encode selected categorical columns


st.markdown("<br>", unsafe_allow_html= True)
tab1, tab2 = st.tabs(["Prediction Pane", "Intepretation Pane"])

with tab1:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_var)
        st.write("Predicted Churn is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

# with tab2:
#     st.subheader('Model Interpretation')
#     st.write(f"Churn = {model.intercept_.round(2)} + {model.coef_[0].round(2)} REGULARITY + {model.coef_[1].round(2)} REVENUE + {model.coef_[2].round(2)} FREQUENCE_RECH + {model.coef_[3].round(2)} FREQUENCE + {model.coef_[4].round(2)} REGION + {model.coef_[5].round(2)} TENURE")

#     st.markdown("<br>", unsafe_allow_html= True)

#     st.markdown(f"- The expected Churn is {model.intercept_}")

#     st.markdown(f"- For every additional 1 dollar generated on REGULARITY, the expected CHURN is expected to decrease by ${model.coef_[0].round(2)}  ")

#     st.markdown(f"- For every additional 1 dollar generated as REVENUE, churn is expected to increase by ${model.coef_[1].round(2)}  ")

#     st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}  ")

