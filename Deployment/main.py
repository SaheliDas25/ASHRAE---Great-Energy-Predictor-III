import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import pickle
from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re


import flask
app = Flask(__name__,template_folder='template')


# In[ ]:


''' add new features'''
def add_feature(test_copy):
    test_copy['timestamp']=pd.to_datetime(test_copy['timestamp'])
    test_copy['hour']=np.uint8(test_copy['timestamp'].dt.hour)
    test_copy['day']=np.uint8(test_copy['timestamp'].dt.day)
    test_copy['month']=np.uint8(test_copy['timestamp'].dt.month)
    test_copy['weekday']=np.uint8(test_copy['timestamp'].dt.dayofweek)
    ''' change the datatype to reduce memory usage'''
    #test_copy['building_id']=np.uint8(test_copy['building_id'])
    #test_copy['meter']=np.uint8(test_copy['meter'])
    #test_copy['site_id']=np.uint8(test_copy['site_id'])
    #test_copy['square_feet']=np.uint8(test_copy['square_feet'])
    
    return test_copy


# In[ ]:


def preprocessed(test_copy):
    LabelEn = load('LabelEncoder.joblib')
    test_copy['primary_use'] = LabelEn.transform(test_copy['primary_use'])
    test_copy['year_built'].fillna(1969.0,inplace  = True)
    test_copy['floor_count'].fillna(3.0,inplace  = True)
    test_copy['air_temperature'].fillna(15.505707129926357,inplace  = True)
    test_copy['cloud_coverage'].fillna(1.9733458449444876,inplace  = True)
    test_copy['dew_temperature'].fillna(7.585971305989484,inplace  = True)
    test_copy['precip_depth_1_hr'].fillna(0.9182989150029545,inplace  = True)
    test_copy['sea_level_pressure'].fillna(1016.1999821450886,inplace  = True)
    test_copy['wind_direction'].fillna(173.84081711747407,inplace  = True)
    test_copy['wind_speed'].fillna(3.3861529705667808,inplace  = True)
    test_copy.drop('timestamp',axis=1,inplace=True)
    return test_copy


# In[ ]:


@app.route('/')
def hello_world():
    return 'Hello World Saheli!'


@app.route('/Ashrae_home')
def Ashrae_home():
    return flask.render_template('Ashrae_home.html')

@app.route('/predict', methods=['POST'])
def predict():
	print(request.form)
	to_predict_list = request.form.to_dict();
	for i in to_predict_list.keys():
		if to_predict_list[i] == '' or to_predict_list[i] == None or to_predict_list[i] == 'None':
			to_predict_list[i] = np.nan
		else:
			if i != 'primary_use' and i != 'timestamp':
				val=float(to_predict_list[i])
				to_predict_list[i] = val
	df = pd.DataFrame(to_predict_list,index=[0])
	feature_added_df=add_feature(df)
	preprocesse_df=preprocessed(feature_added_df)
	df_pred = pd.DataFrame()
	clf1 = load('Lasso_1.joblib')
	clf1_pred=clf1.predict(preprocesse_df)
	#stack.append(clf1.predict(preprocesse_df))
	print(clf1_pred,type(clf1_pred))
	df_pred = pd.concat([df_pred,pd.DataFrame(clf1_pred)],axis=1)
	clf2 = load('Ridge_2.joblib')
	clf2_pred=clf2.predict(preprocesse_df)
	#stack.append(clf2.predict(preprocesse_df))
	df_pred = pd.concat([df_pred,pd.DataFrame(clf2_pred)],axis=1)
	clf3 = load('ElasticNet_3.joblib')
	clf3_pred=clf3.predict(preprocesse_df)
	#stack.append(clf3.predict(preprocesse_df))
	df_pred = pd.concat([df_pred,pd.DataFrame(clf3_pred)],axis=1)
	clf4 = load('XGBRegressor_4.joblib')
	clf4_pred=clf4.predict(preprocesse_df)
	df_pred = pd.concat([df_pred,pd.DataFrame(clf4_pred)],axis=1)
	#stack.append(clf4.predict(preprocesse_df))
	clf5 = load('DecisionTreeRegressor_5.joblib')
	clf5_pred=clf5.predict(preprocesse_df)
	df_pred = pd.concat([df_pred,pd.DataFrame(clf5_pred)],axis=1)
	#stack.append(clf5.predict(preprocesse_df))
	clf6 = load('LinearRegression_6.joblib')
	clf6_pred=clf6.predict(preprocesse_df)
	df_pred = pd.concat([df_pred,pd.DataFrame(clf6_pred)],axis=1)
	#stack.append(clf6.predict(preprocesse_df))
	clf7 = load('Lasso_7.joblib')
	clf7_pred=clf7.predict(preprocesse_df)
	df_pred = pd.concat([df_pred,pd.DataFrame(clf7_pred)],axis=1)
	#stack.append(clf7.predict(preprocesse_df))
	clf8 = load('Ridge_8.joblib')
	clf8_pred=clf8.predict(preprocesse_df)
	df_pred = pd.concat([df_pred,pd.DataFrame(clf8_pred)],axis=1)
	#stack.append(clf8.predict(preprocesse_df))
	clf9 = load('ElasticNet_9.joblib')
	clf9_pred=clf9.predict(preprocesse_df)
	df_pred = pd.concat([df_pred,pd.DataFrame(clf9_pred)],axis=1)
	#stack.append(clf9.predict(preprocesse_df))
	clf10 = load('XGBRegressor_10.joblib')
	clf10_pred=clf10.predict(preprocesse_df)
	df_pred = pd.concat([df_pred,pd.DataFrame(clf10_pred)],axis=1)
	#stack.append(clf10.predict(preprocesse_df))
	
	print(df_pred.head())

	#stack=np.array(stack)   
	Metamodel = load('Metamodel.joblib')
	prediction=Metamodel.predict(df_pred)
	print(prediction[0])

	return jsonify({'meter_reading': str(prediction[0])})
    


# In[ ]:


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

