import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))
pkl_file = open('bust_size_encoder.pkl', 'rb')
le_bust_size = pickle.load(pkl_file) 

#test_df['bust_size'] = le_bust_size.fit_transform(test_df['bust_size'])
#pkl_file.close()

pkl_file = open('body_type_encoder.pkl', 'rb')
le_body_type = pickle.load(pkl_file) 

#test_df['body_type'] = le_body_type.fit_transform(test_df['body_type'])
#pkl_file.close()

pkl_file = open('category_encoder.pkl', 'rb')
le_category = pickle.load(pkl_file) 

#test_df['category'] = le_category.fit_transform(test_df['category'])
#pkl_file.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    print(int_features)
    data = {'bust_size':[int_features[0]],
            'weight':[int_features[1]],
    'body_type':[int_features[2]],
    'category':[int_features[3]],
    'height':[int_features[4]],
    'age':[int_features[5]]}
     
    # Create DataFrame
    test_df = pd.DataFrame(data)
    print(test_df)
    bust_size1 = int_features[0]
    print('size',bust_size1)
    test_df['bust_size'] = le_bust_size.fit_transform(test_df['bust_size'])
    #int_features[0]=le_bust_size.fit_transform(bust_size1)
    weight = request.form.get("weight")
    test_df['weight'] = int_features[1]#weight
    #bust_size = request.form.get("bust_size")
    body_type1 = request.form.get("body_type")
    test_df['body_type'] = le_body_type.fit_transform(test_df['body_type'])
    category1 = request.form.get("category")
    test_df['category'] = le_category.fit_transform(test_df['category'])
    height = request.form.get("height")
    test_df['height'] = int_features[4]#height
    age = request.form.get("age")
    test_df['age'] = int_features[5]#age
    final_features = [np.array(int_features)]
    print(test_df)
    prediction = model.predict(test_df)

    output = prediction[0]

    return render_template('index.html', prediction_text='Dress Size should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)