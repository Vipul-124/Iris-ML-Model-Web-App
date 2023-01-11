import pickle
import streamlit as st

# Load the trained ML Models
log_reg = pickle.load(open('lr_model.pkl', 'rb'))
decision_tree = pickle.load(open('dt_model.pkl', 'rb'))
random_forest = pickle.load(open('rf_model.pkl', 'rb'))

st.title('Iris ML Web App')

ml_model = ['Logistic Regression', 'DecisionTree Classifier', 'RandomForest Classifier']
option = st.sidebar.selectbox('Select the ML Model which you want to use', ml_model)

# ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']


sepal_length = st.slider('Select Sepal Length', 0.0, 10.0)
sepal_width = st.slider('Select Sepal Width', 0.0, 10.0)
petal_length = st.slider('Select Petal Length', 0.0, 10.0)
petal_width = st.slider('Select Petal Width', 0.0, 10.0)

test = [[sepal_length, sepal_width, petal_length, petal_width]]
st.write('Test_Data', test)

if st.button('Classify') :
    if option == 'Logistic Regression' :
        st.success(log_reg.predict(test)[0])
    elif option == 'DecisionTree Classifier' :
        st.success(decision_tree.predict(test)[0])
    else :
        st.success(random_forest.predict(test)[0])

# Terminal Commands
# To run Streamlit Web App - streamlit run appname.py
# To stop the server - Ctrl + C