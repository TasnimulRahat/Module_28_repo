import gradio as gr
#date,gender,age,address,famsize,Pstatus,M_Edu,F_Edu,M_Job,F_Job,relationship,smoker,tuition_fee,time_friends,ssc_result,hsc_result add "" to all these in the next line
#'gender','age','address','famsize','Pstatus','M_Edu','F_Edu','M_Job','F_Job','relationship','smoker','tuition_fee','time_friends','ssc_result'

#pandas
import pandas as pd
#pkl
import pickle
#np
import numpy as np
#load model
with open('student_rf_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)
def predict_gpa(gender,age,address,famsize,Pstatus,M_Edu,F_Edu,
M_Job,F_Job,relationship,smoker,tuition_fee,time_friends,ssc_result):
    # Create a DataFrame for the input data
    input_df = pd.DataFrame(
        [[gender,age,address,famsize,Pstatus,M_Edu,F_Edu,M_Job,F_Job,relationship,
    smoker,tuition_fee,time_friends,ssc_result]],
    
    columns =['gender','age','address','famsize','Pstatus','M_Edu','F_Edu',
    'M_Job','F_Job','relationship','smoker','tuition_fee','time_friends','ssc_result'])
    #predict
    prediction = model.predict(input_df)[0]
    return f'Predicted HSC GPA: { np.clip(prediction, 0, 5) :.2f}'


inputs=[
    gr.Radio(choices=['M', 'F'], label='Gender'),
    gr.Number(label='Age',value=18),
    gr.Radio(choices=['Urban', 'Rural'], label='Address'),
    gr.Radio(choices=['GT3','LE3'], label='Family Size'),
    gr.Radio(choices=['Together', 'Apart'], label='Parental Status'),
    gr.Slider(minimum=0, maximum=4, step=1, label='Mother Education'),
    gr.Slider(minimum=0, maximum=4, step=1, label='Father Education'),
    gr.Dropdown(choices=['Teacher', 'Health', 'Services', 'At_home', 'Other'], label='Mother Job'),
    gr.Dropdown(choices=['Teacher', 'Health', 'Services', 'Business','Farmer', 'Other'], label='Father Job'),
    #added relationship and smoker
    gr.Radio(choices=['Yes', 'No'], label='Relationship'),
    gr.Radio(choices=['Yes', 'No'], label='Smoker'),
    gr.Number(label='Tuition Fee'),
    gr.Slider(minimum=0, maximum=5, step=1, label='Time with Friends'),
    gr.Number(label='SSC Result'),
]

#interface
app = gr.Interface(
    fn=predict_gpa,
    inputs=inputs,
    outputs='text',
    title='Student HSC GPA Prediction',
)

#launching with share=True for public access
app.launch(share=True)