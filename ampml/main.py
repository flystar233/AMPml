#!/usr/bin/env Python3      
import PySimpleGUI as sg      
from amp import *
sg.LOOK_AND_FEEL_TABLE['MyTheme'] = {'BACKGROUND': '#f2f4f6',
                                        'TEXT': '#313131',
                                        'INPUT': '#c7e78b',
                                        'TEXT_INPUT': '#000000',
                                        'SCROLL': '#c7e78b',
                                        'BUTTON': ('#2C2C2C', '#96ceb4'),
                                        'PROGRESS': ('#01826B', '#D0D0D0'),
                                        'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0,
                                        }
sg.theme('MyTheme')      
      
# ------ Column Definition ------ #      
column1 = [[sg.Text('Column 1', background_color='#F7F3EC', justification='center', size=(10, 1))],      
            [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 1')],      
            [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 2')],      
            [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 3')]]      

train_layout = [
    [sg.Radio('train',  "RADIO1", font=("Helvetica", 11),size=(12, 1))],
    [sg.Checkbox('test trees', size=(10,1)),  sg.Checkbox('feature importance')],
    [sg.Text('ML methods',size=(15,1)), sg.Drop(values=('RF', 'bayes','SVM','GT'))],
    [sg.Text('seed',size=(15,1)), sg.Spin(values=[i for i in range(1, 10000)], initial_value=2020)],
    [sg.Text('representation',size=(15,1)),sg.Drop(values=('AAC','CTDD','PAAC'))],
    [sg.Text('num-trees',size=(15,1)), sg.Spin(values=[i for i in range(1, 500)], initial_value=100)],
    [sg.Text('_'*40)],
    [sg.Text('Choose files',font=("Helvetica", 12))],
    [sg.Text('positive file',size=(12, 1)), sg.Input(), sg.FileBrowse()],
    [sg.Text('negative file',size=(12, 1)), sg.Input(), sg.FileBrowse()],
    [sg.Text('drop features file',size=(12, 1)), sg.Input(), sg.FileBrowse()],
    [sg.Text('Train result',font=("Helvetica", 12))],  
    [sg.Multiline('', key='result',size=(60,5))]
]

predict_layout = [
    [sg.Radio('prediction',  "RADIO1", font=("Helvetica", 11))],
    [sg.Text('seed',size=(12,1)), sg.Spin(values=[i for i in range(1, 10000)], initial_value=2020)],
    [sg.Text('representation',size=(12,1)),sg.Drop(values=('AAC','CTDD','PAAC'))],
    [sg.Text('threshold',size=(12,1)),sg.InputText('')],
    [sg.Text('_'*40)],
    [sg.Text('Choose files',font=("Helvetica", 12))], 
    [sg.Text('model file',size=(12,1)), sg.Input(), sg.FileBrowse()],
    [sg.Text('input sequences',size=(12,1)), sg.Input(), sg.FileBrowse()],
    [sg.Text('drop features file',size=(12,1)), sg.Input(), sg.FileBrowse()]
]

# 将标题放在一个 sg.Column 中，并设置水平居中对齐
title_column = [[sg.Text('AMP prediction by ML', justification='center', font=("Consolas", 15), relief=sg.RELIEF_RIDGE)]]

layout = [
    [sg.Column(title_column, justification='center')],  # 使用 sg.Column 并设置对齐方式
    [sg.TabGroup([[sg.Tab('Train', train_layout), sg.Tab('Predict', predict_layout)]])],
    [sg.Button('Calculate'),sg.Button('Cancel')]
]

window = sg.Window('AMPml', layout, default_element_size=(40, 1), grab_anywhere=False)
while  True:
    event, values = window.read()
    if event == 'Calculate':
        new_value = {}
        if values[0]:
            new_value['tree_test']=values[1]
            new_value['feature_importance']=values[2]
            new_value['method']=values[3]
            new_value['seed']=values[4]
            new_value['representation']=values[5]
            new_value['num_trees']=values[6]
            new_value['positive']=values['Browse']
            new_value['negative']=values['Browse0']
            new_value['drop_feature']=values['Browse1']
            new_value = DottableDict(new_value)
            result = train(new_value)
            if new_value['method'] == 'RF':
                result = f'Out-of-bag accuracy:\t{result[0]}\nOut-of-bag balanced accuracy:\t{result[1]}\n\
                AUC Score:\t{result[2]}\nconfusion matrix 0:\t{result[3][0]}\nconfusion matrix 1:\t{result[3][1]}'
            elif new_value['method'] == 'SVM':
                result = f"Cross-validation accuracy : {result[0]}\nconfusion matrix 0:\t{result[1][0]}\nconfusion matrix 1:\t{result[1][1]}"
            elif new_value['method'] == 'bayes':
                result = f"Cross-validation accuracy : {result[0]}\nconfusion matrix 0:\t{result[1][0]}\nconfusion matrix 1:\t{result[1][1]}"
            elif new_value['method'] == 'GT':
                result = f"Cross-validation accuracy : {result[0]}\nconfusion matrix 0:\t{result[1][0]}\nconfusion matrix 1:\t{result[1][1]}"
            if result:
                window.Element('result').Update(result, text_color='black')
        else:
            new_value['seed']=values[11]
            new_value['representation']=values[12]
            new_value['threshold']=values[13]
            new_value['model']=values['Browse2']
            new_value['seq_file']=values['Browse3']
            new_value['drop_feature']=values['Browse4']
            new_value = DottableDict(new_value)
            predict(new_value)
            sg.popup("The AMP prediction file: AMPpred.tsv has saved in current path!")
    elif event == 'Cancel':
        break
    else:
        break
window.Close()
