import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
import numpy as np
#import glob

from collections import Counter

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)

#material_names=glob.glob('/data/*.csv')
#df = pd.read_csv('data/dftrim.csv')
#dfc=pd.read_csv('data/dfctrim.csv')
substrate_materials=['None','C','Si']
substrate_thickness=['500','750','1000']
grating_materials=['W','Au','Al','Au80Sn20','Cu','Steel']
grating_thickness=['100','150','200','250','300','500','1000']
attenuator_materials=['None','Al','Ti','Steel','Be']
attenuator_thickness=['1','2']
detector_materials=['None','CdTe']
detector_thickness=['1','2']


def calc_transmission(a, rho, thickness):
    '''Make this seperate just in case'''
    thick=float(thickness)*10**-4
    transmission = [np.exp(-1*x*thick*rho) for x in a]
    return transmission

def get_transm(substrate,sthick,grating,gthick,attenuator,athick,detector,dthick):
    transm_dict={}
    gdata=pd.read_csv('data/'+grating+'.csv',sep=',',header=0)
    genergy=gdata['E (keV)']
    #try to see if it's pre-calculated first
    try:
        gtransm = gdata['P(xi) (d='+gthick+' um)']
    except KeyError:
        gtransm=calc_transmission(gdata['a (cm^2/g)'],gdata['rho (g/cm^3)'][0],gthick)
    transm_dict['genergy']=genergy
    transm_dict['gtransm']=gtransm*0.5
    
    if substrate !="None":
        sdata=pd.read_csv('data/'+substrate+'.csv',sep=',',header=0)
        senergy=sdata['E (keV)']
        #try to see if it's pre-calculated first
        try:
            stransm = sdata['P(xi) (d='+sthick+' um)']
        except KeyError:
            stransm=calc_transmission(sdata['a (cm^2/g)'],sdata['rho (g/cm^3)'][0],sthick)
        transm_dict['senergy']=senergy
        transm_dict['stransm']=np.array(stransm)*0.5
    else:
        transm_dict['senergy']=genergy
        transm_dict['stransm']=np.ones(len(genergy))

    if attenuator !="None":
        adata=pd.read_csv('data/'+attenuator+'.csv',sep=',',header=0)
        aenergy=adata['E (keV)']
        #try to see if it's pre-calculated first
        try:
            atransm = adata['P(xi) (d='+athick+'000 um)']
        except KeyError:
            atransm=calc_transmission(adata['a (cm^2/g)'],adata['rho (g/cm^3)'][0],athick)
        transm_dict['aenergy']=aenergy
        transm_dict['atransm']=atransm
    else:
        transm_dict['aenergy']=genergy
        transm_dict['atransm']=np.ones(len(genergy))

    if detector !="None":
        ddata=pd.read_csv('data/'+detector+'.csv',sep=',',header=0)
        denergy=ddata['E (keV)']
        #try to see if it's pre-calculated first
        try:
            dtransm = ddata['P(xi) (d='+dthick+'000 um)']
        except KeyError:
            dtransm=calc_transmission(ddata['a (cm^2/g)'],ddata['rho (g/cm^3)'][0],dthick)
        transm_dict['denergy']=denergy
        transm_dict['dtransm']=1.-dtransm
    else:
        transm_dict['denergy']=genergy
        transm_dict['dtransm']=np.ones(len(genergy))

    #interpolate everything to the same energy range
    evector = np.linspace(2.08, 433, num=400, endpoint=True) #check that this spans the correct rang

    sinterp=np.interp(evector,transm_dict['senergy'],transm_dict['stransm'])
    ginterp=np.interp(evector,transm_dict['genergy'],transm_dict['gtransm'])
    ainterp=np.interp(evector,transm_dict['aenergy'],transm_dict['atransm'])
    dinterp=np.interp(evector,transm_dict['denergy'],transm_dict['dtransm'])
    #        print(k,transm_dict[k+'energy'])
    #        print(k,transm_dict[k+'transm'])

    transm_dict['tenergy']=evector
    transm_dict['ttransm']=(sinterp+ginterp)*ainterp*dinterp
        #print(np.shape(transm_dict['tenergy']))
        #print(np.shape(transm_dict['ttransm']))
    return transm_dict

app.layout = html.Div(children=[
    html.H1(children='X-Ray Grating Transmission Calculator'),
        html.Div(className='row',children=[
            html.Div(className='three columns div-user-controls',
            children=[
            html.H2(children='Components and Materials'),
            html.H3("Substrate"),
            dcc.Dropdown(
                id='substrate',
                options=[{'label': i, 'value': i} for i in substrate_materials],
                value='C'),
            dcc.Dropdown(
                id='sthick',
                options=[{'label': i + ' microns', 'value': i} for i in substrate_thickness],
                value='1000'
            ),
            html.H3("Gratings"),
            dcc.Dropdown(
                id='gratings',
                options=[{'label': i, 'value': i} for i in grating_materials],
                value='Au'),
            dcc.Dropdown(
                id='gthick',
                options=[{'label': i+ ' microns', 'value': i} for i in grating_thickness],
                value='250'
            ),
            html.H3("Attenuator"),
            dcc.Dropdown(
                id='attenuator',
                options=[{'label': i, 'value': i} for i in attenuator_materials],
                value='None'),
            dcc.Dropdown(
                id='athick',
                options=[{'label': i + ' mm', 'value': i} for i in attenuator_thickness]
            ),
            html.H3("Detector"),
            dcc.Dropdown(
                id='detector',
                options=[{'label': i, 'value': i} for i in detector_materials],
                value='None'),
            dcc.Dropdown(
                id='dthick',
                options=[{'label': i+' mm', 'value': i} for i in detector_thickness]
            )

                    ],style={'height': '100%', 'display': 'inline-block'}),  # Define the left element
            html.Div(className='nine columns div-for-charts bg-grey',
            children =[dcc.Graph(id='transmission')],style={'height': '100%', 'display': 'inline-block'})
        ]),
    html.Div(className='row',children=[
    html.Div(className='three columns div-user-controls',
    children=[html.H2(children='Expected Counts for C-class flare'),
    dcc.RadioItems(
        id='bin_type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )]),

    html.Div(className='nine columns div-for-charts bg-grey',
            children =[dcc.Graph(id='flare_counts')],style={'height': '100%', 'display': 'inline-block'})
        ]),
    html.Div(className='row',children=[
        html.P("2020, Erica Lastufka"),
        html.P("elastufka@gmail.com")])
])


############ Transmission ###############
@app.callback(
    Output('transmission', 'figure'),
    [Input('substrate', 'value'),
    Input('sthick', 'value'),
    Input('gratings', 'value'),
    Input('gthick', 'value'),
    Input('attenuator', 'value'),
    Input('athick', 'value'),
    Input('detector', 'value'),
    Input('dthick', 'value')])

def update_graph(substrate,sthick,gratings,gthick,attenuator,athick,detector,dthick):

    transm_dict=get_transm(substrate,sthick,gratings,gthick,attenuator,athick,detector,dthick)

    fig = go.Figure()
    if substrate != 'None':
        fig.add_trace(go.Scatter(x=transm_dict['senergy'], y=transm_dict['stransm'], name='Substrate'))
    fig.add_trace(go.Scatter(x=transm_dict['genergy'], y=transm_dict['gtransm'], name='Gratings'))
    if attenuator != 'None':
        fig.add_trace(go.Scatter(x=transm_dict['aenergy'], y=transm_dict['atransm'], name='Attenuator'))
    if detector != 'None':
        fig.add_trace(go.Scatter(x=transm_dict['denergy'], y=transm_dict['dtransm'], name='Detector'))
    fig.add_trace(go.Scatter(x=transm_dict['tenergy'], y=transm_dict['ttransm'], name='Total'))
    fig.update_layout(title='Transmission',yaxis_type = 'log',xaxis_type='log',xaxis_range=[.5,2.5],xaxis_title='Energy (keV)',yaxis_range=[-3,0.1],yaxis_title='Percent Transmission')
    #fig.update_traces(marker=dict(colors=ccolors))
    return fig


#######################################

############ Flare Counts ###############
@app.callback(
    Output('flare_counts', 'figure'),
    [Input('bin_type', 'value'),
    Input('substrate', 'value'),
    Input('sthick', 'value'),
    Input('gratings', 'value'),
    Input('gthick', 'value'),
    Input('attenuator', 'value'),
    Input('athick', 'value'),
    Input('detector', 'value'),
    Input('dthick', 'value')])

def update_graph(bin_type,substrate,sthick,gratings,gthick,attenuator,athick,detector,dthick):
    transm_dict=get_transm(substrate,sthick,gratings,gthick,attenuator,athick,detector,dthick)
    
    dist = pd.read_csv('data/flare_xr_dist.csv') #[energy, thermal part, non-thermal part]
    genergy=transm_dict['tenergy']
    prob = np.interp(genergy,dist['Energy'],dist['Dist'])
    ntnt = np.interp(genergy,dist['Energy'],dist['NT'])
    thth = np.interp(genergy,dist['Energy'],dist['TH'])

    total_counts= prob*transm_dict['ttransm']
    thermal_counts= thth*transm_dict['ttransm']
    nonthermal_counts= ntnt*transm_dict['ttransm']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dist['Energy'],y=thermal_counts,name='thermal',line= {"shape": 'hv'},
  mode= 'lines',type='scatter'))
    fig.add_trace(go.Scatter(x=dist['Energy'],y=nonthermal_counts,name='non-thermal',line= {"shape": 'hv'},
  mode= 'lines',type='scatter'))
    fig.add_trace(go.Scatter(x=dist['Energy'],y=prob,name='Total',line= {"shape": 'hv'},
  mode= 'lines',type='scatter'))
    fig.update_layout(title='Flare Counts',yaxis_type = 'log',xaxis_type='log',xaxis_range=[.5,2.5],xaxis_title='Energy (keV)',yaxis_range=[0,5],yaxis_title='Predicted Flare Counts')

    return fig


#######################################


if __name__ == '__main__':
    app.run_server(debug=True)
