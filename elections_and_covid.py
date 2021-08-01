import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
server = app.server

########
confirmed = pd.read_csv(
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
deaths = pd.read_csv(
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")

country_list = confirmed['Country/Region'].tolist()

confirmed_a = confirmed.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])
conf_col = confirmed_a.columns.tolist()
confirmed_diff_a = np.diff(confirmed_a.values, axis=1)
confirmed_diff_a = pd.DataFrame(confirmed_diff_a, columns=conf_col[1:], index=country_list)


# reset_index
confirmed_diff_a = confirmed_diff_a.reset_index()
confirmed_diff_a = confirmed_diff_a.rename(columns={'index': 'Country'})
# confirmed_diff_a.head()


deaths_a = deaths.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])
death_col = deaths_a.columns.tolist()
deaths_diff_a = np.diff(deaths_a.values, axis=1)
deaths_diff_a = pd.DataFrame(deaths_diff_a, columns=death_col[1:], index=country_list)
# deaths_diff_a.head()


# reset_index
deaths_diff_a = deaths_diff_a.reset_index()
deaths_diff_a = deaths_diff_a.rename(columns={'index': 'Country'})
# deaths_diff_a.head()


##Importing the elections data
elections_data = pd.read_excel("COVID and ELECTIONS.xlsx")


elections_cleaned = elections_data.copy()
elections_cleaned['TYPE OF ELECTION'] = np.where(elections_cleaned['TYPE OF ELECTION'].str.contains('rimary'),
                                                 'Primary elections', elections_cleaned['TYPE OF ELECTION'])
elections_cleaned['TYPE OF ELECTION'] = np.where(elections_cleaned['TYPE OF ELECTION'].str.contains('eneral'),
                                                 'General elections', elections_cleaned['TYPE OF ELECTION'])
elections_cleaned['COUNTRY'] = elections_cleaned['COUNTRY'].str.lstrip()
elections_cleaned['COUNTRY'] = elections_cleaned['COUNTRY'].str.rstrip()
elections_cleaned['TYPE OF ELECTION'] = elections_cleaned['TYPE OF ELECTION'].str.lstrip()
elections_cleaned['TYPE OF ELECTION'] = elections_cleaned['TYPE OF ELECTION'].str.rstrip()
# elections_cleaned.head()


covid_elections = elections_cleaned.merge(confirmed_diff_a, left_on='COUNTRY', right_on='Country', how='left')
covid_elections = covid_elections[covid_elections['Country'].notna()]


covid_elections_d = elections_cleaned.merge(deaths_diff_a, left_on='COUNTRY', right_on='Country', how='left')
covid_elections_d = covid_elections_d[covid_elections_d['Country'].notna()]

part1 = pd.melt(covid_elections,
                id_vars=['COUNTRY_raw', 'TYPE OF ELECTION', 'DATE OF ELECTION', 'Start_date', 'End_date', 'Country'],
                value_vars=conf_col[1:])
part1.variable = part1.variable.astype('datetime64')
part1 = part1.rename(columns={'value': 'confirmed'})


part2 = pd.melt(covid_elections_d,
                id_vars=['COUNTRY_raw', 'TYPE OF ELECTION', 'DATE OF ELECTION', 'Start_date', 'End_date', 'Country'],
                value_vars=death_col[1:])
part2.variable = part2.variable.astype('datetime64')
part2 = part2.rename(columns={'value': 'deaths'})
part2 = part2[['Country', 'variable', 'deaths']]


part1.variable.max()
part1["only_date"] = pd.to_datetime(part1['variable'], format='%m%y', errors='coerce')

part1["only_date"] = part1["variable"].dt.strftime('%m/%Y')

part1['month'], part1['year'] = part1['only_date'].str.split('/', 1).str
part1["only_date"] = pd.to_datetime(part1[["year", 'month']].assign(day=1))

part1_sub = part1[['only_date', 'month']]
part1_sub = part1_sub.drop_duplicates(subset=['only_date'])
part1_sub['month_slider'] = np.arange(0, len(part1_sub)).tolist()
part1_sub.head()
part1 = part1.drop(columns=['month', 'year'])
part1 = part1.merge(part1_sub, on='only_date', how='left')


numdate = [x for x in range(len(part1['only_date'].unique()))]

g_data = part1.merge(part2, on=['Country', 'variable'], how='left')
# dropping negatives
g_data = g_data[g_data['confirmed'] >= 0]
g_data = g_data[g_data['deaths'] >= 0]


# Max number od cases per cat
max_conf = g_data.sort_values(by=['TYPE OF ELECTION', 'Country', 'confirmed'], ascending=False)
max_conf = max_conf.drop_duplicates(subset=['TYPE OF ELECTION', 'Country'], keep='first')
max_conf = max_conf[['TYPE OF ELECTION', 'Country', 'confirmed']]
max_conf = max_conf.rename(columns={'confirmed': 'max_confirmed'})

# Max number of deaths per cat
max_death = g_data.sort_values(by=['TYPE OF ELECTION', 'Country', 'deaths'], ascending=False)
max_death = max_death.drop_duplicates(subset=['TYPE OF ELECTION', 'Country'], keep='first')
max_death = max_death[['TYPE OF ELECTION', 'Country', 'deaths']]
max_death = max_death.rename(columns={'deaths': 'max_death'})

max_set = max_conf.merge(max_death, on=['TYPE OF ELECTION', 'Country'], how='left')


g_data = g_data.merge(max_set, on=['TYPE OF ELECTION', 'Country'], how='left')


g_data_zero = g_data.copy()
g_data_zero = g_data_zero.drop_duplicates(subset=['TYPE OF ELECTION', 'Country'], keep='first')
g_data_zero['max_confirmed'] = 0
g_data_zero['max_death'] = 0
g_data_zero['variable'] = np.nan
g_data = g_data.append(g_data_zero)

# creating the app name
app.layout = html.Div((
    # header row
    html.Div([
        html.Div([
            html.Div([
                html.H1('Covid and Elections (CDDEP)', style={'margin-bottom': '3px', 'textAlign': 'center'}),
            ])

        ], className='six column', id="Title")
    ], id='header', className='row flex-display', style={'margin-bottom': '25px'}),

    # creating the region dropdown
    html.Div([
        html.P("Select country:"),
        dcc.Dropdown(id='region',
                     multi=False,
                     clearable=True,
                     disabled=False,
                     style={'display': True},
                     placeholder='Select country',
                     options=[{'label': i, 'value': i} for i in g_data.Country.unique()])
    ], className='one row', style={'width': '48%', 'display': 'inline-block', 'margin': '4px'}),
    html.Div([
        html.P("Select Election Type:"),
        dcc.Dropdown(id='country',
                     multi=False,
                     clearable=True,
                     disabled=False,
                     style={'display': True},
                     placeholder='Election type',
                     options=[])
    ], className='one row', style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'margin': '4px'}),

    # Graph holder
    html.Div(className='row', children=[
        dcc.Graph(
            id='graph1', style={"width": 650, "margin": 0, 'display': 'inline-block'}),
        dcc.Graph(
            id='graph2', style={"width": 650, "margin": 0, 'display': 'inline-block'})]),

    # slider
    html.Div([
        html.Div([
            dcc.RangeSlider(id='year_range',
                            min=numdate[0],  # the first date
                            max=numdate[-1],  # the last date
                            value=[0, 20],  # default: the first
                            marks={numd: date.strftime('%b/%y') for numd, date in
                                   zip(numdate, g_data['only_date'].dt.date.unique())}
                            ),
        ], style={'width': '98%', 'margin-left': '1%'}),
    ], className='row flex-display'),

))


##Callback for countries
@app.callback(Output('country', 'options'),
              [Input('region', 'value')]
              )
def variable_options0(region):
    dff1 = g_data[g_data.Country == region]
    return [{'label': i, 'value': i} for i in dff1['TYPE OF ELECTION'].unique()]


##Creating the first graph

@app.callback(Output('graph1', 'figure'),
              [Input('region', 'value'),
               Input('country', 'value'),
               Input('year_range', 'value')]
              )
def update_graph(region, country, year_range):
    gr1_data = g_data[(g_data['Country'] == region) & (g_data['TYPE OF ELECTION'] == country) & (
                g_data['month_slider'] >= year_range[0]) & (g_data['month_slider'] <= year_range[1])]

    # grx_data=gr1_data.drop_duplicates(by=['COUNTRY_raw','TYPE OF ELECTION','DATE OF ELECTION','Start_date','End_date','Country''variable','confirmed','only_date','month','month_slider','deaths'])
    # gr1_data=gr1_data[['year','valuex']]
    # s=gr1_data['DATE OF ELECTION'].iloc[1]+''
    trace1 = go.Scatter(x=gr1_data['variable'],
                        y=gr1_data['confirmed'],
                        mode='lines',
                        line=dict(color='orange'),
                        name='Confirmed cases',
                        showlegend=True,
                        )

    trace2 = go.Scatter(x=gr1_data['Start_date'],
                        y=gr1_data['max_confirmed'],
                        mode='lines',
                        line=dict(color='blue'),
                        name='Election Date',
                        showlegend=True
                        )

    return {
        'data': [trace1, trace2],
        'layout': go.Layout(title=country,
                            xaxis_title="Date",
                            yaxis_title="No. of cases",
                            hovermode='closest')
        #         'layout':go.Layout(title={'text':gr1_data['DATE OF ELECTION'].tolist()[0]},
        #                     xaxis={'title': 'Days'},
        #                     yaxis={'title': ''},
        #                     hovermode = 'closest')
    }


##Creating the second graph

@app.callback(Output('graph2', 'figure'),
              [Input('region', 'value'),
               Input('country', 'value'),
               Input('year_range', 'value')]
              )
def update_graph(region, country, year_range):
    gr2_data = g_data[(g_data['Country'] == region) & (g_data['TYPE OF ELECTION'] == country) & (
                g_data['month_slider'] >= year_range[0]) & (g_data['month_slider'] <= year_range[1])]
    # gr1_data=gr1_data[['year','valuex']]
    # s=gr1_data['DATE OF ELECTION'].iloc[1]+''
    trace3 = go.Scatter(x=gr2_data['variable'],
                        y=gr2_data['deaths'],
                        mode='lines',
                        line=dict(color='red'),
                        name='Confirmed deaths',
                        showlegend=True,
                        )

    trace4 = go.Scatter(x=gr2_data['Start_date'],
                        y=gr2_data['max_death'],
                        mode='lines',
                        line=dict(color='blue'),
                        name='Election Date',
                        showlegend=True
                        )

    return {
        'data': [trace3, trace4],
        'layout': go.Layout(title=country,
                            xaxis_title="Date",
                            yaxis_title="No. of deaths",
                            hovermode='closest')
        #         'layout':go.Layout(title={'text':gr1_data['DATE OF ELECTION'].tolist()[0]},
        #                     xaxis={'title': 'Days'},
        #                     yaxis={'title': ''},
        #                     hovermode = 'closest')
    }

if __name__ == '__main__':
    app.run_server(debug=True)