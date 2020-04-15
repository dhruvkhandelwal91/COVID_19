import pandas as pd
import requests
from requests.exceptions import HTTPError
from datetime import datetime
from bs4 import BeautifulSoup
import re
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import random
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

pd.set_option('max_columns', 15)


# Get data from ECDC website and return as dataframe
def get_data():
    # Get excel filename from ECDC website
    url_ecdc = 'https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide'
    try:
        source = requests.get(url_ecdc).text
    except HTTPError:
        print('The ECDC website is currently being updated with the latest data. Try again after some time.')
        exit()
    soup = BeautifulSoup(source, 'lxml')
    url_data = soup.find(href=re.compile('xlsx')).get('href')
    # Import data from scraped excel url
    df = pd.read_excel(url_data, encoding='cp1252')
    return df


# Pre-process data
def preprocess_data(df):
    outlier_thresh_pop = 100000# Threshold for removing outliers e.g., low population countries
    df = df.sort_values(['countriesAndTerritories', 'dateRep'], ascending=True, axis=0)
    df_country_grp = df.groupby('countriesAndTerritories')

    # compute cumulative cases in each country and normalize by population
    df['cumulativeCases'] = df_country_grp['cases'].cumsum()
    df['cumulativeCasesPerMillion'] = df['cumulativeCases'].divide(df['popData2018']) * 1000000

    # compute cumulative deaths in each country and normalize by population
    df['cumulativeDeaths'] = df_country_grp['deaths'].cumsum()
    df['cumulativeDeathsPerMillion'] = df['cumulativeDeaths'].divide(df['popData2018']) * 1000000

    # Country names as labels
    df['countryNames'] = df['countriesAndTerritories'].transform(
        lambda x: x.replace('_', ' '))
    # Remove outliers
    df = df.loc[df['popData2018'] > outlier_thresh_pop]
    # Dates for labels
    df_dates = pd.to_datetime(df['dateRep'].unique()).sort_values()
    return df, df_dates


# Apply threshold to data grouped by country and introduce a country-specific relative date index
def thresh_grouped_data(ungrouped_df, group_col, thresh_col, thresh, idx=False, idx_col_base='date',
                        idx_name='idx'):
    grouped_df = ungrouped_df.groupby(group_col)
    filt_thresh = grouped_df[thresh_col].apply(lambda x: x >= thresh)
    df_filtered = ungrouped_df.loc[filt_thresh]

    # If country-specific relative date index is requested
    if idx:
        idx_series = df_filtered.groupby(group_col)[idx_col_base].transform(
            lambda x: (x - min(x)))
        idx_series = idx_series.transform(lambda x: x.days)
        idx_series.rename(idx_name, inplace=True)
        df_filtered = pd.concat([df_filtered, idx_series], axis=1).sort_values([group_col, idx_name])
    return df_filtered


def generate_random_colors(num_colors=1, seed=7):
    random.seed(seed)
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num_colors)]
    return color


# Generate map plot of COVID-19 data with options: cases or deaths, normalized or un-normalized
def generate_COVID_map(df, col_name, date):
    fig = px.choropleth(df.loc[df['dateRep'] == date], locations="countryterritoryCode",
                        color=col_name,
                        range_color=(min(df.loc[df['dateRep'] == date, col_name]),
                                     max(df.loc[df['dateRep'] == date, col_name])),
                        hover_name="countryNames",  # column to add to hover information
                        color_continuous_scale='Viridis',
                        projection="natural earth"
                        )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


# Generate line plots of COVID-19 cases and deaths per country with options: normalized or un-normalized, linear or log
def generate_COVID_plot(df, countries_of_interest, normalize):
    if normalize == 'Population':
        thresh_cases = 10
        thresh_deaths = 0.5
        col_cases = 'cumulativeCasesPerMillion'
        col_deaths = 'cumulativeDeathsPerMillion'
    elif normalize == 'None':
        thresh_cases = 100
        thresh_deaths = 10
        col_cases = 'cumulativeCases'
        col_deaths = 'cumulativeDeaths'
    else:
        thresh_cases = 100
        thresh_deaths = 10
        col_cases = 'cumulativeCases'
        col_deaths = 'cumulativeDeaths'
    countries_of_interest.sort()
    match = df['countriesAndTerritories'].isin(countries_of_interest)
    df_filtered_countries = df.loc[match]
    df_filtered_cases = thresh_grouped_data(df_filtered_countries, 'countriesAndTerritories', col_cases,
                                            thresh_cases, idx=True, idx_col_base='dateRep', idx_name='numDays')
    df_filtered_deaths = thresh_grouped_data(df_filtered_countries, 'countriesAndTerritories', col_deaths,
                                             thresh_deaths, idx=True, idx_col_base='dateRep', idx_name='numDays')

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Number of confirmed cases', 'Number of confirmed deaths'),
                        horizontal_spacing=0.1, vertical_spacing=0.1)
    colormap = generate_random_colors(len(countries_of_interest))
    cnt = 0
    for key, grp in df_filtered_cases.groupby('countriesAndTerritories'):
        fig.add_trace(go.Scatter(x=grp['numDays'].tolist(), y=grp[col_cases].tolist(),
                                 mode='lines+markers', name=key.replace('_', ' '),
                                 legendgroup=f'group{cnt}', marker_color=colormap[cnt],
                                 text=[f'{country}<br>Day: {xval}<br>Cases: {yval}' for country, xval, yval in
                                       zip(grp['countryNames'].tolist(), grp['numDays'].tolist(),
                                           grp[col_cases].tolist())],
                                 hoverinfo='text'
                                 ),
                      row=1, col=1)
        cnt = cnt + 1
    cnt = 0
    for key, grp in df_filtered_deaths.groupby('countriesAndTerritories'):
        fig.add_trace(go.Scatter(x=grp['numDays'].tolist(), y=grp[col_deaths].tolist(),
                                 mode='lines+markers', name=key.replace('_', ' '),
                                 legendgroup=f'group{cnt}', marker_color=colormap[cnt], showlegend=False,
                                 text=[f'{country}<br>Day: {xval}<br>Deaths: {yval}' for country, xval, yval in
                                       zip(grp['countryNames'].tolist(), grp['numDays'].tolist(),
                                           grp[col_deaths].tolist())],
                                 hoverinfo='text'
                                 ),
                      row=2, col=1)
        cnt = cnt + 1
    return fig


df = get_data()
df, df_dates = preprocess_data(df)
countries_of_interest = ['Netherlands', 'India', 'China', 'Italy', 'Spain', 'South_Korea',
                         'United_States_of_America', 'Germany']

# Plotly app for visualizing data

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='COVID-19 DATA',
            style={
                'textAlign': 'center'
            }),

    html.Div(children='''
        The following is a visualization of the latest COVID-19 data reported by ECDC.
    ''',
             style={
                 'textAlign': 'center'
             }
             ),
    html.Article(children=[

        html.H2(children='Map illustration'),

        html.Div([

            html.Div([
                dcc.Graph(
                    id='COVID_19_map',
                    # style={'height': '2100x', 'width': '1500px'}
                ),
                dcc.Slider(
                    id='date_slider',
                    min=0,
                    max=len(df_dates) - 1,
                    value=len(df_dates) - 1,
                    marks={date_idx: df_dates[date_idx].strftime('%d-%b-%y') for date_idx in
                           range(0, len(df_dates), 15)},
                    step=1
                )
            ],
                style={
                    'display': 'block'
                }
            ),

            html.Div([

                html.Div([
                    html.Label('Quantity on map '),

                    dcc.RadioItems(
                        id='map_qty',
                        options=[{'label': i, 'value': i} for i in ['Cases', 'Deaths']],
                        value='Deaths',
                        inputStyle={'margin-right': '0.4em'},
                        labelStyle={
                            'margin-right': '0.9em',
                            'display': 'block'
                        }
                        # labelStyle={'display': 'inline-block'}
                    )
                ],
                    style={
                        'marginRight': '2em'
                    }
                ),
                html.Div([
                    html.Label('Normalization'),

                    dcc.RadioItems(
                        id='map_norm',
                        options=[{'label': i, 'value': i} for i in ['None', 'Population']],
                        value='Population',
                        inputStyle={'margin-right': '0.4em'},
                        labelStyle={
                            'margin-right': '0.9em',
                            'display': 'block'
                        }
                        # labelStyle={'display': 'inline-block'}
                    )
                ],
                    style={
                        'marginRight': '2em'
                    }
                )
            ],
                style={
                    'display': 'inline-block',
                    'textAlign': 'left',
                    'marginTop': '0.5em',
                    'marginBottom': '0.5em'
                }
            ),
        ]),
    ],
        style={
            'borderStyle': 'outset',
            'paddingLeft': '1em',
            'paddingRight': '1em',
            'paddingTop': '1em',
            'paddingBottom': '1em',
            'marginTop': '1em',
            'marginBottom': '1em'
        }
    ),

    html.Article(children=[

        html.H2(children='Country Data'),

        html.Div([
            html.Div([
                html.P(children='Select country: '),

                dcc.Dropdown(
                    id='country_list',
                    options=[
                        {'label': country.replace('_', ' '), 'value': country} for country in
                        df['countriesAndTerritories'].unique()
                    ],
                    value=countries_of_interest,
                    multi=True,
                    style={
                        'marginTop': '0.5em',
                        'marginBottom': '0.5em'
                    }
                ),
            ],
                style={'display': 'inline'}
            ),

            html.Div([

                html.Label('Y-axis scale'),

                dcc.RadioItems(
                    id='yaxis-type',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Log',
                    inputStyle={'margin-right': '0.4em'},
                    labelStyle={
                        'margin-right': '0.9em',
                        'display': 'block'
                    }
                )],
                style={
                    'display': 'inline-block',
                    'textAlign': 'left',
                    'marginTop': '0.5em',
                    'marginBottom': '0.5em'
                }
            ),
            html.Div([
                html.Label('Normalization'),

                dcc.RadioItems(
                    id='plot_norm',
                    options=[{'label': i, 'value': i} for i in ['None', 'Population']],
                    value='Population',
                    inputStyle={'margin-right': '0.4em'},
                    labelStyle={
                        'margin-right': '0.9em',
                        'display': 'block'
                    }
                    # labelStyle={'display': 'inline-block'}
                )
            ],
                style={
                    'marginRight': '2em'
                }
            ),

            html.Div([
                dcc.Graph(
                    id='COVID_19_cases_deaths',
                    style={'height': '1000px', 'width': '1000px'}
                )],
                style={
                    'display': 'block'
                }
            )
        ]),
    ],
        style={
            'borderStyle': 'outset',
            'paddingLeft': '1em',
            'paddingRight': '1em',
            'paddingTop': '1em',
            'paddingBottom': '1em',
            'marginTop': '1em',
            'marginBottom': '1em'
        }
    )
],
    style={
        'marginLeft': '3em',
        'marginRight': '3em',
    })


# Callback for map
@app.callback(
    Output('COVID_19_cases_deaths', 'figure'),
    [
        Input('country_list', 'value'),
        Input('yaxis-type', 'value'),
        Input('plot_norm', 'value')
    ])
def update_COVID_plot(countries_of_interest, yaxis_type, normalize):
    fig = generate_COVID_plot(df, countries_of_interest, normalize)
    fig.update_yaxes(type=yaxis_type.lower(), row=1, col=1)
    fig.update_yaxes(type=yaxis_type.lower(), row=2, col=1)
    return fig

# Callback for plots
@app.callback(
    Output('COVID_19_map', 'figure'),
    [
        Input('map_qty', 'value'),
        Input('map_norm', 'value'),
        Input('date_slider', 'value')
        # Input('yaxis2-type', 'value')
    ])
def update_COVID_map(col_name, normalize, date_idx):
    if normalize == 'Population':
        col_name = 'cumulative' + col_name + 'PerMillion'
    elif normalize == 'None':
        col_name = 'cumulative' + col_name
    date = df_dates[date_idx]
    fig = generate_COVID_map(df, col_name, date)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

