from dash import dcc, html


def MainBox():
    return html.Div([
        html.Div([
            dcc.Slider(min=0,
                        max=100,
                        step=1,
                        value=50,
                        marks={0: {'label': '0'},
                                25: {'label': '25'},
                                50: {'label': '50'},
                                75: {'label': '75'},
                                100: {'label': '100'},
                },
                className="w-full")], className="w-1/4 h-full flex flex-col justify-evenly items-center"),
    ])