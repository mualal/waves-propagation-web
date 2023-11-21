from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np


def create_input_field(txt, id, holder, type, min, val):
    input_field = dbc.InputGroup([
        dbc.InputGroupText(txt),
        dbc.Input(
            id=id,
            placeholder=holder,
            type=type,
            min=min,
            value=val,
            ),
        ],
        className="mb-3",
        )
    return input_field


def specify_initial_and_boundary(num, beta, n_0, u_0, stiffness, foundation_stiffness, mass, a, omega):

    k_1 = np.arcsin(np.sqrt(mass*(omega**2-foundation_stiffness/mass)/(4*stiffness))) * 2 / a
    g_1 = a/(2*omega)*np.sqrt((omega**2-foundation_stiffness/mass)*((4*stiffness+foundation_stiffness)/mass-omega**2))

    disp = u_0 * np.exp(-beta**2/2 * (num - n_0)**2) * np.sin(num * a * k_1)
    disp[np.where(num >= -1)] = 0

    vel = -u_0 * np.exp(-beta**2/2 * (num - n_0)**2)
    vel *= (omega * np.cos(k_1*a*num) - beta**2*g_1/a*(num-n_0)*np.sin(num * a * k_1))
    vel[np.where(num >= -1)] = 0
    
    return disp, vel


def construct_app(app):

    fig = go.Figure()

    app.layout = html.Div(children=[
        html.H1(children='Chain-chain Interface'),
        html.Div(children='Parameters Values'),
        create_input_field('leftx', 'leftx', 'Min 0.01', 'number', -2000, -400),
        create_input_field('rightx = ', 'rightx', 'Min 0.01', 'number', 0, 600),
        create_input_field('m1 = ', 'm1', 'Min 0.01', 'number', 0.01, 1.0),
        create_input_field('m2 = ', 'm2', 'Min 0.01', 'number', 0.01, 0.5),
        create_input_field('c1 = ', 'c1', 'Min 0.01', 'number', 0.01, 1.0),
        create_input_field('c2 = ', 'c2', 'Min 0.01', 'number', 0.01, 1.0),
        create_input_field('c12 = ', 'c12', 'Min 0.01', 'number', 0, 3.0),
        create_input_field('d1 = ', 'd1', 'Min 0.01', 'number', 0.0, 0.0),
        create_input_field('d2 = ', 'd2', 'Min 0.01', 'number', 0.0, 0.0),
        create_input_field('a = ', 'a', 'Min 0.01', 'number', 1, 1),
        create_input_field('omega = ', 'omega', 'Min 0.01', 'number', 0.01, 1),
        create_input_field('beta = ', 'beta', 'Min 0.01', 'number', 0, 0.03),
        create_input_field('n0 = ', 'n0', 'Min 0.01', 'number', -1000, -150),
        create_input_field('u0 = ', 'u0', 'Min 0.01', 'number', 0.01, 1),
        create_input_field('dt', 'dt', 'Min 0.01', 'number', 0.000001, 0.005),
        create_input_field('tmax = ', 'tmax', 'Min 0.01', 'number', 1, 350),
        dcc.Graph(
        id='example-graph',
        figure=fig
        )
    ])
    
    @app.callback(
        Output('example-graph','figure'),
        Input('leftx','value'),
        Input('rightx','value'),
        Input('m1','value'),
        Input('m2','value'),
        Input('c1','value'),
        Input('c2','value'),
        Input('c12','value'),
        Input('d1','value'),
        Input('d2','value'),
        Input('a','value'),
        Input('omega','value'),
        Input('beta','value'),
        Input('n0','value'),
        Input('u0','value'),
        Input('dt','value'),
        Input('tmax','value'),
    )
    def update_figure(leftx, rightx, m1, m2, c1, c2, c12, d1, d2, a, omega, beta, n0, u0, dt, tmax):
        num = np.round(np.arange(leftx, rightx, a)/a)
        mass = np.array([m1 if i < 0 else m2 for i in num])
        foundation_stiffness = np.array([d1 if j < 0 else d2 for j in num])
        stiffness = np.array([c1 if k < 0 else c2 for k in num])
        stiffness[np.where(num == -1)[0][0]] = c12
        disp, vel = specify_initial_and_boundary(num, beta, n0, u0, stiffness, foundation_stiffness, mass, a, omega)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=num,y=disp*m1, name='Начальные перемещения', line=dict(color="#000000")))
        return fig


if __name__ == '__main__':

    app = Dash(__name__,
               external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME])

    construct_app(app)
    app.run_server(debug=True)
