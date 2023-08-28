"""
Display 3D Greeks graphs

"""
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D # pylint: disable=unused-import
import numpy as np
import plotly.graph_objects as go
from matplotlib import axes
from plotly.offline import plot
# pylint: disable=invalid-name

class Greeks_3D():
    """
    Display 3D Greeks graphs

    """

    @staticmethod
    def graph_space_prep(params:dict) -> dict:
        """
        Prepare the axis ranges to be used in 3D graph.

        Parameters
        ----------
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma',
            'vega', 'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed',
            'color', 'ultima', 'vega_bleed', 'charm'. The default is
            'delta'
        S : Float
             Underlying Stock Price. The default is 100.
        axis : Str
            Whether the x-axis is 'price' or 'vol'. The default
            is 'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D
            graphs. The default is 100.

        Returns
        -------
        Various
            Updated parameters to be used in 3D graph.

        """

        graph_params = {}

        # Select the strike and Time ranges for each greek from the 3D
        # chart ranges dictionary
        SA_lower = params['3D_chart_ranges'][str(params['greek'])]['SA_lower']
        SA_upper = params['3D_chart_ranges'][str(params['greek'])]['SA_upper']
        TA_lower = params['3D_chart_ranges'][str(params['greek'])]['TA_lower']
        TA_upper = params['3D_chart_ranges'][str(params['greek'])]['TA_upper']

        # Set the volatility range from 5% to 50%
        sigmaA_lower = 0.05
        sigmaA_upper = 0.5

        # create arrays of 100 equally spaced points for the ranges of
        # strike prices, volatilities and maturities
        SA = np.linspace(SA_lower * params['S'],
                         SA_upper * params['S'],
                         int(params['spacegrain']))
        TA = np.linspace(TA_lower,
                         TA_upper,
                         int(params['spacegrain']))
        sigmaA = np.linspace(sigmaA_lower,
                             sigmaA_upper,
                             int(params['spacegrain']))

        # set y-min and y-max labels
        graph_params['ymin'] = TA_lower
        graph_params['ymax'] = TA_upper
        graph_params['axis_label2'] = 'Time to Expiration (Days)'

        # set x-min and x-max labels
        if params['axis'] == 'price':
            graph_params['x'], graph_params['y'] = np.meshgrid(SA, TA)
            graph_params['xmin'] = SA_lower
            graph_params['xmax'] = SA_upper
            graph_params['graph_scale'] = 1
            graph_params['axis_label1'] = 'Underlying Value'

        if params['axis'] == 'vol':
            graph_params['x'], graph_params['y'] = np.meshgrid(sigmaA, TA)
            graph_params['xmin'] = sigmaA_lower
            graph_params['xmax'] = sigmaA_upper
            graph_params['graph_scale'] = 100
            graph_params['axis_label1'] = 'Volatility %'

        return graph_params


    @classmethod
    def vis_greeks_3D(
        cls,
        graph_params: dict,
        params: dict) -> None | dict | tuple[mplfig.Figure, axes.Axes, str, int]:
        """
        Display 3D greeks graph.

        Returns
        -------
        If 'interactive' is False, a matplotlib static graph.
        If 'interactive' is True, a plotly graph that can be rotated
        and zoomed.

        """

        # Reverse the z-axis data if direction is 'short'
        if params['direction'] == 'short':
            graph_params['z'] = -graph_params['z']

        graph_params['titlename'] = cls._titlename(params=params)
        graph_params['axis_label3'] = str(params['greek'].title())

        if params['interactive']:
        # Create a plotly graph

            # Set the ranges for the contour values and reverse / rescale axes
            graph_params = cls._plotly_3D_ranges(graph_params=graph_params)

            if params['data_output']:
                return {
                    'params': params,
                    'graph_params': graph_params
                }
            
            return cls._plotly_3D(graph_params=graph_params, params=params)

        # Otherwise create a matplotlib graph
        graph_params = cls._mpl_axis_format(graph_params=graph_params)

        if params['gif']:
            fig, ax, titlename, title_font_scale = cls._mpl_3D(
                graph_params=graph_params, params=params)
            return fig, ax, titlename, title_font_scale

        return cls._mpl_3D(graph_params=graph_params, params=params)


    @staticmethod
    def _titlename(params: dict) -> str:
        """
        Create graph title based on option type, direction and greek

        Returns
        -------
        Graph title.

        """

        titlename = str(str(params['direction'].title())
                        +' '
                        +params['option_title']
                        +' Option '
                        +str(params['greek'].title()))

        return titlename


    @staticmethod
    def _plotly_3D_ranges(graph_params: dict) -> dict:
        """
        Generate contour ranges and format axes for plotly 3D graph

        Returns
        -------
        axis ranges

        """
        # Set the ranges for the contour values and reverse / rescale axes
        graph_params['x'], graph_params['y'] = (
            graph_params['y'] * 365,
            graph_params['x'] * graph_params['graph_scale'])
        graph_params['x_start'] = graph_params['ymin']
        graph_params['x_stop'] = graph_params['ymax'] * 360
        graph_params['x_size'] = graph_params['x_stop'] / 18
        graph_params['y_start'] = graph_params['xmin']
        graph_params['y_stop'] = (
            graph_params['xmax'] * graph_params['graph_scale'])
        graph_params['y_size'] = (
            int((graph_params['xmax'] - graph_params['xmin']) / 20))
        graph_params['z_start'] = np.min(graph_params['z'])
        graph_params['z_stop'] = np.max(graph_params['z'])
        graph_params['z_size'] = (
            int((np.max(graph_params['z']) - np.min(graph_params['z'])) / 10))

        return graph_params


    @staticmethod
    def _plotly_3D(
        graph_params: dict,
        params: dict) -> go.Figure | None:
        """
        Display 3D greeks graph.

        Returns
        -------
        plotly 3D graph

        """
        # create plotly figure object
        fig = go.Figure(
            data=[go.Surface(
                x=graph_params['x'],
                y=graph_params['y'],
                z=graph_params['z'],

                # set the colorscale to the chosen
                # colorscheme
                colorscale=params['colorscheme'],

                # Define the contours
                contours = {
                    "x": {"show": True,
                        "start": graph_params['x_start'],
                        "end": graph_params['x_stop'],
                        "size": graph_params['x_size'],
                        "color": "white"},
                    "y": {"show": True,
                        "start": graph_params['y_start'],
                        "end": graph_params['y_stop'],
                        "size": graph_params['y_size'],
                        "color": "white"},
                    "z": {"show": True,
                        "start": graph_params['z_start'],
                        "end": graph_params['z_stop'],
                        "size": graph_params['z_size']}
                    },
                )
            ]
        )

        # Set initial view position
        camera = {
            'eye': {
                'x': 2,
                'y': 1,
                'z': 1
            }
        }

        # Set x-axis to decrease from left to right
        fig.update_scenes(xaxis_autorange="reversed")

        # Set y-axis to increase from left to right
        fig.update_scenes(yaxis_autorange="reversed")
        fig.update_layout(
            scene={
                'xaxis': {
                    'backgroundcolor': "rgb(200, 200, 230)",
                    'gridcolor': "white",
                    'showbackground': True,
                    'zerolinecolor': "white"
                    },
                'yaxis': {
                    'backgroundcolor': "rgb(230, 200, 230)",
                    'gridcolor': "white",
                    'showbackground': True,
                    'zerolinecolor': "white"
                    },
                'zaxis': {
                    'backgroundcolor': "rgb(230, 230, 200)",
                    'gridcolor': "white",
                    'showbackground': True,
                    'zerolinecolor': "white"
                    },
                    # Label axes
                    'xaxis_title': graph_params['axis_label2'],
                    'yaxis_title': graph_params['axis_label1'],
                    'zaxis_title': graph_params['axis_label3']
                    },
            title={
                'text': graph_params['titlename'],
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 20,
                    'color': "black"},
            },
            margin={
                'l': 65,
                'r': 50,
                'b': 65,
                't': 90
            },
            scene_camera=camera)

        if params['web_graph'] is False:
            fig.update_layout(
                autosize=False,
                width=800,
                height=800
                )

        # If running in an iPython notebook the chart will display
        # in line
        if params['notebook']:
            # If output is sent to Dash
            if params['web_graph']:
                return fig

            fig.show()
            return None

        # Otherwise create an HTML file that opens in a new window
        plot(fig, auto_open=True)
        return None


    @staticmethod
    def _mpl_axis_format(graph_params: dict) -> dict:
        """
        Rescale Matplotlib axis values

        Returns
        -------
        x, y axis values

        """
        graph_params['x'] = graph_params['x'] * graph_params['graph_scale']
        graph_params['y'] = graph_params['y'] * 365

        return graph_params


    @staticmethod
    def _mpl_3D(
        params: dict,
        graph_params: dict) -> tuple[mplfig.Figure, axes.Axes, str, int] | None:
        """
        Display 3D greeks graph.

        Returns
        -------
        Matplotlib static graph.

        """

        # Update chart parameters
        plt.style.use('seaborn-darkgrid')
        plt.rcParams.update(params['mpl_3d_params'])

        # create figure with specified size tuple
        fig = plt.figure(figsize=params['size3d'])
        ax = fig.add_subplot(111,
                             projection='3d',
                             azim=params['azim'],
                             elev=params['elev'])

        # Set background color to white
        ax.set_facecolor('w')

        # Create values that scale fonts with fig_size
        ax_font_scale = int(round(params['size3d'][0] * 1.1))
        title_font_scale = int(round(params['size3d'][0] * 1.8))

        # Tint the axis panes, RGB values from 0-1 and alpha denoting
        # color intensity
        ax.w_xaxis.set_pane_color((0.9, 0.8, 0.9, 0.8))
        ax.w_yaxis.set_pane_color((0.8, 0.8, 0.9, 0.8))
        ax.w_zaxis.set_pane_color((0.9, 0.9, 0.8, 0.8))

        # Set z-axis to left hand side
        ax.zaxis._axinfo['juggled'] = (1, 2, 0) # pylint: disable=protected-access

        # Set fontsize of axis ticks
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=ax_font_scale,
                       pad=10)

        # Label axes
        ax.set_xlabel(graph_params['axis_label1'],
                      fontsize=ax_font_scale,
                      labelpad=ax_font_scale*1.5)
        ax.set_ylabel(graph_params['axis_label2'],
                      fontsize=ax_font_scale,
                      labelpad=ax_font_scale*1.5)
        ax.set_zlabel(graph_params['axis_label3'],
                      fontsize=ax_font_scale,
                      labelpad=ax_font_scale*1.5)

        # Auto scale the z-axis
        ax.set_zlim(auto=True)

        # Set x-axis to decrease from left to right
        ax.invert_xaxis()

        # apply graph_scale so that if volatility is the x-axis it
        # will be * 100
        ax.plot_surface(graph_params['x'],
                        graph_params['y'],
                        graph_params['z'],
                        rstride=2,
                        cstride=2,

                        # set the colormap to the chosen colorscheme
                        cmap=plt.get_cmap(params['colorscheme']),

                        # set the alpha value to the chosen
                        # colorintensity
                        alpha=params['colorintensity'],
                        linewidth=0.25)

        # Specify title
        st = fig.suptitle(graph_params['titlename'],
                          fontsize=title_font_scale,
                          fontweight=0,
                          color='black',
                          style='italic',
                          y=1.02)

        st.set_y(0.98)
        fig.subplots_adjust(top=1)

        if params['gif']:
            return fig, ax, graph_params['titlename'], title_font_scale

        # Display graph
        return plt.show()
