"""
Display 2D and 3D Greeks graphs

"""

from matplotlib import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D # pylint: disable=unused-import
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from optionvisualizer.sensitivities import Sens
# pylint: disable=invalid-name

class Greeks():
    """
    Display 2D and 3D Greeks graphs

    """

    @classmethod
    def greeks_graphs_2D(cls, params):
        """
        Plot chosen 2D greeks graph.


        Parameters
        ----------
        x_plot : Str
                 The x-axis variable ('price', 'strike', 'vol' or
                 'time'). The default is 'price'.
        y_plot : Str
                 The y-axis variable ('value', 'delta', 'gamma', 'vega'
                 or 'theta'). The default is 'delta.
        S : Float
             Underlying Stock Price. The default is 100.
        G1 : Float
             Strike Price of option 1. The default is 90.
        G2 : Float
             Strike Price of option 2. The default is 100.
        G3 : Float
             Strike Price of option 3. The default is 110.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        T1 : Float
             Time to Maturity of option 1. The default is 0.25
             (3 months).
        T2 : Float
             Time to Maturity of option 1. The default is 0.25
             (3 months).
        T3 : Float
             Time to Maturity of option 1. The default is 0.25
             (3 months).
        time_shift : Float
             Difference between T1 and T2 in rho graphs. The default
             is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        size2d : Tuple
            Figure size for matplotlib chart. The default is (6, 4).
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity.
            The default is False.

        Returns
        -------
        Runs method to create data for 2D greeks graph.

        """

        if params['gif']:
            fig, ax = cls._2D_general_graph(params=params)

            return fig, ax

        return cls._2D_general_graph(params=params)


    @classmethod
    def greeks_graphs_3D(cls, params):
        """
        Plot chosen 3D greeks graph.

        Parameters
        ----------
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma',
            'vega', 'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed',
            'color', 'ultima', 'vega_bleed', 'charm'. The default is
            'delta'
        S : Float
             Underlying Stock Price. The default is 100.
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        interactive : Bool
            Whether to show matplotlib (False) or plotly(True) graph.
            The default is False.
        notebook : Bool
            Whether the function is being run in an IPython notebook and
            hence whether it should output in line or to an HTML file.
            The default is False.
        colorscheme : Str
            The matplotlib colormap or plotly colorscale to use. The
            default is 'jet' (which works in both).
        colorintensity : Float
            The alpha value indicating level of transparency /
            intensity. The default is 1.
        size3d : Tuple
            Figure size for matplotlib chart. The default is (15, 12).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        axis : Str
            Whether the x-axis is 'price' or 'vol'. The default
            is 'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D
            graphs. The default is 100.
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity.
            The default is False.
        gif : Bool
            Whether to create an animated gif. The default is False.

        Returns
        -------
        Runs method to display 3D greeks graph.

        """

        # Select the input name and method name from the greek
        # dictionary
        for greek_label in params['greek_dict'].keys():

            # If the greek is the same for call or put, set the option
            # value to 'Call / Put'
            if params['greek'] in params['equal_greeks']:
                params['option'] = 'Call / Put'

            # For the specified greek
            if params['greek'] == greek_label:

                # Prepare the graph axes
                graph_params = cls._graph_space_prep(params=params)

                if params['axis'] == 'price':

                    # Select the individual greek method from sensitivities
                    graph_params['z'] = Sens.sensitivities_static(
                        params=params, S=graph_params['x'], K=params['S'],
                        T=graph_params['y'], r=params['r'],
                        q=params['q'], sigma=params['sigma'],
                        option=params['option'], greek=params['greek'],
                        price_shift=0.25, vol_shift=0.001,
                        ttm_shift=(1 / 365), num_sens=params['num_sens'])

                if params['axis'] == 'vol':

                    # Select the individual greek method from sensitivities
                    graph_params['z'] = Sens.sensitivities_static(
                        params=params, S=params['S'], K=params['S'],
                        T=graph_params['y'], r=params['r'], q=params['q'],
                        sigma=graph_params['x'], option=params['option'],
                        greek=params['greek'], price_shift=0.25,
                        vol_shift=0.001, ttm_shift=(1 / 365),
                        num_sens=params['num_sens'])

        # Run the 3D visualisation method
        if params['gif']:
            fig, ax, titlename, title_font_scale = cls._vis_greeks_3D(
                graph_params=graph_params, params=params)
            return fig, ax, titlename, title_font_scale

        return cls._vis_greeks_3D(graph_params=graph_params, params=params)


    @classmethod
    def _2D_general_graph(cls, params):
        """
        Creates data for 2D greeks graph.

        Returns
        -------
        Runs method to graph using Matplotlib.

        """

        # create arrays of 1000 equally spaced points for a range of
        # strike prices, volatilities and maturities
        params['SA'] = np.linspace(0.8 * params['S'], 1.2 * params['S'], 1000)
        params['sigmaA'] = np.linspace(0.05, 0.5, 1000)
        params['TA'] = np.linspace(0.01, 1, 1000)

        # y-axis parameters other than rho require 3 options to be
        # graphed
        if params['y_plot'] in params['y_name_dict'].keys():

            params = cls._non_rho_data(params)

        # rho requires 4 options to be graphed
        if params['y_plot'] == 'rho':

            params = cls._rho_data(params)

        # Convert the x-plot and y-plot values to axis labels
        xlabel = params['label_dict'][str(params['x_plot'])]
        ylabel = params['label_dict'][str(params['y_plot'])]

        # If the greek is rho or the same for a call or a put, set the
        # option name to 'Call / Put'
        if params['y_plot'] in [params['equal_greeks'], 'rho']:
            params['option'] = 'Call / Put'

        # Create chart title as direction plus option type plus y-plot
        # vs x-plot
        title = (str(params['direction'].title())
                 +' '
                 +str(params['option'].title())
                 +' '
                 +params['y_plot'].title()
                 +' vs '
                 +params['x_plot'].title())

        # Set the x-axis array as price, vol or time
        x_name = str(params['x_plot'])
        if x_name in params['x_name_dict'].keys():
            xarray = (params[str(params['x_name_dict'][x_name])] *
                      params['x_scale_dict'][x_name])

        vis_params = {
            'x_plot':params['x_plot'],
            'yarray1':params['C1'],
            'yarray2':params['C2'],
            'yarray3':params['C3'],
            'yarray4':params['C4'],
            'xarray':xarray,
            'label1':params['label1'],
            'label2':params['label2'],
            'label3':params['label3'],
            'label4':params['label4'],
            'xlabel':xlabel,
            'ylabel':ylabel,
            'title':title,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style'],
            'gif':params['gif']
            }

        # Plot 3 option charts
        if params['y_plot'] in params['y_name_dict'].keys():
            if params['interactive']:
                return cls._vis_greeks_plotly(
                    vis_params=vis_params, params=params)
            else:
                if params['gif']:
                    fig, ax = cls._vis_greeks_mpl(
                        vis_params=vis_params, params=params)
                    return fig, ax

                return cls._vis_greeks_mpl(
                    vis_params=vis_params, params=params)

        # Plot Rho charts
        if params['y_plot'] == 'rho':
            if params['interactive']:
                return cls._vis_greeks_plotly(
                    vis_params=vis_params, params=params)
            else:
                vis_params.update({'gif':False})
                return cls._vis_greeks_mpl(
                    vis_params=vis_params, params=params)

        return print("Please select a valid pair")


    @classmethod
    def _non_rho_data(cls, params):

        for opt in [1, 2, 3]:
            if params['x_plot'] == 'price':

                # For price we set S to the array SA
                params['C'+str(opt)] = Sens.sensitivities_static(
                    params=params, S=params['SA'],
                    K=params['G'+str(opt)], T=params['T'+str(opt)],
                    r=params['r'], q=params['q'], sigma=params['sigma'],
                    option=params['option'],
                    greek=params['y_name_dict'][params['y_plot']],
                    price_shift=0.25, vol_shift=0.001,
                    ttm_shift=(1 / 365), rate_shift=0.0001,
                    num_sens=params['num_sens'])

            if params['x_plot'] == 'vol':

                # For vol we set sigma to the array sigmaA
                params['C'+str(opt)] = Sens.sensitivities_static(
                    params=params, S=params['S'],
                    K=params['G'+str(opt)], T=params['T'+str(opt)],
                    r=params['r'], q=params['q'], sigma=params['sigmaA'],
                    option=params['option'],
                    greek=params['y_name_dict'][params['y_plot']],
                    price_shift=0.25, vol_shift=0.001,
                    ttm_shift=(1 / 365), rate_shift=0.0001)

            if params['x_plot'] == 'time':

                # For time we set T to the array TA
                params['C'+str(opt)] = Sens.sensitivities_static(
                    params=params, S=params['S'],
                    K=params['G'+str(opt)], T=params['TA'], r=params['r'],
                    q=params['q'], sigma=params['sigma'],
                    option=params['option'],
                    greek=params['y_name_dict'][params['y_plot']],
                    price_shift=0.25, vol_shift=0.001,
                    ttm_shift=(1 / 365), rate_shift=0.0001)

        # Reverse the option value if direction is 'short'
        if params['direction'] == 'short':
            for opt in [1, 2, 3]:
                params['C'+str(opt)] = -params['C'+str(opt)]

        # Call strike_tenor_label method to assign labels to chosen
        # strikes and tenors
        params = cls._strike_tenor_label(params)

        return params


    @staticmethod
    def _rho_data(params):

        # Set T1 and T2 to the specified time and shifted time
        params['T1'] = params['T']
        params['T2'] = params['T'] + params['time_shift']

        # 2 Tenors
        tenor_type = {1:1, 2:2, 3:1, 4:2}

        # And call and put for each tenor
        opt_type = {1:'call', 2:'call', 3:'put', 4:'put'}
        for opt in [1, 2, 3, 4]:
            if params['x_plot'] == 'price':

                # For price we set S to the array SA
                params['C'+str(opt)] = Sens.sensitivities_static(
                    params=params, S=params['SA'], K=params['G2'],
                    T=params['T'+str(tenor_type[opt])], r=params['r'],
                    q=params['q'], sigma=params['sigma'],
                    option=opt_type[opt], greek=params['y_plot'],
                    price_shift=0.25, vol_shift=0.001, ttm_shift=(1 / 365),
                    rate_shift=0.0001)

            if params['x_plot'] == 'strike':

                # For strike we set K to the array SA
                params['C'+str(opt)] = Sens.sensitivities_static(
                    params=params, S=params['S'], K=params['SA'],
                    T=params['T'+str(tenor_type[opt])], r=params['r'],
                    q=params['q'], sigma=params['sigma'],
                    option=opt_type[opt], greek=params['y_plot'],
                    price_shift=0.25, vol_shift=0.001, ttm_shift=(1 / 365),
                    rate_shift=0.0001)

            if params['x_plot'] == 'vol':

                # For vol we set sigma to the array sigmaA
                params['C'+str(opt)] = Sens.sensitivities_static(
                    params=params, S=params['S'], K=params['G2'],
                    T=params['T'+str(tenor_type[opt])], r=params['r'],
                    q=params['q'], sigma=params['sigmaA'],
                    option=opt_type[opt], greek=params['y_plot'],
                    price_shift=0.25, vol_shift=0.001, ttm_shift=(1 / 365),
                    rate_shift=0.0001)

        # Reverse the option value if direction is 'short'
        if params['direction'] == 'short':
            for opt in [1, 2, 3, 4]:
                params['C'+str(opt)] = -params['C'+str(opt)]

        # Assign the option labels
        params['label1'] = str(int(params['T1'] * 365))+' Day Call'
        params['label2'] = str(int(params['T2'] * 365))+' Day Call'
        params['label3'] = str(int(params['T1'] * 365))+' Day Put'
        params['label4'] = str(int(params['T2'] * 365))+' Day Put'

        return params


    @staticmethod
    def _strike_tenor_label(params):
        """
        Assign labels to chosen strikes and tenors in 2D greeks graph

        Returns
        -------
        Str
            Labels for each of the 3 options in 2D greeks graph.

        """
        strike_label = dict()
        for strike, strike_value in {'G1':'label1',
                                     'G2':'label2',
                                     'G3':'label3'}.items():

            # If the strike is 100% change name to 'ATM'
            if params[str(strike)] == params['S']:
                strike_label[strike_value] = 'ATM Strike'
            else:
                strike_label[strike_value] = (
                    str(int(params[strike]))
                    +' Strike')

        for tenor, tenor_value in {'T1':'label1',
                                   'T2':'label2',
                                   'T3':'label3'}.items():

            # Make each label value the number of days to maturity
            # plus the strike level
            params[tenor_value] = (
                str(int(params[str(tenor)]*365))
                +' Day '
                +strike_label[str(tenor_value)])

        return params


    @staticmethod
    def _vis_greeks_mpl(vis_params, params):
        """
        Display the 2D greeks chart using matplotlib

        Parameters
        ----------
        xarray : Array
            x-axis values.
        yarray1 : Array
            y-axis values for option 1.
        yarray2 : Array
            y-axis values for option 2.
        yarray3 : Array
            y-axis values for option 3.
        yarray4 : Array
            y-axis values for option 4.
        label1 : Str
            Option 1 label.
        label2 : Str
            Option 2 label.
        label3 : Str
            Option 3 label.
        label4 : Str
            Option 4 label.
        xlabel : Str
            x-axis label.
        ylabel : Str
            y-axis label.
        title : Str
            Chart title.

        Returns
        -------
        2D Greeks chart.

        """

        # Set style to chosen mpl_style (default is Seaborn Darkgrid)
        plt.style.use(vis_params['mpl_style'])

        # Update chart parameters
        pylab.rcParams.update(params['mpl_params'])

        # Create the figure and axes objects
        fig, ax = plt.subplots(figsize=vis_params['size2d'])

        # If plotting against time, show time to maturity reducing left
        # to right
        if vis_params['x_plot'] == 'time':
            ax.invert_xaxis()

        # Plot the 1st option
        ax.plot(vis_params['xarray'],
                vis_params['yarray1'],
                color='blue',
                label=vis_params['label1'])

        # Plot the 2nd option
        ax.plot(vis_params['xarray'],
                vis_params['yarray2'],
                color='red',
                label=vis_params['label2'])

        # Plot the 3rd option
        ax.plot(vis_params['xarray'],
                vis_params['yarray3'],
                color='green',
                label=vis_params['label3'])

        # 4th option only used in Rho graphs
        if vis_params['label4'] is not None:
            ax.plot(vis_params['xarray'],
                    vis_params['yarray4'],
                    color='orange',
                    label=vis_params['label4'])

        # Apply a grid
        plt.grid(True)

        # Apply a black border to the chart
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')

        # Set x and y axis labels and title
        ax.set(xlabel=vis_params['xlabel'],
               ylabel=vis_params['ylabel'],
               title=vis_params['title'])

        # Create a legend
        ax.legend(loc=0, fontsize=10)

        if vis_params['gif']:
            return fig, ax

            # Display the chart
        return plt.show()


    @classmethod
    def _vis_greeks_plotly(cls, vis_params, params):
        """
        Display the 2D greeks chart using plotly

        Parameters
        ----------
        xarray : Array
            x-axis values.
        yarray1 : Array
            y-axis values for option 1.
        yarray2 : Array
            y-axis values for option 2.
        yarray3 : Array
            y-axis values for option 3.
        yarray4 : Array
            y-axis values for option 4.
        label1 : Str
            Option 1 label.
        label2 : Str
            Option 2 label.
        label3 : Str
            Option 3 label.
        label4 : Str
            Option 4 label.
        xlabel : Str
            x-axis label.
        ylabel : Str
            y-axis label.
        title : Str
            Chart title.

        Returns
        -------
        2D Greeks chart.

        """

        # Create the figure
        fig = go.Figure()

        # If plotting against time, show time to maturity reducing left
        # to right
        if vis_params['x_plot'] == 'time':
            fig.update_xaxes(autorange="reversed")

        # Plot the 1st option
        fig.add_trace(go.Scatter(x=vis_params['xarray'],
                                 y=vis_params['yarray1'],
                                 line=dict(color='blue'),
                                 name=vis_params['label1']))

        # Plot the 2nd option
        fig.add_trace(go.Scatter(x=vis_params['xarray'],
                                 y=vis_params['yarray2'],
                                 line=dict(color='red'),
                                 name=vis_params['label2']))

        # Plot the 3rd option
        fig.add_trace(go.Scatter(x=vis_params['xarray'],
                                 y=vis_params['yarray3'],
                                 line=dict(color='green'),
                                 name=vis_params['label3']))

        # 4th option only used in Rho graphs
        if vis_params['label4'] is not None:
            fig.add_trace(go.Scatter(x=vis_params['xarray'],
                                     y=vis_params['yarray4'],
                                     line=dict(color='orange'),
                                     name=vis_params['label4']))
            rho_graph=True
        else:
            rho_graph=False

        xmin, xmax, ymin, ymax = cls._graph_range_2d(
            vis_params=vis_params, rho_graph=rho_graph)

        fig.update_layout(
            title={'text': vis_params['title'],
                   'y':0.95,
                   'x':0.5,
                   'xanchor':'center',
                   'yanchor':'top',
                   'font':dict(size=20,
                               color="#f2f5fa")},
            xaxis_title={'text': vis_params['xlabel'],
                         'font':dict(size=15,
                                     color="#f2f5fa")},
            yaxis_title={'text': vis_params['ylabel'],
                         'font':dict(size=15,
                                     color="#f2f5fa")},
            font={'color': '#f2f5fa'},
            paper_bgcolor='black',
            plot_bgcolor='black',
            legend=dict(
                x=0.05,
                y=0.95,
                traceorder="normal",
                bgcolor='rgba(0, 0, 0, 0)',
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="#f2f5fa"
                ),
            ),

            width=800,
            height=600
        )

        fig.update_xaxes(showline=True,
                         linewidth=2,
                         linecolor='#2a3f5f',
                         mirror=True,
                         range = [xmin, xmax],
                         gridwidth=1,
                         gridcolor='#2a3f5f',
                         zeroline=False)

        fig.update_yaxes(showline=True,
                         linewidth=2,
                         linecolor='#2a3f5f',
                         mirror=True,
                         range = [ymin, ymax],
                         gridwidth=1,
                         gridcolor='#2a3f5f',
                         zeroline=False)

        # If running in an iPython notebook the chart will display
        # in line
        if params['notebook']:
            # If output is sent to Dash
            if params['web_graph']:
                return fig
            else:
                fig.show()
                return

        # Otherwise create an HTML file that opens in a new window
        else:
            plot(fig, auto_open=True)
            return


    @staticmethod
    def _graph_range_2d(vis_params, rho_graph):
        """
        Set 2D graph ranges

        Parameters
        ----------
        vis_params : Dict
            Dictionary of parameters.

        Returns
        -------
        xmin : Float
            x-axis minimum.
        xmax : Float
            x-axis maximum.
        ymin : Float
            y-axis minimum.
        ymax : Float
            y-axis maximum.

        """

        min_x = vis_params['xarray'].min()
        max_x = vis_params['xarray'].max()
        min_y1 = vis_params['yarray1'].min()
        max_y1 = vis_params['yarray1'].max()
        min_y2 = vis_params['yarray2'].min()
        max_y2 = vis_params['yarray2'].max()
        min_y3 = vis_params['yarray3'].min()
        max_y3 = vis_params['yarray3'].max()

        if rho_graph:
            min_y4 = vis_params['yarray4'].min()
            max_y4 = vis_params['yarray4'].max()
        else:
            min_y4 = min_y1
            max_y4 = max_y1

        x_scale_shift = (max_x - min_x) * 0.05
        xmin = min_x - x_scale_shift
        xmax = max_x + x_scale_shift
        y_scale_shift = (
            (max(max_y1, max_y2, max_y3, max_y4)
            - min(min_y1, min_y2, min_y3, min_y4))
            * 0.05)
        ymin = min(min_y1, min_y2, min_y3, min_y4) - y_scale_shift
        ymax = max(max_y1, max_y2, max_y3, max_y4) + y_scale_shift

        return xmin, xmax, ymin, ymax


    @staticmethod
    def _graph_space_prep(params):
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
    def _vis_greeks_3D(cls, graph_params, params):
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

            return cls._plotly_3D(graph_params=graph_params, params=params)

        # Otherwise create a matplotlib graph
        graph_params = cls._mpl_axis_format(graph_params=graph_params)

        if params['gif']:
            fig, ax, titlename, title_font_scale = cls._mpl_3D(
                graph_params=graph_params, params=params)
            return fig, ax, titlename, title_font_scale

        return cls._mpl_3D(graph_params=graph_params, params=params)


    @staticmethod
    def _titlename(params):
        """
        Create graph title based on option type, direction and greek

        Returns
        -------
        Graph title.

        """
        # Label the graph based on whether it is different for calls
        # & puts or the same
        if params['option'] == 'Call / Put':
            titlename = str(str(params['direction'].title())
                            +' '
                            +params['option']
                            +' Option '
                            +str(params['greek'].title()))
        else:
            titlename = str(str(params['direction'].title())
                            +' '
                            +str(params['option'].title())
                            +' Option '
                            +str(params['greek'].title()))
        return titlename


    @staticmethod
    def _plotly_3D_ranges(graph_params):
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
    def _plotly_3D(graph_params, params):
        """
        Display 3D greeks graph.

        Returns
        -------
        plotly 3D graph

        """
        # create plotly figure object
        fig = go.Figure(
            data=[go.Surface(x=graph_params['x'],
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
                                       "color":"white"},
                                 "y": {"show": True,
                                       "start": graph_params['y_start'],
                                       "end": graph_params['y_stop'],
                                       "size": graph_params['y_size'],
                                       "color":"white"},
                                 "z": {"show": True,
                                       "start": graph_params['z_start'],
                                       "end": graph_params['z_stop'],
                                       "size": graph_params['z_size']}
                                 },
                             )
                  ]
            )

        # Set initial view position
        camera = dict(
            eye=dict(x=2, y=1, z=1)
        )

        # Set x-axis to decrease from left to right
        fig.update_scenes(xaxis_autorange="reversed")

        # Set y-axis to increase from left to right
        fig.update_scenes(yaxis_autorange="reversed")
        fig.update_layout(scene = dict(
                            xaxis = dict(
                                 backgroundcolor="rgb(200, 200, 230)",
                                 gridcolor="white",
                                 showbackground=True,
                                 zerolinecolor="white",),
                            yaxis = dict(
                                backgroundcolor="rgb(230, 200,230)",
                                gridcolor="white",
                                showbackground=True,
                                zerolinecolor="white"),
                            zaxis = dict(
                                backgroundcolor="rgb(230, 230,200)",
                                gridcolor="white",
                                showbackground=True,
                                zerolinecolor="white",),
                            # Label axes
                            xaxis_title=graph_params['axis_label2'],
                            yaxis_title=graph_params['axis_label1'],
                            zaxis_title=graph_params['axis_label3'],),
                          title={'text':graph_params['titlename'],
                                 'y':0.9,
                                 'x':0.5,
                                 'xanchor':'center',
                                 'yanchor':'top',
                                 'font':dict(size=20,
                                             color="black")},
                          autosize=False,
                          width=800, height=800,
                          margin=dict(l=65, r=50, b=65, t=90),
                          scene_camera=camera)

        # If running in an iPython notebook the chart will display
        # in line
        if params['notebook']:
            # If output is sent to Dash
            if params['web_graph']:
                return fig
            else:
                fig.show()
                return

        # Otherwise create an HTML file that opens in a new window
        else:
            plot(fig, auto_open=True)
            return


    @staticmethod
    def _mpl_axis_format(graph_params):
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
    def _mpl_3D(params, graph_params):
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
