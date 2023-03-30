"""
Display 2D Greeks graphs

"""

#from matplotlib import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D # pylint: disable=unused-import
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from optionvisualizer.sensitivities import Sens

# pylint: disable=invalid-name

class Greeks_2D():
    """
    Display 2D Greeks graphs

    """

    @classmethod
    def vis_greeks_2D(cls, params):
        """
        Creates data for 2D greeks graph.

        Returns
        -------
        Runs method to graph using Matplotlib.

        """

        # create arrays of (default is 1000) equally spaced points for a range of strike prices, volatilities and maturities
        params['SA'] = np.linspace(params['strike_min'] * params['S'], params['strike_max'] * params['S'], params['linspace_granularity'])
        params['sigmaA'] = np.linspace(params['vol_min'], params['vol_max'], params['linspace_granularity'])
        params['TA'] = np.linspace(params['time_min'], params['time_max'], params['linspace_granularity'])

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
            'color1':params['option1_color'],
            'color2':params['option2_color'],
            'color3':params['option3_color'],
            'color4':params['option4_color'],
            'xlabel':xlabel,
            'ylabel':ylabel,
            'title':title,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style'],
            'gif':params['gif'],
            'graph_figure':params['graph_figure'] 
            }

        # Plot 3 option charts
        if params['y_plot'] in params['y_name_dict'].keys():
            if params['interactive']:
                return cls._vis_greeks_plotly(
                    vis_params=vis_params, params=params)

            if params['gif'] or params['graph_figure']:
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

            vis_params.update({'gif':False})
            if params['graph_figure']:
                fig, ax = cls._vis_greeks_mpl(
                vis_params=vis_params, params=params)
                return fig, ax
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
        #pylab.rcParams.update(params['mpl_params'])
        mpl.rcParams.update(params['mpl_params'])

        # Create the figure and axes objects
        fig, ax = plt.subplots(figsize=vis_params['size2d'])

        # If plotting against time, show time to maturity reducing left
        # to right
        if vis_params['x_plot'] == 'time':
            ax.invert_xaxis()

        # Plot the 1st option
        ax.plot(vis_params['xarray'],
                vis_params['yarray1'],
                vis_params['color1'],
                label=vis_params['label1'])

        # Plot the 2nd option
        ax.plot(vis_params['xarray'],
                vis_params['yarray2'],
                vis_params['color2'],
                label=vis_params['label2'])

        # Plot the 3rd option
        ax.plot(vis_params['xarray'],
                vis_params['yarray3'],
                vis_params['color3'],
                label=vis_params['label3'])

        # 4th option only used in Rho graphs
        if vis_params['label4'] is not None:
            ax.plot(vis_params['xarray'],
                    vis_params['yarray4'],
                    vis_params['color4'],
                    label=vis_params['label4'])

        # Apply a grid
        plt.grid(True)

        # Apply a black border to the chart
        #ax.patch.set_edgecolor('black')
        #ax.patch.set_linewidth('1')

        fig.patch.set(linewidth=1, edgecolor='black')

        # Set x and y axis labels and title
        ax.set(xlabel=vis_params['xlabel'],
               ylabel=vis_params['ylabel'],
               title=vis_params['title'])

        # Create a legend
        ax.legend(loc=0, fontsize=10)

        if vis_params['gif'] or vis_params['graph_figure']:
            plt.show()
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
        )

        if params['web_graph'] is False:
            fig.update_layout(
                autosize=False,
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

            fig.show()
            return

        # Otherwise create an HTML file that opens in a new window
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

        ranges = {}
        ranges['min_x'] = vis_params['xarray'].min()
        ranges['max_x'] = vis_params['xarray'].max()
        ranges['min_y1'] = vis_params['yarray1'].min()
        ranges['max_y1'] = vis_params['yarray1'].max()
        ranges['min_y2'] = vis_params['yarray2'].min()
        ranges['max_y2'] = vis_params['yarray2'].max()
        ranges['min_y3'] = vis_params['yarray3'].min()
        ranges['max_y3'] = vis_params['yarray3'].max()

        if rho_graph:
            ranges['min_y4'] = vis_params['yarray4'].min()
            ranges['max_y4'] = vis_params['yarray4'].max()
        else:
            ranges['min_y4'] = ranges['min_y1']
            ranges['max_y4'] = ranges['max_y1']

        x_scale_shift = (ranges['max_x'] - ranges['min_x']) * 0.05
        xmin = ranges['min_x'] - x_scale_shift
        xmax = ranges['max_x'] + x_scale_shift
        y_scale_shift = (
            (max(ranges['max_y1'],
                 ranges['max_y2'],
                 ranges['max_y3'],
                 ranges['max_y4'])
            - min(ranges['min_y1'],
                  ranges['min_y2'],
                  ranges['min_y3'],
                  ranges['min_y4']))
            * 0.05)
        ymin = min(ranges['min_y1'],
                   ranges['min_y2'],
                   ranges['min_y3'],
                   ranges['min_y4']) - y_scale_shift
        ymax = max(ranges['max_y1'],
                   ranges['max_y2'],
                   ranges['max_y3'],
                   ranges['max_y4']) + y_scale_shift

        return xmin, xmax, ymin, ymax
