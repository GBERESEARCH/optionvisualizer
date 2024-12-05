"""
Display 2D and 3D Greeks graphs

"""
import matplotlib.figure as mplfig
from matplotlib import axes
from optionvisualizer.sensitivities import Sens
from optionvisualizer.greeks_2d import Greeks_2D
from optionvisualizer.greeks_3d import Greeks_3D

# pylint: disable=invalid-name

class Greeks():
    """
    Display 2D and 3D Greeks graphs

    """

    @staticmethod
    def greeks_graphs_2D(
        params: dict) -> tuple[mplfig.Figure, axes.Axes] | dict | None:
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
            The default is 'seaborn-v0_8-darkgrid'.
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity.
            The default is False.

        Returns
        -------
        Runs method to create data for 2D greeks graph.

        """

        if params['gif'] or params['graph_figure']:
            fig, ax = Greeks_2D.vis_greeks_2D(params=params)
            return fig, ax

        if params['data_output']:
            data_dict = Greeks_2D.vis_greeks_2D(params=params)
            return data_dict

        return Greeks_2D.vis_greeks_2D(params=params)


    @staticmethod
    def greeks_graphs_3D(
        params: dict) -> tuple[mplfig.Figure, axes.Axes, str, int] | dict | None:
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
                params['option_title'] = 'Call / Put'
            else:
                params['option_title'] = params['option'].title()

            # For the specified greek
            if params['greek'] == greek_label:

                # Prepare the graph axes
                graph_params = Greeks_3D.graph_space_prep(params=params)

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
            fig, ax, titlename, title_font_scale = Greeks_3D.vis_greeks_3D(
                graph_params=graph_params, params=params)
            return fig, ax, titlename, title_font_scale

        if params['data_output']:
            data_dict = Greeks_3D.vis_greeks_3D(
                graph_params=graph_params, params=params)
            return data_dict

        return Greeks_3D.vis_greeks_3D(
            graph_params=graph_params, params=params)
