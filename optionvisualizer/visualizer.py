"""
Visualize option payoffs and greeks

"""
import copy
from optionvisualizer.visualizer_params import vis_params_dict
from optionvisualizer.option_formulas import Option
from optionvisualizer.simple_payoffs import SimplePayoff
from optionvisualizer.multi_payoffs import MultiPayoff
from optionvisualizer.utilities import Utils
from optionvisualizer.greeks import Greeks
from optionvisualizer.sensitivities import Sens
from optionvisualizer.animated_gifs import Gif
from optionvisualizer.barriers import Barrier

# pylint: disable=invalid-name

class Visualizer():
    """
    Visualize option payoffs and greeks

    """
    def __init__(self, **kwargs):

        # Dictionary of parameter defaults
        self.df_dict = copy.deepcopy(vis_params_dict)

        # Store initial inputs
        inputs = {}
        for key, value in kwargs.items():
            inputs[key] = value

        # Initialise system parameters
        params = Utils._init_params(inputs)

        self.params = params


    def option_data(self, option_value, **kwargs):
        """
        Calculate Option prices or Greeks

        Parameters
        ----------
        option_value : str
            The value to return; price or specified greek
        params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'


        Returns
        -------
        Price / Greek.

        """

        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        opt_params = {
            'S':self.params['S'],
            'K':self.params['K'],
            'T':self.params['T'],
            'r':self.params['r'],
            'q':self.params['q'],
            'sigma':self.params['sigma'],
            'option':self.params['option'],
            }

        try:
            # Select the chosen option value from the available functions
            function = self.params['greek_dict'][option_value]
            return getattr(Option, function)(
                opt_params=opt_params, params=self.params)

        except KeyError:
            return print("Please enter a valid function from 'price', "\
                   "'delta', 'gamma', 'vega', 'theta', 'rho', 'vomma', "\
                       "'vanna', 'zomma', 'speed', 'color', 'ultima', "\
                           "'vega bleed', 'charm'")


    def barrier(self, **kwargs):
        """
        Return the Barrier option price

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100.
        K : Float
            Strike Price. The default is 100.
        H : Float
            Barrier Level. The default is 105.
        R : Float
            Rebate. The default is 0.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        barrier_direction : Str
            Up or Down. The default is 'up'.
        knock : Str
            knock-in or knock-out. The default is 'in'.
        option : Str
            Option type, Put or Call. The default is 'call'


        Returns
        -------
        Float
            Barrier option price.

        """
        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        barrier_price, self.params = Barrier.barrier_price(params=self.params)

        return barrier_price


    def sensitivities(self, **kwargs):
        """
        Sensitivities of the option.

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100.
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        greek : Str
            Sensitivity to return. Select from 'delta', 'gamma', 'vega',
            'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 'color',
            'ultima', 'vega bleed', 'charm'. The default is 'delta'
        price_shift : Float
            The size of the price shift in decimal terms. The default
            is 0.25.
        vol_shift : Float
            The size of the volatility shift in decimal terms. The
            default is 0.001.
        ttm_shift : Float
            The size of the time to maturity shift in decimal terms. The
            default is 1/365.
        rate_shift : Float
            The size of the interest rate shift in decimal terms. The
            default is 0.0001.
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity.
            The default is False.

        Returns
        -------
        Float
            Option Sensitivity.

        """

        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        opt_params = {
            'S':self.params['S'],
            'K':self.params['K'],
            'T':self.params['T'],
            'r':self.params['r'],
            'q':self.params['q'],
            'sigma':self.params['sigma'],
            'option':self.params['option'],
            }

        if self.params['num_sens']:
            return Sens.numerical_sensitivities(
                opt_params=opt_params, params=self.params)

        return Sens.analytical_sensitivities(
            opt_params=opt_params, params=self.params)


    def visualize(self, **kwargs):
        """
        Plot the chosen graph of risk or payoff.

        Parameters
        ----------
        risk : Bool
            Whether to display risk graph or payoff graph. The default
            is True.
        S : Float
            Underlying Stock Price. Used in risk & payoff graphs. The
            default is 100.
        T : Float
            Time to Maturity. Used in risk & payoff graphs. The default
            is 0.25 (3 months).
        r : Float
            Interest Rate. Used in risk & payoff graphs. The default
            is 0.05 (5%).
        q : Float
            Dividend Yield. Used in risk & payoff graphs. The default
            is 0.
        sigma : Float
            Implied Volatility. Used in risk & payoff graphs. The default
            is 0.2 (20%).
        option : Str
            Option type, Put or Call. Used in risk & payoff graphs. The
            default is 'call'
        direction : Str
            Whether the payoff is long or short. Used in risk & payoff
            graphs. The default is 'long'.
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma',
            'vega', 'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed',
            'color', 'ultima', 'vega_bleed', 'charm'. Used in risk
            graphs. The default is 'delta'.
        graphtype : Str
            Whether to plot 2D or 3D graph. Used in risk graphs. The
            default is 2D.
        x_plot : Str
            The x-axis variable - 'price', 'strike', 'vol' or 'time'.
            Used in 2D-risk graphs. The default is 'price'.
        y_plot : Str
            The y-axis variable - 'value', 'delta', 'gamma', 'vega' or
            'theta'. Used in 2D-risk graphs. The default is 'delta.
        G1 : Float
            Strike Price of option 1. Used in 2D-risk graphs. The
            default is 90.
        G2 : Float
            Strike Price of option 2. Used in 2D-risk graphs. The
            default is 100.
        G3 : Float
            Strike Price of option 3. Used in 2D-risk graphs. The
            default is 110.
        T1 : Float
            Time to Maturity of option 1. Used in 2D-risk graphs. The
            default is 0.25 (3 months).
        T2 : Float
            Time to Maturity of option 1. Used in 2D-risk graphs. The
            default is 0.25 (3 months).
        T3 : Float
            Time to Maturity of option 1. Used in 2D-risk graphs. The
            default is 0.25 (3 months).
        time_shift : Float
            Difference between T1 and T2 in rho graphs. Used in 2D-risk
            graphs. The default is 0.25 (3 months).
        interactive : Bool
            Whether to show matplotlib (False) or plotly (True) graph.
            Used in 3D-risk graphs. The default is False.
        notebook : Bool
            Whether the function is being run in an IPython notebook and
            hence whether it should output in line or to an HTML file.
            Used in 3D-risk graphs. The default is False.
        colorscheme : Str
            The matplotlib colormap or plotly colorscale to use. Used
            in 3D-risk graphs. The default is 'jet' (which is a palette
            that works in both plotly and matplotlib).
        colorintensity : Float
            The alpha value indicating level of transparency /
            intensity. The default is 1.
        size2d : Tuple
            Figure size for matplotlib chart. Used in 2D-risk & payoff
            graphs. The default is (6, 4).
        size3d : Tuple
            Figure size for matplotlib chart. Used in 3D-risk graphs.
            The default is (15, 12).
        axis : Str
            Whether the x-axis is 'price' or 'vol'. Used in 3D-risk
            graphs. The default is 'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D
            graphs. Used in 3D-risk graphs. The default is 100.
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.
        K : Float
            Strike Price of option 1. Used in payoff graphs. The
            default is 100 (individual payoffs may have own defaults).
        K1 : Float
             Strike Price of option 1. Used in payoff graphs. The
             default is 95 (individual payoffs may have own defaults).
        K2 : Float
             Strike Price of option 2. Used in payoff graphs. The
             default is 105 (individual payoffs may have own defaults).
        K3 : Float
             Strike Price of option 3. Used in payoff graphs. The
             default is 105 (individual payoffs may have own defaults).
        K4 : Float
             Strike Price of option 4. Used in payoff graphs. The
             default is 105 (individual payoffs may have own defaults).
        cash : Bool
            Whether to discount forward to present value. Used in
            forward payoff graph. The default is False.
        ratio : Int
            Multiple of OTM options to be sold for ITM purchased. Used
            in backspread, ratio vertical spread payoff graphs. The
            default is 2.
        value : Bool
            Whether to show the current value as well as the terminal
            payoff. Used in payoff graphs. The default is False.
        combo_payoff : Str
            The payoff to be displayed. Used in payoff graphs. The
            default is 'straddle'.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity.
            The default is False.
        gif : Bool
            Whether to create an animated gif. The default is False.

        Returns
        -------
        Displays graph of either 2D / 3D greeks or payoff diagram.

        """

        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        # If a risk graph is selected
        if self.params['risk']:

            # Run 2D greeks method
            if self.params['graphtype'] == '2D':
                return Greeks.greeks_graphs_2D(params=self.params)

            # Run 3D greeks method
            if self.params['graphtype'] == '3D':
                return Greeks.greeks_graphs_3D(params=self.params)

        else:
            return self.payoffs(payoff_type=self.params['payoff_type'])


    def greeks(self, **kwargs):
        """
        Plot the chosen 2D or 3D graph

        Parameters
        ----------
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma',
            'vega', 'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed',
            'color', 'ultima', 'vega_bleed', 'charm'. The default is
            'delta'
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
        interactive : Bool
            Whether to show matplotlib (False) or plotly (True) graph.
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
        size2d : Tuple
            Figure size for matplotlib chart. The default is (6, 4).
        size3d : Tuple
            Figure size for matplotlib chart. The default is (15, 12).
        axis : Str
            Whether the x-axis is 'price' or 'vol'. The default is
            'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D
            graphs. The default is 100.
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.
        graphtype : Str
            Whether to plot 2D or 3D graph. The default is 2D.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity.
            The default is False.
        gif : Bool
            Whether to create an animated gif. The default is False.

        Returns
        -------
        Runs method to display either 2D or 3D greeks graph.

        """

        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        # Run 2D greeks method
        if self.params['graphtype'] == '2D':
            Greeks.greeks_graphs_2D(params=self.params)

        # Run 3D greeks method
        elif self.params['graphtype'] == '3D':
            Greeks.greeks_graphs_3D(params=self.params)

        else:
            print("Please select a '2D' or '3D' graphtype")


    def payoffs(self, payoff_type, **kwargs):
        """
        Displays the graph of the specified combo payoff.

        Parameters
        ----------
        payoff_type : str
            The payoff to be displayed.
        **kwargs : Dict
            S : Float
                 Underlying Stock Price. The default is 100.
            K : Float
                Strike Price of option 1. The default is 100
                (individual payoffs may have own defaults).
            K1 : Float
                 Strike Price of option 1. The default is 95
                 (individual payoffs may have own defaults).
            K2 : Float
                 Strike Price of option 2. The default is 105
                 (individual payoffs may have own defaults).
            K3 : Float
                 Strike Price of option 3. The default is 105
                 (individual payoffs may have own defaults).
            K4 : Float
                 Strike Price of option 4. The default is 105
                 (individual payoffs may have own defaults).
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
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
            cash : Bool
                Whether to discount forward to present value. The default
                is False.
            ratio : Int
                Multiple of OTM options to be sold for ITM purchased. The
                default is 2.
            value : Bool
                Whether to show the current value as well as the terminal
                payoff. The default is False.
            combo_payoff : Str
                The payoff to be displayed.
            mpl_style : Str
                Matplotlib style template for 2D risk charts and payoffs.
                The default is 'seaborn-darkgrid'.

        Returns
        -------
        Runs the specified combo payoff method.

        """
        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        # Specify the combo payoff so that parameter initialisation
        # takes into account specific defaults
        self.params['combo_payoff'] = payoff_type

        # Update pricing input parameters to default if not supplied
        self.params = Utils.refresh_combo_params(
            params=self.params, inputs=kwargs)

        if payoff_type in self.params['combo_simple_dict']:

            # Select the chosen payoff from the available functions
            function = self.params['combo_name_dict'][payoff_type]
            return getattr(SimplePayoff, function)(params=self.params)

        if payoff_type in self.params['combo_multi_dict']:

            # Select the chosen payoff from the available functions
            function = self.params['combo_name_dict'][payoff_type]
            return getattr(MultiPayoff, function)(params=self.params)

        # Otherwise prompt for a valid payoff
        return print("Please enter a valid payoff from 'call', 'put', "\
               "'stock', 'forward', 'collar', 'spread', 'backspread', "\
                   "'ratio vertical spread', 'straddle', 'strangle', "\
                       "'butterfly', 'christmas tree', 'condor', "\
                           "'iron butterfly', 'iron condor'")


    def animated_gif(self, graphtype, **kwargs):
        """
        Create an animated gif of the selected pair of parameters or the
        selected greek 3D graph..

        Parameters
        ----------
        **kwargs : Dict
            gif_folder : Str
                The folder to save the files into. The default is
                'images/greeks'.
            gif_filename : Str
                The filename for the animated gif. The default is 'greek'.
            T : Float
                Time to Maturity. The default is 0.5 (6 months).
            steps : Int
                Number of images to combine. The default is 40.
            x_plot : Str
                     The x-axis variable ('price', 'strike' or 'vol').
                     The default is 'price'.
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
            greek : Str
                The sensitivity to be charted. Select from 'delta', 'gamma',
                'vega', 'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed',
                'color', 'ultima', 'vega_bleed', 'charm'. The default is
                'delta'
            gif_folder : Str
                The folder to save the files into. The default is
                'images/greeks'.
            gif_filename : Str
                The filename for the animated gif. The default is 'greek'.
            gif_frame_update : Int
                The number of degrees of rotation between each frame used to
                construct the animated gif. The default is 2.
            gif_min_dist : Float
                The minimum zoom distance. The default is 9.0.
            gif_max_dist : Float
                The maximum zoom distance. The default is 10.0.
            gif_min_elev : Float
                The minimum elevation. The default is 10.0.
            gif_max_elev : Float
                The maximum elevation. The default is 60.0.
            gif_start_azim : Float
                The azimuth to start the gif from. The default is 0.
            gif_end_azim : Float
                The azimuth to end the gif on. The default is 360.
            gif_dpi : Int
                The image resolution to save. The default is 50 dpi.
            gif_ms : Int
                The time to spend on each frame in the gif. The default is
                100ms.

        Returns
        -------
        Saves an animated gif.

        """
        self.params = copy.deepcopy(vis_params_dict)

        self.params['graphtype'] = graphtype
        self.params['gif'] = True

        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        # Create a 2D animated gif
        if self.params['graphtype'] == '2D':
            return Gif.animated_2D_gif(params=self.params)

        # Create a 3D animated gif
        if self.params['graphtype'] == '3D':
            return Gif.animated_3D_gif(params=self.params)

        # Otherwise prompt for a valid gif type
        return print("Please select gif_type as '2D' or '3D'")
