"""
Calculate Analytical or Numerical sensitivities

"""

from optionvisualizer.option_formulas import Option
# pylint: disable=invalid-name

class Sens():
    """
    Summary functions for calculating sensitivities

    """
    @classmethod
    def sensitivities_static(cls, params, **kwargs):
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

        opt_params = {}
        op_keys = ['S','K', 'T', 'r', 'q', 'sigma', 'option']

        # Update params with the specified parameters
        for key, value in kwargs.items():

            if key not in ['params']:
                if key in op_keys:
                    # Add to the option parameter dictionary
                    opt_params[key] = value
                # Replace the default parameter with that provided
                else:
                    params[key] = value

        for parameter in op_keys:
            if parameter not in opt_params:
                opt_params[parameter] = params[parameter]

        if params['num_sens']:
            return cls.numerical_sensitivities(
                opt_params=opt_params, params=params)

        return cls.analytical_sensitivities(
            opt_params=opt_params, params=params)


    @staticmethod
    def analytical_sensitivities(opt_params, params):
        """
        Sensitivities of the option calculated analytically from closed
        form solutions.

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

        Returns
        -------
        Float
            Option Sensitivity.

        """

        for key, value in params['greek_dict'].items():
            if str(params['greek']) == key:
                return getattr(Option, value)(
                    opt_params=opt_params, params=params)

        return print("Please enter a valid Greek")


    @classmethod
    def numerical_sensitivities(cls, opt_params, params):
        """
        Sensitivities of the option calculated numerically using shifts
        in parameters.

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

        Returns
        -------
        Float
            Option Sensitivity.

        """

        for key, value in params['greek_dict'].items():
            if key == params['greek']:
                return getattr(cls, '_num_'+value)(
                    opt_params=opt_params, params=params)

        return print("Please enter a valid Greek")


    @classmethod
    def _num_price(cls, opt_params, params):
        option_names = []
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return opt_dict['opt']


    @classmethod
    def _num_delta(cls, opt_params, params):
        option_names = ['price_up', 'price_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return ((opt_dict['opt_price_up'] - opt_dict['opt_price_down'])
                / (2 * params['price_shift']))


    @classmethod
    def _num_gamma(cls, opt_params, params):
        option_names = ['price_up', 'price_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return ((opt_dict['opt_price_up']
                 - (2 * opt_dict['opt'])
                 + opt_dict['opt_price_down'])
                / (params['price_shift'] ** 2))


    @classmethod
    def _num_vega(cls, opt_params, params):
        option_names = ['vol_up', 'vol_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return (((opt_dict['opt_vol_up'] - opt_dict['opt_vol_down'])
                 / (2 * params['vol_shift'])) / 100)


    @classmethod
    def _num_theta(cls, opt_params, params):
        option_names = ['ttm_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return ((opt_dict['opt_ttm_down'] - opt_dict['opt'])
                / (params['ttm_shift'] * 100))


    @classmethod
    def _num_rho(cls, opt_params, params):
        option_names = ['rate_up', 'rate_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return ((opt_dict['opt_rate_up'] - opt_dict['opt_rate_down'])
                / (2 * params['rate_shift'] * 10000))


    @classmethod
    def _num_vomma(cls, opt_params, params):
        option_names = ['vol_up', 'vol_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return (((opt_dict['opt_vol_up']
                  - (2 * opt_dict['opt'])
                  + opt_dict['opt_vol_down'])
                 / (params['vol_shift'] ** 2)) / 10000)


    @classmethod
    def _num_vanna(cls, opt_params, params):
        option_names = ['price_up_vol_up', 'price_up_vol_down',
                        'price_down_vol_up', 'price_down_vol_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return (((1 / (4 * params['price_shift'] * params['vol_shift']))
                 * (opt_dict['opt_price_up_vol_up']
                    - opt_dict['opt_price_up_vol_down']
                    - opt_dict['opt_price_down_vol_up']
                    + opt_dict['opt_price_down_vol_down'])) / 100)


    @classmethod
    def _num_charm(cls, opt_params, params):
        option_names = ['price_up', 'price_down',
                        'price_up_ttm_down', 'price_down_ttm_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return ((((opt_dict['opt_price_up_ttm_down']
                   - opt_dict['opt_price_down_ttm_down'])
                  / (2 * params['price_shift']))
                     - ((opt_dict['opt_price_up']
                         - opt_dict['opt_price_down'])
                        / (2 * params['price_shift'])))
                    / (params['ttm_shift'] * 100))


    @classmethod
    def _num_zomma(cls, opt_params, params):
        option_names = ['vol_up', 'vol_down', 'price_up_vol_up',
                        'price_up_vol_down', 'price_down_vol_up',
                        'price_down_vol_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return (((opt_dict['opt_price_up_vol_up']
                  - (2 * opt_dict['opt_vol_up'])
                  + opt_dict['opt_price_down_vol_up'])
                 - opt_dict['opt_price_up_vol_down']
                 + (2 * opt_dict['opt_vol_down'])
                 - opt_dict['opt_price_down_vol_down'])
                / (2 * params['vol_shift'] * (params['price_shift'] ** 2))
                / 100)


    @classmethod
    def _num_speed(cls, opt_params, params):
        option_names = ['price_up', 'price_down', 'double_price_up']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return (1 / (params['price_shift'] ** 3)
                * (opt_dict['opt_double_price_up']
                   - (3 * opt_dict['opt_price_up'])
                   + 3 * opt_dict['opt']
                   - opt_dict['opt_price_down']))


    @classmethod
    def _num_color(cls, opt_params, params):
        option_names = ['price_up', 'price_down', 'price_up_ttm_down',
                        'price_down_ttm_down', 'ttm_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return ((((opt_dict['opt_price_up_ttm_down']
                   - (2 * opt_dict['opt_ttm_down'])
                   + opt_dict['opt_price_down_ttm_down'])
                  / (params['price_shift'] ** 2))
                 - ((opt_dict['opt_price_up']
                     - (2 * opt_dict['opt'])
                     + opt_dict['opt_price_down'])
                    / (params['price_shift'] ** 2) ))
                / (params['ttm_shift'] * 100))


    @classmethod
    def _num_ultima(cls, opt_params, params):
        option_names = ['vol_up', 'vol_down', 'double_vol_up']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return ((1 / (params['vol_shift'] ** 3)
                 * (opt_dict['opt_double_vol_up']
                    - (3 * opt_dict['opt_vol_up'])
                    + 3 * opt_dict['opt']
                    - opt_dict['opt_vol_down']))
                * (params['vol_shift'] ** 2))


    @classmethod
    def _num_vega_bleed(cls, opt_params, params):
        option_names = ['vol_up', 'vol_down', 'vol_up_ttm_down',
                        'vol_down_ttm_down']
        opt_dict = cls._option_prices(
            opt_params=opt_params, params=params, option_names=option_names)
        return ((((opt_dict['opt_vol_up_ttm_down']
                   - opt_dict['opt_vol_down_ttm_down'])
                  / (2 * params['vol_shift']))
                 - ((opt_dict['opt_vol_up']
                     - opt_dict['opt_vol_down'])
                    / (2 * params['vol_shift'])))
                / (params['ttm_shift'] * 10000))


    @classmethod
    def _option_prices(cls, opt_params, params, option_names):

        if params['greek'] in params['equal_greeks']:
            opt_params['option'] = 'call'

        shift_dict = cls._parameter_shifts(
            opt_params=opt_params, params=params)

        opt_dict = {}
        opt_dict['opt'] = Option.price(opt_params=opt_params, params=params)

        for option_name, option_params in shift_dict.items():
            if option_name in option_names:
                opt_dict['opt_'+option_name] = Option.price(
                opt_params=option_params, params=params)

        return opt_dict


    @staticmethod
    def _parameter_shifts(opt_params, params):

        shift_dict = {}
        shift_inputs = {'price':'S',
                        'vol':'sigma',
                        'ttm':'T',
                        'rate':'r'}

        for shift, shift_param in shift_inputs.items():
            shift_dict[shift+'_up'] = {
                **opt_params, shift_param:(
                    opt_params[shift_param] + params[shift+'_shift'])}
            shift_dict[shift+'_down'] = {
                **opt_params, shift_param:(
                    opt_params[shift_param] - params[shift+'_shift'])}

        shift_dict['price_up_vol_up'] ={
            **opt_params,
            'S':(opt_params['S'] + params['price_shift']),
            'sigma':(opt_params['sigma'] + params['vol_shift'])}

        shift_dict['price_down_vol_up'] ={
            **opt_params,
            'S':(opt_params['S'] - params['price_shift']),
            'sigma':(opt_params['sigma'] + params['vol_shift'])}

        shift_dict['price_up_vol_down'] ={
            **opt_params,
            'S':(opt_params['S'] + params['price_shift']),
            'sigma':(opt_params['sigma'] - params['vol_shift'])}

        shift_dict['price_down_vol_down'] ={
            **opt_params,
            'S':(opt_params['S'] - params['price_shift']),
            'sigma':(opt_params['sigma'] - params['vol_shift'])}

        shift_dict['price_up_ttm_down'] ={
            **opt_params,
            'S':(opt_params['S'] + params['price_shift']),
            'T':(opt_params['T'] - params['ttm_shift'])}

        shift_dict['price_down_ttm_down'] ={
            **opt_params,
            'S':(opt_params['S'] - params['price_shift']),
            'T':(opt_params['T'] - params['ttm_shift'])}

        shift_dict['vol_up_ttm_down'] ={
            **opt_params,
            'sigma':(opt_params['sigma'] + params['vol_shift']),
            'T':(opt_params['T'] - params['ttm_shift'])}

        shift_dict['vol_down_ttm_down'] ={
            **opt_params,
            'sigma':(opt_params['sigma'] - params['vol_shift']),
            'T':(opt_params['T'] - params['ttm_shift'])}

        shift_dict['double_price_up'] = {
            **opt_params,
            'S':(opt_params['S'] + 2 * params['price_shift'])}

        shift_dict['double_vol_up'] = {
            **opt_params,
            'sigma':(opt_params['sigma'] + 2 * params['vol_shift'])}

        return shift_dict


    @staticmethod
    def option_data_static(params, option_value, **kwargs):
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
        opt_params ={}

        # Update params with the specified parameters
        for key, value in kwargs.items():
            if key not in ['params']:
                if key in ['S','K', 'T', 'r', 'q', 'sigma', 'option']:
                    # Add to the option parameter dictionary
                    opt_params[key] = value
                # Replace the default parameter with that provided
                else:
                    params[key] = value

        try:
            # Select the chosen option value from the available functions
            function = params['greek_dict'][option_value]
            return getattr(Option, function)(
                opt_params=opt_params, params=params)

        except KeyError:
            return print("Please enter a valid function from 'price', "\
                   "'delta', 'gamma', 'vega', 'theta', 'rho', 'vomma', "\
                       "'vanna', 'zomma', 'speed', 'color', 'ultima', "\
                           "'vega bleed', 'charm'")
