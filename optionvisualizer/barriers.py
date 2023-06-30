"""
Calculate barrier option

"""

from optionvisualizer.utilities import Utils
# pylint: disable=invalid-name

class Barrier():
    """
    Calculate barrier option

    """
    @classmethod
    def barrier_price(cls, params: dict) -> tuple[float, dict]:
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
        default : Bool
            Whether the function is being called directly (in which
            case values that are not supplied are set to default
            values) or used within a graph call where they have
            already been updated.

        Returns
        -------
        Float
            Barrier option price.

        """

        # Down and In Call
        if (params['barrier_direction'] == 'down'
            and params['knock'] == 'in'
            and params['option'] == 'call'):

            opt_barrier_payoff = cls._di_call(params)


        # Up and In Call
        if (params['barrier_direction'] == 'up'
                and params['knock'] == 'in'
                and params['option'] == 'call'):

            opt_barrier_payoff = cls._ui_call(params)


        # Down and In Put
        if (params['barrier_direction'] == 'down'
                and params['knock'] == 'in'
                and params['option'] == 'put'):

            opt_barrier_payoff = cls._di_put(params)


        # Up and In Put
        if (params['barrier_direction'] == 'up'
            and params['knock'] == 'in'
            and params['option'] == 'put'):

            opt_barrier_payoff = cls._ui_put(params)


        # Down and Out Call
        if (params['barrier_direction'] == 'down'
            and params['knock'] == 'out'
            and params['option'] == 'call'):

            opt_barrier_payoff = cls._do_call(params)


        # Up and Out Call
        if (params['barrier_direction'] == 'up'
            and params['knock'] == 'out'
            and params['option'] == 'call'):

            opt_barrier_payoff = cls._uo_call(params)


        # Down and Out Put
        if (params['barrier_direction'] == 'down'
            and params['knock'] == 'out'
            and params['option'] == 'put'):

            opt_barrier_payoff = cls._do_put(params)

        # Up and Out Put
        if (params['barrier_direction'] == 'up'
            and params['knock'] == 'out'
            and params['option'] == 'put'):

            opt_barrier_payoff = cls._uo_put(params)

        return opt_barrier_payoff, params


    @staticmethod
    def _di_call(params: dict) -> float:

        params['eta'] = 1
        params['phi'] = 1

        barrier_factors, params = Utils.barrier_factors(params=params)

        if params['K'] > params['H']:
            opt_barrier_payoff = (
                barrier_factors['C'] + barrier_factors['E'])
        if params['K'] < params['H']:
            opt_barrier_payoff = (
                barrier_factors['A'] - barrier_factors['B']
                + barrier_factors['D'] + barrier_factors['E'])

        return opt_barrier_payoff


    @staticmethod
    def _ui_call(params: dict) -> float:

        params['eta'] = -1
        params['phi'] = 1

        barrier_factors, params = Utils.barrier_factors(params=params)

        if params['K'] > params['H']:
            opt_barrier_payoff = (
                barrier_factors['A'] + barrier_factors['E'])
        if params['K'] < params['H']:
            opt_barrier_payoff = (
                barrier_factors['B'] - barrier_factors['C']
                + barrier_factors['D'] + barrier_factors['E'])

        return opt_barrier_payoff


    @staticmethod
    def _di_put(params: dict) -> float:

        params['eta'] = 1
        params['phi'] = -1

        barrier_factors, params = Utils.barrier_factors(params=params)

        if params['K'] > params['H']:
            opt_barrier_payoff = (
                barrier_factors['B'] - barrier_factors['C']
                + barrier_factors['D'] + barrier_factors['E'])
        if params['K'] < params['H']:
            opt_barrier_payoff = (
                barrier_factors['A'] + barrier_factors['E'])

        return opt_barrier_payoff


    @staticmethod
    def _ui_put(params: dict) -> float:

        params['eta'] = -1
        params['phi'] = -1

        barrier_factors, params = Utils.barrier_factors(params=params)

        if params['K'] > params['H']:
            opt_barrier_payoff = (
                barrier_factors['A'] - barrier_factors['B']
                + barrier_factors['D'] + barrier_factors['E'])
        if params['K'] < params['H']:
            opt_barrier_payoff = (
                barrier_factors['C'] + barrier_factors['E'])

        return opt_barrier_payoff


    @staticmethod
    def _do_call(params: dict) -> float:

        params['eta'] = 1
        params['phi'] = 1

        barrier_factors, params = Utils.barrier_factors(params=params)

        if params['K'] > params['H']:
            opt_barrier_payoff = (
                barrier_factors['A'] - barrier_factors['C']
                + barrier_factors['F'])
        if params['K'] < params['H']:
            opt_barrier_payoff = (
                barrier_factors['B'] - barrier_factors['D']
                + barrier_factors['F'])

        return opt_barrier_payoff


    @staticmethod
    def _uo_call(params: dict) -> float:

        params['eta'] = -1
        params['phi'] = 1

        barrier_factors, params = Utils.barrier_factors(params=params)

        if params['K'] > params['H']:
            opt_barrier_payoff = barrier_factors['F']
        if params['K'] < params['H']:
            opt_barrier_payoff = (
                barrier_factors['A'] - barrier_factors['B']
                + barrier_factors['C'] - barrier_factors['D']
                + barrier_factors['F'])

        return opt_barrier_payoff


    @staticmethod
    def _do_put(params: dict) -> float:

        params['eta'] = 1
        params['phi'] = -1

        barrier_factors, params = Utils.barrier_factors(params=params)

        if params['K'] > params['H']:
            opt_barrier_payoff = (
                barrier_factors['A'] - barrier_factors['B']
                + barrier_factors['C'] - barrier_factors['D']
                + barrier_factors['F'])
        if params['K'] < params['H']:
            opt_barrier_payoff = barrier_factors['F']

        return opt_barrier_payoff


    @staticmethod
    def _uo_put(params: dict) -> float:

        params['eta'] = -1
        params['phi'] = -1

        barrier_factors, params = Utils.barrier_factors(params=params)

        if params['K'] > params['H']:
            opt_barrier_payoff = (
                barrier_factors['B'] - barrier_factors['D']
                + barrier_factors['F'])
        if params['K'] < params['H']:
            opt_barrier_payoff = (
                barrier_factors['A'] - barrier_factors['C']
                + barrier_factors['F'])

        return opt_barrier_payoff
