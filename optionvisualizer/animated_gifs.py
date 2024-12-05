"""
Create Animated Gifs

"""
import glob
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from optionvisualizer.greeks import Greeks
# pylint: disable=invalid-name, consider-using-f-string, unused-variable

class Gif():
    """
    Create Animated Gifs

    """
    @classmethod
    def animated_2D_gif(cls, params: dict) -> None:
        """
        Create an animated gif of the selected pair of parameters.

        Parameters
        ----------
        params : Dict
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
                The default is 'seaborn-v0_8-darkgrid'.
            num_sens : Bool
                Whether to calculate numerical or analytical sensitivity.
                The default is False.

        Returns
        -------
        Saves an animated gif.

        """
        params['gif']=True

        # Set up folders to save files
        params = cls._gif_defaults_setup(params=params)

        # split the countdown from T to maturity in steps equal steps
        time_steps = np.linspace(params['T'], 0.001, params['steps'])

        # create a plot for each time_step
        for counter, step in enumerate(time_steps):

            # create filenumber and filename
            filenumber = '{:03d}'.format(counter)
            filename = '{}, {}'.format(
                params['gif_filename'], filenumber)

            params['T'] = step
            params['T1'] = step
            params['T2'] = step
            params['T3'] = step

            # call the greeks_graphs_2d function to create graph
            fig, ax = Greeks.greeks_graphs_2D(
                params=params)

            # save the image as a file
            plt.savefig('{}/{}/img{}.png'.format(
                params['gif_folder'], params['gif_filename'], filename),
                dpi=50)

            # close the image object
            plt.close()

        cls._create_animation(params)


    @classmethod
    def animated_3D_gif(cls, params: dict) -> None:
        """
        Create an animated gif of the selected greek 3D graph.

        Parameters
        ----------
        params : Dict
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
            direction : Str
                Whether the payoff is long or short. The default is 'long'.
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
            num_sens : Bool
                Whether to calculate numerical or analytical sensitivity.
                The default is False.
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

        params = cls._gif_defaults_setup(params=params)

        fig, ax, params['titlename'], \
            params['title_font_scale'] = Greeks.greeks_graphs_3D(
                params=params)

        # set the range for horizontal rotation
        params['azim_range'] = (
            params['gif_end_azim'] - params['gif_start_azim'])

        # set number of frames for the animated gif as the range of horizontal
        # rotation divided by the number of degrees rotation between frames
        params['steps'] = math.floor(
            params['azim_range']/params['gif_frame_update'])

        # a viewing perspective is composed of an elevation, distance, and
        # azimuth define the range of values we'll cycle through for the
        # distance of the viewing perspective
        params['dist_range'] = np.arange(
            params['gif_min_dist'],
            params['gif_max_dist'],
            (params['gif_max_dist']-params['gif_min_dist'])/params['steps'])

        # define the range of values we'll cycle through for the elevation of
        # the viewing perspective
        params['elev_range'] = np.arange(
            params['gif_max_elev'],
            params['gif_min_elev'],
            (params['gif_min_elev']-params['gif_max_elev'])/params['steps'])

        # now create the individual frames that will be combined later into the
        # animation
        for idx, azimuth in enumerate(
                range(params['gif_start_azim'],
                      params['gif_end_azim'],
                      params['gif_frame_update'])):

            # pan down, rotate around, and zoom out
            ax.azim = float(azimuth)
            ax.elev = params['elev_range'][idx]
            ax.dist = params['dist_range'][idx]

            # set the figure title
            st = fig.suptitle(params['titlename'],
                              fontsize=params['title_font_scale'],
                              fontweight=0,
                              color='black',
                              style='italic',
                              y=1.02)

            st.set_y(0.98)
            fig.subplots_adjust(top=1)

            # save the image as a png file
            plt.savefig('{}/{}/img{:03d}.png'.format(
                params['gif_folder'],
                params['gif_filename'],
                azimuth),
                dpi=params['gif_dpi'])

        # close the image object
        plt.close()

        cls._create_animation(params)


    @staticmethod
    def _gif_defaults_setup(params: dict) -> dict:

        if params['gif_folder'] is None:
            params['gif_folder'] = params['gif_folder_'+params['graphtype']]
        if params['gif_filename'] is None:
            params['gif_filename'] = params['gif_filename_'+params['graphtype']]

        params['working_folder'] = '{}/{}'.format(
            params['gif_folder'], params['gif_filename'])
        if not os.path.exists(params['working_folder']):
            os.makedirs(params['working_folder'])

        return params


    @staticmethod
    def _create_animation(params: dict) -> None:

        # load all the static images into a list then save as an animated gif
        gif_filepath = '{}/{}.gif'.format(
            params['gif_folder'], params['gif_filename'])
        images = ([Image.open(image) for image in sorted(
            glob.glob('{}/*.png'.format(params['working_folder'])))])
        gif = images[0]
        gif.info['duration'] = params['gif_ms'] #milliseconds per frame
        gif.info['loop'] = 0 #how many times to loop (0=infinite)
        gif.save(fp=gif_filepath, format='gif', save_all=True,
                 append_images=images[1:])
