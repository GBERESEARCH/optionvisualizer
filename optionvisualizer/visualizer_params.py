"""
Key parameters for visualizer

"""

# pylint: disable=invalid-name

# Dictionary of default parameters
vis_params_dict = {
    'S':100,
    'K':100,
    'K1':95,
    'K2':105,
    'K3':105,
    'K4':105,
    'G1':90,
    'G2':100,
    'G3':110,
    'H':105,
    'R':0,
    'T':0.25,
    'T1':0.25,
    'T2':0.25,
    'T3':0.25,
    'T4':0.25,
    'r':0.005,
    'b':0.005,
    'q':0,
    'sigma':0.2,
    'eta':1,
    'phi':1,
    'C4':None,
    'label4':None,
    'barrier_direction':'down',
    'knock':'in',
    'option':'call',
    'option1':'call',
    'option2':'call',
    'option3':'call',
    'option4':'call',
    'direction':'long',
    'value':False,
    'ratio':2,
    'refresh':'Std',
    'combo_payoff':'straddle',
    'payoff_type':'straddle',
    'price_shift':0.25,
    'price_shift_type':'avg',
    'vol_shift':0.001,
    'ttm_shift':1/365,
    'rate_shift':0.0001,
    'greek':'delta',
    'num_sens':False,
    'interactive':False,
    'notebook':True,
    'web_graph':False,
    'colorscheme':'jet',
    'colorintensity':1,
    'size3d':(15, 12),
    'size2d':(8, 6),
    'graphtype':'2D',
    'y_plot':'delta',
    'x_plot':'price',
    'time_shift':0.25,
    'cash':False,
    'axis':'price',
    'spacegrain':100,
    'azim':-60,
    'elev':30,
    'risk':True,
    'gif':False,
    'gif_folder':None,
    'gif_filename':None,
    'gif_folder_2D':'images/greeks2D',
    'gif_filename_2D':'greek2D',
    'gif_folder_3D':'images/greeks3D',
    'gif_filename_3D':'greek3D',
    'gif_frame_update':2,
    'gif_min_dist':9.0,
    'gif_max_dist':10.0,
    'gif_min_elev':10.0,
    'gif_max_elev':60.0,
    'gif_dpi':50,
    'gif_ms':100,
    'gif_start_azim':0,
    'gif_end_azim':360,
    'mpl_style':'seaborn-darkgrid',
    'steps':40,
    'titlename':None,
    'title_font_scale':None,

     # List of default parameters used when refreshing
     'params_list':[
         'S', 'K', 'K1', 'K2', 'K3', 'K4', 'G1', 'G2', 'G3', 'H', 'R',
         'T', 'T1', 'T2', 'T3', 'T4', 'r', 'b', 'q', 'sigma', 'eta',
         'phi', 'barrier_direction', 'knock', 'option', 'option1',
         'option2', 'option3', 'option4', 'direction', 'value',
         'ratio', 'refresh', 'price_shift', 'price_shift_type',
         'vol_shift','ttm_shift', 'rate_shift', 'greek', 'num_sens',
         'interactive', 'notebook', 'colorscheme', 'colorintensity',
         'size3d', 'size2d', 'graphtype', 'cash', 'axis', 'spacegrain',
         'azim', 'elev', 'risk', 'mpl_style'
         ],

     # List of Greeks where call and put values are the same
     'equal_greeks':[
         'gamma',
         'vega',
         'vomma',
         'vanna',
         'zomma',
         'speed',
         'color',
         'ultima',
         'vega bleed'
         ],

     # Payoffs requiring changes to default parameters
     'mod_payoffs':[
         'call',
         'put',
         'collar',
         'straddle',
         'butterfly',
         'christmas tree',
         'condor',
         'iron butterfly',
         'iron condor'
         ],

     # Those parameters that need changing
     'mod_params':[
         'S',
         'K',
         'K1',
         'K2',
         'K3',
         'K4',
         'T',
         'T1',
         'T2',
         'T3',
         'T4'
         ],

     # Combo parameter values differing from standard defaults
     'combo_dict':{
         'call':{
             'S':100,
             'K':100,
             'K1':100,
             'T1':0.25
             },
         'put':{
             'S':100,
             'K':100,
             'K1':100,
             'T1':0.25
             },
         'collar':{
             'S':100,
             'K':100,
             'K1':98,
             'K2':102,
             'T1':0.25,
             'T2':0.25
             },
         'straddle':{
             'S':100,
             'K':100,
             'K1':100,
             'K2':100,
             'T1':0.25,
             'T2':0.25
             },
         'butterfly':{
             'S':100,
             'K':100,
             'K1':95,
             'K2':100,
             'K3':105,
             'T1':0.25,
             'T2':0.25,
             'T3':0.25
             },
         'christmas tree':{
             'S':100,
             'K':100,
             'K1':95,
             'K2':100,
             'K3':105,
             'T1':0.25,
             'T2':0.25,
             'T3':0.25
             },
         'condor':{
             'S':100,
             'K':100,
             'K1':90,
             'K2':95,
             'K3':100,
             'K4':105,
             'T1':0.25,
             'T2':0.25,
             'T3':0.25,
             'T4':0.25
             },
         'iron butterfly':{
             'S':100,
             'K':100,
             'K1':95,
             'K2':100,
             'K3':100,
             'K4':105,
             'T1':0.25,
             'T2':0.25,
             'T3':0.25,
             'T4':0.25
             },
         'iron condor':{
             'S':100,
             'K':100,
             'K1':90,
             'K2':95,
             'K3':100,
             'K4':105,
             'T1':0.25,
             'T2':0.25,
             'T3':0.25,
             'T4':0.25
             }
         },

    'combo_parameters':{
        'call':[
            'S',
            'K',
            'T',
            'r',
            'q',
            'sigma',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        'put':[
            'S',
            'K',
            'T',
            'r',
            'q',
            'sigma',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        'stock':[
            'S',
            'direction',
            'mpl_style',
            'size2d'
            ],
        'forward':[
            'S',
            'T',
            'r',
            'q',
            'sigma',
            'direction',
            'cash',
            'mpl_style',
            'size2d'
            ],
        'collar':[
            'S',
            'K1',
            'K2',
            'T',
            'r',
            'q',
            'sigma',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        'spread':[
            'S',
            'K1',
            'K2',
            'T',
            'r',
            'q',
            'sigma',
            'option',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        'backspread':[
            'S',
            'K1',
            'K2',
            'T',
            'r',
            'q',
            'sigma',
            'option',
            'ratio',
            'value',
            'mpl_style',
            'size2d'
            ],
        'ratio vertical spread':[
            'S',
            'K1',
            'K2',
            'T',
            'r',
            'q',
            'sigma',
            'option',
            'ratio',
            'value',
            'mpl_style',
            'size2d'
            ],
        'straddle':[
            'S',
            'K',
            'T',
            'r',
            'q',
            'sigma',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        'strangle':[
            'S',
            'K1',
            'K2',
            'T',
            'r',
            'q',
            'sigma',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        'butterfly':[
            'S',
            'K1',
            'K2',
            'K3',
            'T',
            'r',
            'q',
            'sigma',
            'option',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        'christmas tree':[
            'S',
            'K1',
            'K2',
            'K3',
            'T',
            'r',
            'q',
            'sigma',
            'option',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        'condor':[
            'S',
            'K1',
            'K2',
            'K3',
            'K4',
            'T',
            'r',
            'q',
            'sigma',
            'option',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        'iron butterfly':[
            'S',
            'K1',
            'K2',
            'K3',
            'K4',
            'T',
            'r',
            'q',
            'sigma',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        'iron condor':[
            'S',
            'K1',
            'K2',
            'K3',
            'K4',
            'T',
            'r',
            'q',
            'sigma',
            'direction',
            'value',
            'mpl_style',
            'size2d'
            ],
        },

    'combo_name_dict':{
        'call':'call',
        'put':'put',
        'stock':'stock',
        'forward':'forward',
        'collar':'collar',
        'spread':'spread',
        'backspread':'backspread',
        'ratio vertical spread':'ratio_vertical_spread',
        'straddle':'straddle',
        'strangle':'strangle',
        'butterfly':'butterfly',
        'christmas tree':'christmas_tree',
        'condor':'condor',
        'iron butterfly':'iron_butterfly',
        'iron condor':'iron_condor',
        },

    'combo_simple_dict':{
        'call':'call',
        'put':'put',
        'stock':'stock',
        'forward':'forward',
        'collar':'collar',
        'spread':'spread',
        'backspread':'backspread',
        'ratio vertical spread':'ratio_vertical_spread',
        'straddle':'straddle',
        'strangle':'strangle',
        },

    'combo_multi_dict':{
        'butterfly':'butterfly',
        'christmas tree':'christmas_tree',
        'condor':'condor',
        'iron butterfly':'iron_butterfly',
        'iron condor':'iron_condor',
        },

     # Dictionary mapping function parameters to x axis labels
     # for 2D graphs
     'x_name_dict':{
         'price':'SA',
         'strike':'SA',
         'vol':'sigmaA',
         'time':'TA'
         },

     # Dictionary mapping scaling parameters to x axis labels
     # for 2D graphs
     'x_scale_dict':{
         'price':1,
         'strike':1,
         'vol':100,
         'time':365
         },

     # Dictionary mapping function parameters to y axis labels
     # for 2D graphs
     'y_name_dict':{
         'value':'price',
         'delta':'delta',
         'gamma':'gamma',
         'vega':'vega',
         'theta':'theta'
         },

     # Dictionary mapping function parameters to axis labels
     # for 3D graphs
     'label_dict':{
         'price':'Underlying Price',
         'value':'Theoretical Value',
         'vol':'Volatility %',
         'time':'Time to Expiration (Days)',
         'delta':'Delta',
         'gamma':'Gamma',
         'vega':'Vega',
         'theta':'Theta',
         'rho':'Rho',
         'strike':'Strike Price'
         },

     # Ranges of Underlying price and Time to Expiry for 3D
     # greeks graphs
     '3D_chart_ranges':{
         'price':{
             'SA_lower':0.8,
             'SA_upper':1.2,
             'TA_lower':0.01,
             'TA_upper':1
             },
         'delta':{
             'SA_lower':0.25,
             'SA_upper':1.75,
             'TA_lower':0.01,
             'TA_upper':2
             },
         'gamma':{
             'SA_lower':0.8,
             'SA_upper':1.2,
             'TA_lower':0.01,
             'TA_upper':5
             },
         'vega':{
             'SA_lower':0.5,
             'SA_upper':1.5,
             'TA_lower':0.01,
             'TA_upper':1
             },
         'theta':{
             'SA_lower':0.8,
             'SA_upper':1.2,
             'TA_lower':0.01,
             'TA_upper':1
             },
         'rho':{
             'SA_lower':0.8,
             'SA_upper':1.2,
             'TA_lower':0.01,
             'TA_upper':0.5
             },
         'vomma':{
             'SA_lower':0.5,
             'SA_upper':1.5,
             'TA_lower':0.01,
             'TA_upper':1
             },
         'vanna':{
             'SA_lower':0.5,
             'SA_upper':1.5,
             'TA_lower':0.01,
             'TA_upper':1
             },
         'zomma':{
             'SA_lower':0.8,
             'SA_upper':1.2,
             'TA_lower':0.01,
             'TA_upper':0.5
             },
         'speed':{
             'SA_lower':0.8,
             'SA_upper':1.2,
             'TA_lower':0.01,
             'TA_upper':0.5
             },
         'color':{
             'SA_lower':0.8,
             'SA_upper':1.2,
             'TA_lower':0.01,
             'TA_upper':0.5
             },
         'ultima':{
             'SA_lower':0.5,
             'SA_upper':1.5,
             'TA_lower':0.01,
             'TA_upper':1
             },
         'vega bleed':{
             'SA_lower':0.5,
             'SA_upper':1.5,
             'TA_lower':0.01,
             'TA_upper':1
             },
         'charm':{
             'SA_lower':0.8,
             'SA_upper':1.2,
             'TA_lower':0.01,
             'TA_upper':0.25
             }
         },

     # Greek names as function input and individual function
     # names
     'greek_dict':{
         'price':'price',
         'delta':'delta',
         'gamma':'gamma',
         'vega':'vega',
         'theta':'theta',
         'rho':'rho',
         'vomma':'vomma',
         'vanna':'vanna',
         'zomma':'zomma',
         'speed':'speed',
         'color':'color',
         'ultima':'ultima',
         'vega bleed':'vega_bleed',
         'charm':'charm'
         },

     # Parameters to overwrite mpl_style defaults
     'mpl_params':{
         'legend.fontsize': 'x-large',
         'legend.fancybox':False,
         'figure.dpi':72,
         'axes.labelsize': 'medium',
         'axes.titlesize':'large',
         'axes.spines.bottom':True,
         'axes.spines.left':True,
         'axes.spines.right':True,
         'axes.spines.top':True,
         'axes.edgecolor':'black',
         'axes.titlepad':20,
         'axes.autolimit_mode':'data',#'round_numbers',
         'axes.xmargin':0.05,
         'axes.ymargin':0.05,
         'axes.linewidth':2,
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium',
         'xtick.major.pad':10,
         'ytick.major.pad':10,
         'lines.linewidth':3.0,
         'lines.color':'black',
         'grid.color':'black',
         'grid.linestyle':':',
         'font.size':14
         },

     'mpl_3d_params':{
         'axes.facecolor':'w',
         'axes.labelcolor':'k',
         'axes.edgecolor':'w',
         'axes.titlepad':5,
         'lines.linewidth':0.5,
         'xtick.labelbottom':True,
         'ytick.labelleft':True
         }
     }
