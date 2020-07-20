'''
Functions which initialize the 'configuration' -> values which don't change during runtime
and the 'state' -> values which represent the state of the application
'''
import json
from dataclasses import dataclass, field
import logging
from sys import stdout
import datetime
from typing import Dict, List
import pyautogui
import torch
from model import GestureNet, ShrecNet

@dataclass
class Config:
    ''' The configuration of the application. '''
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    else:
        map_location = None

    # Set up logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/debug{}.log".format(
                datetime.datetime.now().strftime("%m.%d:%H.%M.%S"))),
            logging.StreamHandler(stdout)
        ]
    )

    # If lite is true, then the neural networks are not loaded into the config
    # This is useful in scripts which do not use the network, or may modify the network.
    lite: bool

    # Path to action configuration file
    config_path: str = 'data/action_config.json'

    # Seed value for reproducibility
    seed_val: int = 42

    # Refer make_vector() in train_model.py to verify input dimensions
    static_input_dim: int = 49
    static_output_classes: int = 7

    # Refer format_mediapipe() in dynamic_train_model.py to verify input dimensions
    dynamic_input_dim: int = 36
    dynamic_output_classes: int = 15
    shrec_output_classes: int = 14

    # Minimum number of epochs
    min_epochs: int = 15

    static_batch_size: int = 64
    dynamic_batch_size: int = 1

    # value for pytorch-lighting trainer attribute accumulate_grad_batches
    grad_accum: int = 4

    static_gesture_mapping: dict = field(default_factory=dict)
    dynamic_gesture_mapping: dict = field(default_factory=dict)

    # Mapping of gestures to actions
    gesture_action_mapping: dict = field(default_factory=dict)

    static_path: str = 'models/gesture_net.pth'
    shrec_path: str = 'models/shrec_net.pth'
    dynamic_path: str = 'models/user_net.pth'

    gesture_net: GestureNet = field(init=False)
    shrec_net: ShrecNet = field(init=False)

    def __post_init__(self):
        # Fetching gesture mappings
        with open('data/static_gesture_mapping.json', 'r') as jsonfile:
            self.static_gesture_mapping = json.load(jsonfile)
        with open('data/dynamic_gesture_mapping.json', 'r') as jsonfile:
            self.dynamic_gesture_mapping = json.load(jsonfile)

        with open(self.config_path, 'r') as jsonfile:
            self.gesture_action_mapping = json.load(jsonfile)

        # allow mouse to move to edge of screen, and set interval between calls to 0.01
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01

        # Setting up networks
        if not self.lite:
            logging.info('Loading GestureNet...')
            self.gesture_net = GestureNet(self.static_input_dim, self.static_output_classes)
            self.gesture_net.load_state_dict(torch.load(self.static_path,
                                                        map_location=self.map_location))
            self.gesture_net.eval()

            logging.info('Loading ShrecNet..')

            self.shrec_net = ShrecNet(self.dynamic_input_dim, self.shrec_output_classes)
            self.shrec_net.load_state_dict(torch.load(self.shrec_path,
                                                      map_location=self.map_location))
            self.shrec_net.eval()


@dataclass
class State:
    ''' The state of the application. '''

    # flag that indicates whether mouse is tracked.
    mouse_track: bool

    # the mode in which to start the application
    start_mode: str

    # array of flags for mouse control
    mouse_flags: Dict = field(default_factory=dict)

    # maintain a buffer of most recent movements to smoothen mouse movement
    pointer_buffer: List = field(default_factory=list)
    prev_pointer: List = field(default_factory=list)

    # maintain a buffer of most recently detected configs
    static_config_buffer: List = field(default_factory=list)

    modes: List = field(default_factory=list)

    #number of iterations
    iter: int = 0

    # maintain a buffer of keypoints for dynamic gestures
    keypoint_buffer: List = field(default_factory=list)

    # Flag to denote whether the Ctrl key is pressed
    ctrl_flag: bool = False
    # ctrl_flag of the previous timestep. Used to detect change
    prev_flag: bool = False

    def __post_init__(self):

        self.mouse_flags = {'mousedown': False, 'scroll': False}

        self.pointer_buffer = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
        self.prev_pointer = [0, 0]

        self.static_config_buffer = ['', '', '', '', '']

        # Modes:
        # Each mode is a different method of interaction with the system.
        # Any given functionality might require the use of multiple modes
        # The first index represents the current mode. When mode is switched, the list is cycled.
        # 1. Mouse -> In mouse mode, we interact with the system in all the ways that a mouse can.
        #             E.g. left click, right click, scroll
        # 2. Gesture -> Intuitive gesture are performed to do complicated actions, such as switch
        #             worskpace, dim screen brightness etc.
        if self.start_mode == 'mouse':
            self.modes = ['mouse', 'gesture']
        else:
            self.modes = ['gesture', 'mouse']
