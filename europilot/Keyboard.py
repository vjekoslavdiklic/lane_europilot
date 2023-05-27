import os
import sys
import time
import random
from binascii import hexlify
import uinput


BUTTON_NAME_F430_Controller = """
0100=Button:X
0101=Button:O
0102=Button:Triangle
0103=noUSE
0104=Button:L1
0105=Button:R1
0106=wheel-button-left-1
0107=wheel-button-right-1
0108=Button:Select
0109=Button:Start
010a=Button:PS
010b=Button:LeftThumb
010c=noUSE
010d=Button:Up
010e=Button:Down
010f=Button:Left
0110=Button:Right
0200=wheel-axis
0201=brake
0202=gas
0203=Axis:3
0204=Axis:4
0205=AXIS:5
"""
keydict={1: 'ESC',
         2: '1',
         3: '2',
         4: '3',
         5: '4',
         6: '5',
         7: '6',
         8: '7',
         9: '8',
         10: '9',
         11: '0',
         12: 'apostrof',
         13: '+',
         14: 'backspace',
         15: '\t',
         16: 'q',
         17: 'w',
         18: 'e',
         19: 'r',
         20: 't',
         21: 'y',
         22: 'u',
         23: 'i',
         24: 'o',
         25: 'p',
         26: 'š',
         27: 'đ',
         28: '\n',
         30: 'a',
         31: 's',
         32: 'd',
         33: 'f',
         34: 'g',
         35: 'h',
         36: 'j',
         37: 'k',
         38: 'l',
         39: 'č',
         40: 'ć',
         41: 'KEY_GRAVE',
         43: '\\',
         44: 'z',
         45: 'x',
         46: 'c',
         47: 'v',
         48: 'b',
         49: 'n',
         50: 'm',
         51: ',',
         52: '.',
         53: '/',
         55: '*',
         57: ' ',
         59: 'F1',
         96: 'KEY_KPENTER',
         170:'KEY_ISO'
         }


class Bytewurst(object):

    def __init__(self, bs):
        self.raw = bs
        self.ints = map(ord, bs)

    def __repr__(self):
        return ' '.join(map(hexlify, self.raw))

    @property
    def int(self):
        """
        For "01 0A" ints would be [1, 10], so::
            >>> bs = '\x01\x0A'
            >>> bw = Bytewurst(bs)
            >>> bw.int == (1 * 1) + (10 * 256)
        """

        def powergenerator(start=0):
            """Generate powers of 256"""
            i = start
            while True:
                yield 256 ** i
                i += 1

        #return sum(a * b for a, b in zip(self.ints, powergenerator()))
        #return sum(a * b for a, b in zip(int.from_bytes(self.raw[0:2],'big'), powergenerator()))
        return int.from_bytes(self.raw[0:2],'little')

    @property
    def hexLE(self):
        return hexlify(self.raw)

    @property
    def bits(self):
        return ' '.join([format(x, '08b') for x in self.ints])


class Button(Bytewurst):
    def __init__(self, bs):
        super(Button, self).__init__(bs)
        button_namedict = dict(line.split('=') for line in
                               BUTTON_NAME_F430_Controller.strip().split('\n'))
        self.name = button_namedict.get(self.hexLE[2:6].decode(), 'UNKNOWN:%s' % self.raw)


class Value(Bytewurst):
    def __repr__(self):
        if self.int == 0:
            return ' off'
        elif self.int == 1:
            return ' on'
        else:
            return ' %d' % self.int

    def int_normalized(self, name):
        """ Normalizes value to an adequate range

        For wheel values, the output range is normalized to [-32767, 32767].
        For pedal values, the output range is normalized to [0, 65535].
        For Arrow Pad values, the output range is normalized to [-1, 1].
        """
        v = super(Value, self).int
        if name == 'wheel-axis' or name[0:4]=='Axis': #added other Axis's
            if v >= 32769:
                v = v - 65536
        elif name == 'clutch' or name == 'brake' or name == 'gas' :
            if v >= 32769:
                v = -v + 98304
            else:
                v = -v + 32767
            #v=65535-v #to be compatible with logitech g27
        elif name[0:6] == 'Button'or name[0:12]=='wheel-button': #buttons handler
            if v > 1:
                v = 1

        return v


class Message(object):
    def __init__(self, bs):
        self.bs = bs
        self.raw_seq = bs[0:4]
        self.raw_value = bs[4:8]
        self.raw_id = bs[7]
        self.ints = map(ord, bs)
        self.sequence = Bytewurst(bs[0:3])
        self.value = Value(bs[4:8])
        self.button = Button(bs[5:8])

    def __repr__(self):
        values = (self.button.name,
                  self.value.int_normalized(self.button.name))
        return ' '.join(map(str, values))


class LinuxVirtualKeyboard(object):
    def __init__(self, name='Virtual keyboard', bustype=0x003,
                 vendor=0x0bf8, product=0x101e, version=0x0111, events=None):
        if events is None:
            tlist = list()
            for i in range(0, 200):
                tlist.append(tuple((1, i)))
            tlist = tuple(tlist)
            events = tlist

        self.device = uinput.Device(events,
                                    name=name,
                                    bustype=bustype,
                                    vendor=vendor,
                                    product=product,
                                    version=version)

    def emit(self, key):
        # emit as one atomic event, by using syn=True at the last emit call
        # for more information check the source code at
        # https://github.com/tuomasjjrasanen/python-uinput/blob/master/src/__init__.py#L191

        if type(key) == type(list()):
            for each in key:
                self.device.emit(each, 1, syn=True)
            for each in key:
                self.device.emit(each, 0, syn=True)
        else:
            self.device.emit(key, 1, syn=True)
            self.device.emit(key, 0, syn=True)

    def hold(self,key):
        if type(key) == type(list()):
            for each in key:
                self.device.emit(each, 1, syn=True)
        else:
            self.device.emit(key, 1, syn=True)

    def relese(self,key):
        if type(key) == type(list()):
            for each in key:
                self.device.emit(each, 0, syn=True)
        else:
            self.device.emit(key, 0, syn=True)


    def typestring(self,InputString):
        for each in InputString:
            forceshift=0
            if each=='z':
                each='y'
            elif each=='y':
                each='z'
            elif each == 'Y':
                each = 'Z'
            elif each == 'Z':
                each = 'Y'
            elif each =='/':
                each='7'
                forceshift=1
            else:
                each=each


            if each.isupper() or forceshift:
                each=each.lower()
                self.device.emit(uinput.KEY_LEFTSHIFT, 1, syn=True)
                self.device.emit(uinput._chars_to_events(each)[0], 1, syn=True)
                self.device.emit(uinput._chars_to_events(each)[0], 0, syn=True)
                self.device.emit(uinput.KEY_LEFTSHIFT, 0, syn=True)
            else:
                self.device.emit(uinput._chars_to_events(each)[0], 1, syn=True)
                self.device.emit(uinput._chars_to_events(each)[0], 0, syn=True)

    def __del__(self):
        self.device.destroy()


# Handle platform dependent joystick impl
if sys.platform.startswith('linux'):
    import uinput
    VirtualJoystick = LinuxVirtualKeyboard
elif sys.platform == 'darwin':
    pass


if __name__ == '__main__':
    def _dump_messages(input_):
        with open(input_, 'rb') as device:
            while True:
                bs = device.read(16)
                print("raw ",bs)
                message = Message(bs)
                print(message)

    def _dump_dummy_messages():
        # Need to simulate straight wheel axis to test realtime fps adjustment.
        straight = False
        straight_start_time = None
        straight_duration = 5
        while True:
            if not straight and random.randint(0, 99) == 0:
                # Straight wheel axis for 5 secs.
                straight = True
                straight_start_time = time.time()

            if straight and \
                    time.time() - straight_start_time > straight_duration:
                # End of straight wheel axis
                straight = False
                straight_start_time = None

            # Axis value 0 will be regarded as straight wheel regardless of
            # `train.FpsAdjuster._max_straight_wheel_axis`.
            wheel_axis = random.randint(-32767, 32767) if not straight else 0

            print('wheel-axis ' + str(wheel_axis))

    # TODO: Get device path as argument
    device = '/dev/input/event25'
    if os.path.exists(device):
        _dump_messages(device)
    else:
        # When joystick doesn't exist. Let's dump dummy messages.
        # TODO: Warn this to stdout so that we can be aware of mock g27.
        _dump_dummy_messages()
