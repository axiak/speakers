import atexit

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None

GPIO_PIN = 27


def unmute():
    if GPIO is None:
        return

    write_value(GPIO_PIN, 0)

    def set_low(*args, **kwargs):
        mute()
    atexit.register(set_low)


def mute():
    write_value(GPIO_PIN, 1)

def write_value(pin, number):
    try:
        with open('/sys/class/gpio/export', 'w+') as f:
            f.write('{0}\n'.format(pin))
    except:
        pass

    try:
        with open('/sys/class/gpio/gpio{0}/direction'.format(pin), 'w+') as f:
            f.write('out\n')
    except:
        pass

    with open('/sys/class/gpio/gpio{0}/value'.format(pin), 'w+') as f:
        f.write('{0}\n'.format(number))
