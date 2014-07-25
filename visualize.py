#!/usr/bin/env python
"""
Interactive face generator

Dependends on:
    * numpy 1.7
    * matplotlib
"""
from morpher import Morpher
from matplotlib import pyplot


if __name__ == "__main__":
    # Initialize face generator
    morpher = Morpher()

    # Make sure I/O doesn't hang when displaying the image
    pyplot.ion()

    # Build visualization window
    fig = pyplot.figure(figsize=(1, 1), dpi=300)
    im = pyplot.imshow(X=morpher.generate_face(),
                       interpolation='nearest',
                       cmap='gray')
    pyplot.axis('off')

    def onmove(event):
        width, height = fig.canvas.get_width_height()
        x = 2 * event.x / float(width) - 1
        y = 2 * event.y / float(height) - 1
        morpher.set_coordinates(x, y)
        im.set_array(morpher.generate_face())
        pyplot.draw()
    def onclick(event):
        morpher.toggle_freeze()
    fig.canvas.mpl_connect('motion_notify_event', onmove)
    fig.canvas.mpl_connect('button_press_event', onclick)

    pyplot.show()

    # Program loop
    quit = False
    help_message = (
        "\n" +
        "Move your mouse over the image to make it morph, click it to\n" +
        "freeze / unfreeze the image.\n" +
        "\n" +
        "COMMANDS:\n" +
        "    h          Display this help message\n" +
        "    q          Quit the program\n" +
        "    r          Randomize face\n" +
        "    d D1 D2    Select dimensions D1 and D2\n"
    )
    print help_message
    while not quit:
        command_args = raw_input('Type a command: ').strip().split(" ")
        if command_args[0] == 'h':
            print help_message
        elif command_args[0] == 'q':
            quit = True
        elif command_args[0] == 'r':
            morpher.shuffle()
            im.set_array(morpher.generate_face())
            pyplot.draw()
        elif command_args[0] == 'd':
            if len(command_args) < 3:
                print"  ERROR: need two dimensions to select"
            else:
                try:
                    arg0 = int(command_args[1])
                    arg1 = int(command_args[2])
                    morpher.select_dimensions(arg0, arg1)
                except ValueError:
                    print "  ERROR: Dimensions must be integers in [0, 28]"
                except KeyError:
                    print "  ERROR: Dimensions must be integers in [0, 28]"
