__author__ = 'E. David-Bosne'


from matplotlib.widgets import *
import matplotlib.axes
import math

class AngleMeasure(AxesWidget):
    """
    A special cursor that creates a line py pressing and dragging

    *AxesWidget* : matplotlib.widgets.AxesWidget

    For the cursor to remain responsive you much keep a reference to
    it.
    """

    def __init__(self, ax, callonangle=print, **lineprops):
        """
        Add a cursor to *ax*.

        *ax* : matplotlib.axes.Axes
        *lineprops* is a dictionary of line properties.
        """
        AxesWidget.__init__(self, ax)

        self.connect_event('button_press_event', self.on_press)
        self.connect_event('button_release_event', self.on_release)
        self.connect_event('motion_notify_event', self.on_motion)

        self.callonangle = callonangle

        self.visible = False
        self.angle = 0
        self.centerXY = [0, 0]
        self.xLine = [0, 0]
        self.yLine = [0, 0]
        self.dx = 0
        self.dy = 0

        if lineprops:
            self.line = matplotlib.axes.Axes.plot(ax, self.xLine, self.yLine, '', **lineprops)
        else:
            self.line = matplotlib.axes.Axes.plot(ax, self.xLine, self.yLine, 'r-')
            # line is a list of matplotlib.lines.Line2D
            self.line[0].set_linewidth(3)
        self.line[0].set_visible(self.visible)

        self.needclear = False

    def _update(self):
        self.canvas.draw_idle()

    def _resetData(self):
        self.visible = False
        self.angle = 0
        self.centerXY = [0, 0]
        self.xLine = [0, 0]
        self.yLine = [0, 0]
        self.dx = 0
        self.dy = 0
        self.line[0].set_visible(self.visible)

    def on_press(self, event):
        'on button press the central position is saved and line set to visible'
        if self.ignore(event):
            return
        self._resetData()
        self.visible = True
        self.line[0].set_visible(self.visible)
        self.centerXY = [event.xdata, event.ydata]

    def on_motion(self, event):
        """on mouse motion draw the angle line if visible"""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.button != 1:
            return
        if not self.visible:
            return
        if event.inaxes != self.ax:
            self._resetData()

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True

        x0, y0 = self.centerXY
        dx = event.xdata - x0
        dy = event.ydata - y0
        self.dx = dx
        self.dy = dy

        if dx == 0:
            self.xLine = [x0, x0]
        else:
            self.xLine = [x0 - dx, x0 + dx]

        self.yLine = [y0 - dy, y0 + dy]

        self.line[0].set_xdata(self.xLine)
        self.line[0].set_ydata(self.yLine)

        self._update()

    def on_release(self, event):
        'on release print the angle and reset the data'
        if self.ignore(event):
            return
        if not self.visible:
            return
        if self.dx == 0 and self.dy == 0:
            return

        angle = math.atan2(self.dy, self.dx)
        #print('the center is at position ({:0.2f}, {:0.2f})'.format(self.centerXY[0],self.centerXY[1]))
        #print('the angle is', math.floor(math.degrees(angle)*10)/10.0)

        self.callonangle(self.centerXY, math.degrees(angle))

        self._resetData()

        self._update()
