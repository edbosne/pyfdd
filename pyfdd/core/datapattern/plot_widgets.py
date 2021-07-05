__author__ = 'E. David-Bosne'


from matplotlib.widgets import AxesWidget
import matplotlib.axes
import math


class AngleMeasurement(AxesWidget):
    """
    A special axes widget that creates a line py pressing and dragging

    *AxesWidget* : matplotlib.widgets.AxesWidget

    For the cursor to remain responsive you much keep a reference to
    it.
    """

    def __init__(self, ax, callonangle=None, callonmove=None, **lineprops):
        """
        Add an AngleMeasurement widget to *ax*.

        *ax* : matplotlib.axes.Axes
        *lineprops* is a dictionary of line properties.
        """
        AxesWidget.__init__(self, ax)

        # Connect signals
        self.connect_event('button_press_event', self.on_press)
        self.connect_event('button_release_event', self.on_release)
        self.connect_event('motion_notify_event', self.on_motion)

        # Call functions
        self.callonangle = callonangle
        self.callonmove = callonmove

        # Variables
        self.visible = False
        self.angle = 0
        self.point1_xy = [0, 0]
        self.point2_xy = [0, 0]
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

    def _reset_data(self):
        self.visible = False
        self.angle = 0
        self.point1_xy = [0, 0]
        self.point2_xy = [0, 0]
        self.xLine = [0, 0]
        self.yLine = [0, 0]
        self.dx = 0
        self.dy = 0
        self.line[0].set_visible(self.visible)

    def on_press(self, event):
        """on button press the central position is saved and line set to visible"""
        if self.ignore(event):
            return
        self._reset_data()
        self.visible = True
        self.line[0].set_visible(self.visible)
        self.point1_xy = [event.xdata, event.ydata]

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
            self._reset_data()

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True

        self.point2_xy = [event.xdata, event.ydata]
        x1, y1 = self.point1_xy
        x2, y2 = self.point2_xy
        dx = x2 - x1
        dy = y2 - y1
        self.dx = dx
        self.dy = dy

        self.xLine = [x1 - dx, x1 + dx]
        self.yLine = [y1 - dy, y1 + dy]

        self.line[0].set_xdata(self.xLine)
        self.line[0].set_ydata(self.yLine)

        self.angle = math.degrees(math.atan2(self.dy, self.dx))

        if self.callonmove is not None:
            self.callonmove(self.point1_xy, self.angle)

        self._update()

    def on_release(self, event):
        """on release print the angle and reset the data"""
        if self.ignore(event):
            return
        if not self.visible:
            return
        if self.dx == 0 and self.dy == 0:
            return

        self.angle = math.degrees(math.atan2(self.dy, self.dx))

        if self.callonangle is not None:
            self.callonangle(self.point1_xy, self.angle)

        self._reset_data()
        self._update()

    def get_values(self):

        return self.point1_xy, self.angle
