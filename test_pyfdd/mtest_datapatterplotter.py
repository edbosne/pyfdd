import matplotlib as mpl
import matplotlib.pyplot as plt
from pyfdd import DataPattern, DataPatternPlotter


class TestDataPatternPlotter:

    def __init__(self):
        self.pad_path = './data_files/pad_dp_2M.json'
        self.tpx_path = './data_files/tpx_quad_dp_2M.json'

    def make_mock_tpx(self):
        dp = DataPattern(file_path=self.tpx_path)
        return dp

    def make_mock_pad(self):
        dp = DataPattern(file_path=self.pad_path)
        return dp

    def test_autoscale(self, plot_type):
        dp = self.make_mock_pad()
        dp_plotter = DataPatternPlotter(dp)

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.set_aspect('equal')

        dp_plotter.draw(ax, plot_type=plot_type)
        plt.tight_layout()

        f1 = plt.figure()
        #ax = f.add_subplot(111)
        gs = f1.add_gridspec(1, 1, left=0.15, right=0.85, top=0.9, bottom=0.2)
        ax = f1.add_subplot(gs[0])
        ax.set_aspect('equal')

        dp_plotter.draw(ax, plot_type=plot_type)

        f2 = plt.figure()
        gs = f2.add_gridspec(1, 1, left=0.15, right=0.85, top=0.9, bottom=0.2)
        ax = f2.add_subplot(gs[0])
        ax.set_aspect('equal')

        dp_plotter.draw(ax, plot_type=plot_type)

        f3 = plt.figure()

        gs = f3.add_gridspec(1, 1, left=0.15, right=0.85, top=0.9, bottom=0.2)
        ax = f3.add_subplot(gs[0])
        ax.set_aspect('equal')

        dp_plotter.draw(ax, plot_type=plot_type)
        dp_plotter.clear_draw()
        dp_plotter.draw(ax, plot_type=plot_type)
        dp_plotter.clear_draw()
        dp_plotter.draw(ax, plot_type=plot_type)
        dp_plotter.clear_draw()
        dp_plotter.draw(ax, plot_type=plot_type)

        plt.show()

    def test_autoscale_pixels(self):
        self.test_autoscale(plot_type='pixels')

    def test_autoscale_contour(self):
        self.test_autoscale(plot_type='contour')


if __name__ == '__main__':
    dpp_tester = TestDataPatternPlotter()
    #dpp_tester.test_autoscale_pixels()
    dpp_tester.test_autoscale_contour()
