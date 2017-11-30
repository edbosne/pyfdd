#!/usr/bin/python

# example
# /home/eric/PycharmProjects/MedipixMatrix/Convert2dl.py -i '/media/eric/Data/University/CERN-projects/Alphapix-ITN/Data/2012-03-17_6HSiC/Analysis/Better_Depth_analysis_2013-11-22/2D-Pattern_200nm.2db' -o /home/eric/Desktop/test.2dl

import sys, getopt
from .MedipixMatrix import *

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:")
    except getopt.GetoptError:
        print('Convert2dl.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg
    print('Input file is "', inputfile)
    print('Output file is "', outputfile)

    print('creating matrix')
    # Mpx = MedipixMatrix(
    #     nx=512,
    #     ny=512,
    #     nChipsX=2,
    #     nChipsY=2,
    #     filename=inputfile,
    #     smooth_fwhm=0,
    #     percentiles=(0.1,1),
    #     range=(0, 516),
    #     compress_factor=2,
    #     rm_central_pixels=1,
    #     rm_edge_pixels=0
    # )
    # f = plt.figure('fig1')
    # Mpx.draw(f)

    Mpx = MedipixMatrix(file_path=inputfile, nChipsX=2, nChipsY=2, real_size=3)

    # Manipulation methods
    # -Orient
    Mpx.manip_orient('rr,mh')  # TimepixQuad orientation

    # Zero central pix
    Mpx.zero_central_pix(1)

    # Add extra pixels to account for bigger central pixels
    Mpx.manip_correct_central_pix()

    # -Sum pixels, zero central pixels and remove edge pixels all in one
    Mpx.manip_compress(factor=2, rm_central_pix=1, rm_edge_pix=0)

    print('save as 2dl')
    savename = "/home/eric/Desktop/test.2dl"
    Mpx.io_save_origin(outputfile)

if __name__ == "__main__":
    argv = sys.argv[1:]
    infile = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-1101/pattern_d3_Npix0-20.txt'
    path_in, filename_in = os.path.split(infile)
    name, filetype_in = os.path.splitext(filename_in)
    outfile = os.path.join(path_in,name+"rebin2x2.2db")
    #outfile = '/home/eric/Desktop/test.2dl'
    argv = ['-i',
            infile,
            '-o',
            outfile]
    main(argv)

