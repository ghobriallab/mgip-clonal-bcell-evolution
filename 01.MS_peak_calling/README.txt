The npy file is a NumPy array is an extracted portion of the raw MS data (.h5 files).

Structure of npy file:
1st dimension (2): Splits the data into x-values (m/z in Da) and y-values (relative abundance).
2nd dimension (5): Represents the 5 "GAMKL" channels.
3rd dimension (e.g. 1,577): Contains the data points of the MS tracing.


