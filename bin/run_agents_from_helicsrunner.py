import subprocess as sb
import sys

fname = sys.argv[1]

cmd = "helics run --path {} --broker-loglevel=7".format(fname)
p1 = sb.Popen(cmd.split(" "))
p1.wait()
