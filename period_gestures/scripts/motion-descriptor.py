#! /usr/bin/env python

# computes a motion descriptor for an optical flow file in the style of Alex Couture-Beil, adapted from Fathi and Mori: Recognizing Action at a Distance

import sys, os, cPickle as pickle, numpy as np

FLOW_DIR = None
OUT_DIR = None

EPSILON = 0.5 # the value used by Alex in his thesis
BOX_FILTER_RADIUS = 1 # the radius of box filtering

# size of cropped rectangle containing user
H = 50
W = 50

#---------------------------------------------------------------------------------

def compute_motion_descriptor(flowdir):
    print "computing motion descriptor for clip: %s" % OUT_DIR
    descriptors = np.zeros((30,H,W,5))
    aggregate_descriptor = np.zeros((H,W,5))
    for i in range(1,31):
        flowfile = "%s%s%02d.pickle" % (flowdir, os.sep, i)

        with open(flowfile) as fp:
            flow = pickle.load(fp)
            
            h,w = flow.shape[:2]
            fx,fy = flow[:,:,0], flow[:,:,1]

            # motion_descriptor[x,y] =  (fxp, fxn, fyp, fyn, fnorm)
            motion_descriptor = np.zeros((h,w,5))

            for x in range(0,w):
                for y in range(0,h):
                    # normalization and squashing of tiny vectors
                    fx[x,y] = fx[x,y] / (np.linalg.norm((fx[x,y], fy[x,y])) + EPSILON)
                    fy[x,y] = fy[x,y] / (np.linalg.norm((fx[x,y], fy[x,y])) + EPSILON)

                    if fx[x,y] > 0:
                        motion_descriptor[x,y,0] = fx[x,y]
                    else:
                        motion_descriptor[x,y,1] = abs(fx[x,y])

                    if fy[x,y] > 0:
                        motion_descriptor[x,y,2] = fy[x,y]
                    else:
                        motion_descriptor[x,y,3] = abs(fy[x,y])

                    motion_descriptor[x,y,4] = np.linalg.norm((fx[x,y], fy[x,y]))

            # box filter each of the 5 channels
            box_filtered_descriptor = np.zeros((h,w,5))
            for x in range(0,w):
                for y in range(0,h):
                    for comp in range(0,5):
                        avg = 0.0
                        count = 0
                        for radx in range(max(0,x-BOX_FILTER_RADIUS), min(w-1,x+BOX_FILTER_RADIUS) + 1):
                            for rady in range(max(0,y-BOX_FILTER_RADIUS), min(h-1,y+BOX_FILTER_RADIUS) + 1):
                                avg += motion_descriptor[x,y,comp]
                                count += 1
                        box_filtered_descriptor[x,y,comp] = avg / count

            descriptors[i-1,:,:,:] = box_filtered_descriptor
            aggregate_descriptor = np.add(box_filtered_descriptor, aggregate_descriptor)


    # export both the aggregate sum of the descriptor components over the 30 frames, a 50x50x5 vector,
    # and the collection of vectors over all 30 frames, a 30x50x50x5 vector
        
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    outfile_agg = "%s%saggregate.pickle" % (OUT_DIR, os.sep)
    with open(outfile_agg, "wb") as outfp_agg:
        pickle.dump(aggregate_descriptor, outfp_agg)

    outfile_flat = "%s%sflattened.pickle" % (OUT_DIR, os.sep)
    with open(outfile_flat, "wb") as outfp_flat:
            pickle.dump(descriptors, outfp_flat)



#-----------------------------------------------------------------------------------




if __name__ == "__main__":
    for i,arg in enumerate(sys.argv):
        if arg == "--flow-dir":
            FLOW_DIR = os.path.normpath(sys.argv[i+1])
        elif arg == "--out-dir":
            OUT_DIR = os.path.normpath(sys.argv[i+1])

    if FLOW_DIR == None or OUT_DIR == None:
        print "usage: %s --flow-dir <dir> --out-dir <dir>" % sys.argv[0]
        sys.exit(1)

    compute_motion_descriptor(FLOW_DIR)
