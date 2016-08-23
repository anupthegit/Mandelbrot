import numpy as np
import pyopencl as cl


class Mandelbrot:
    """Models Mandelbrot set"""
    __code = '''
                #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
                __kernel void pointHorizons(__global float2 *input,
                    __global ushort *output, ushort const maxiter){
                    int gid = get_global_id(0);
                    float nreal, real = 0;
                    float imag = 0;
                    output[gid] = 0;
                    for(int i=0;i<maxiter;i++){
                        nreal = real*real - imag*imag + input[gid].x;
                        imag = 2*real*imag + input[gid].y;
                        real = nreal;
                        if(real*real + imag*imag > 4.0f){
                            output[gid] = i;
                            return;
                        }
                    }
                }
            '''

    def __init__(self, window, resolution, maxiter):
        self.__window = window
        self.__resolution = resolution
        self.__maxiter = maxiter
        self.__ctx = cl.create_some_context()

    def getGrayscaleMandelbrot(self):
        x1, x2, y1, y2 = self.__window
        w, h = self.__resolution
        xarr = np.arange(x1, x2, (float(x2) - float(x1)) / float(w))
        yarr = np.arange(y1, y2, (float(y2) - float(y1)) / float(h)) * 1j
        points = np.ravel(xarr + yarr[:, np.newaxis]).astype(np.complex64)
        pointHorizons = self.getPointHorizons(points, self.__maxiter)
        image = (pointHorizons.reshape((h, w)) / float(pointHorizons.max()) * 255.0).astype(np.uint8)
        return image

    def getPointHorizons(self, points, maxiter):
        q = cl.CommandQueue(self.__ctx)
        output = np.empty(points.shape, dtype=np.uint16)
        mf = cl.mem_flags
        points_cl = cl.Buffer(self.__ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=points)
        output_cl = cl.Buffer(self.__ctx, mf.WRITE_ONLY, output.nbytes)
        prg = self.kernelProgram()
        prg.pointHorizons(q, output.shape, None, points_cl, output_cl, np.uint16(maxiter))
        cl.enqueue_copy(q, output, output_cl).wait()
        return output

    def kernelProgram(self):
        if hasattr(self, '__prg'):
            return self.__prg
        prg = cl.Program(self.__ctx, self.__code).build()
        self.__prg = prg
        return prg
