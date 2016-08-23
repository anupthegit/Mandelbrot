from mandelbrot import Mandelbrot
from matplotlib import pyplot as plt

mandelbrot = Mandelbrot((-2.0, 1.0, -1.0, 1.0), (1200, 800), 100)
img = mandelbrot.getGrayscaleMandelbrot()
plt.imshow(img)
plt.show()
