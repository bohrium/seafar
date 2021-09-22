import imageio
import numpy as np 

HALF = 64
img = np.zeros(shape=(2*HALF,2*HALF,3))
img[:][:][:] = 0.05

xs,ys = np.meshgrid(np.arange(2*HALF)-HALF, HALF-np.arange(2*HALF))

def add_gauss(img, mu_x, mu_y, size, k_x, k_y, phase=None, intensity=0.5):
    if phase is None: phase = np.random.random() * 2 * np.pi
    gauss = np.exp( - ((xs-mu_x)**2+(ys-mu_y)**2)/(2*size**2) ) 
    wave = 0.5 + 0.5 * np.cos( phase + k_x*xs + k_y*ys ) 
    gauss = np.expand_dims(gauss, axis=2)
    wave = np.expand_dims(wave, axis=2)
    img += (intensity * gauss) * (wave - img)

def body(img, mu_x, mu_y, size, k_x, k_y, texture=2, harmonics=1):
    add_gauss(img, mu_x, mu_y, size, k_x, k_y, intensity=0.9)

    #for i in range(2, 2+harmonics):
    #    add_gauss(img, mu_x, mu_y, size, i*k_x, i*k_y, intensity=0.1)

    for _ in range(texture):
        k_x = np.random.normal()
        k_y = np.random.normal()
        for i in range(2, 2+harmonics):
            add_gauss(img, mu_x, mu_y, size, i*k_x, i*k_y, intensity=0.1)

# render textured gabors:
body(img, 10, 20, 5.0, 0.5, -1.0)
body(img, 15, 15, 5.0, 1.0, -1.0)
body(img, 20, 10, 5.0, 1.5, -1.0)

# render untextured gabors:
add_gauss(img, -10, -20, 5.0, 0.5, -1.0, intensity=0.9)
add_gauss(img, -15, -15, 5.0, 1.0, -1.0, intensity=0.9)
add_gauss(img, -20, -10, 5.0, 1.5, -1.0, intensity=0.9)

img = np.maximum(0.0, np.minimum(1.0, img))

imageio.imsave('sample_textures.png', img)
