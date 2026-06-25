from PIL import Image

filename = 'Photos/Boston.webp'
img1 = Image.open(filename)
print(img1)
img1.show()

img2 = Image.new('RGB', (400,200), (200,200,0))
#img2.show()

img3 = img1.rotate(angle=60,expand=True,fillcolor='green',resample=Image.BICUBIC , center = (100,100) )
#img3.show()

img4 = img1.resize((int(img1.width/2),int(img1.height/2)),resample=Image.BICUBIC, box= (20,20,100,100))
#img4.show()

img5 = img1.crop(box=(20,20,200,200))
img5.show()

