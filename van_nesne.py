import cv2
import numpy as np
img=cv2.imread("resimler/van.png", cv2.IMREAD_COLOR)
frameWidth = 230
frameHeight = 250
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",170,255,empty)
cv2.createTrackbar("Threshold2","Parameters",240,255,empty)
cv2.createTrackbar("Area","Parameters",5000,30000,empty)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

#Fonksiyon bize konturları bulamızı sağlıyor
def getContours(img,imgContour):
    #
    """FindContoures fonksiyonu ile gelen img  contourler aranmaktadır.Burada RETR_EXTERNAL resmin dışındakileri ele almaktadır Dış yuzeyini ele alarak dıştaki contouraları bulmaktadır.
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        hesaplanan_area = cv2.contourArea(cnt)     #Bulduğumuz contours llistesindki contour lardan sırasıyla alanlarını hesaplıyoruz
        area_eşik_değeri = cv2.getTrackbarPos("Area", "Parameters")  #Başta oluşturduğumuz slider daki alan değerini alıyoruz
        if hesaplanan_area > area_eşik_değeri:      #Hesaplanan alan biizm nelirlediğimiz alandan büyük olnları yazdırmak  için if fonksiyonunu koyuyoruz
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            uzunluk = cv2.arcLength(cnt, True)#yay uzunluğu fonksiyonuyla contour uzunluğunu buluyoruz
            approx = cv2.approxPolyDP(cnt, 0.02 * uzunluk, True)#Ele aldığımız nasıl bir şekle sahip olduğunu anlamak için Poly fonksiyonundan yararlanıyoruz
            print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)# belilediğimiz şeklin x y  genişlik ve yukseklik değerlerini buluyoruz
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)   #contours çevresine  alınan değerlere göre bir kare çiziyoruz.
            yüzde_hata = ((int(hesaplanan_area)-3755 ) / 3755)%100
            """ Ekrabna çizilen kenarın çevresine alan ve hata pauımızı yazdırıyoruz """
            cv2.putText(imgContour, "Van golunun gercek alani 3755 dir.", (x + w - 470, y -60), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 255), 2)
            cv2.putText(imgContour, "Yuzde hata Payi =%" + str(int(yüzde_hata)),(x + w - 470, y - 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 255), 2)
            cv2.putText(imgContour, "Alan: " + str(int(hesaplanan_area)), (x + w + 0, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 255, 0), 2)

#Burası true yapıldığında  aşama aşama blur,gray,Canny,dilate aşamalarını tek tek çıktılar şeklinde görülebilmektedir.
while True:

    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)  # resmimize 7 7 lik bir bluur ekliyoruz
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)# resme BGR2GRAY ile  gri tona dönüştürüyoruz
    imgStack_BG=stackImages(0.8,([img,imgBlur,imgGray]))
    cv2.imshow("Bluur-Gray", imgStack_BG)#stackImages ile birleşen resmi ekrana basıyoruz
    cv2.waitKey(1)#resimleri ekranda gösterirken sıkıntı olmaması için içine 1 değri verdim

    """thresold kaydırma cubuğundaki o anki değeri getTrackPos ile alarak threshold1 atıyoruz"""
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    """threshold1 ve 2 değerlerini aldıktan sonra  en son oluşturduğumuz gri resme bu değerlerle candy
    burada canny işleminde gürültüleri ortadan kaldırarak şeklin belirlenmesiamaçlanıyor bunuda thresold değerleriyle oynayarak buluyoruz"""
    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    imgStack_canny = stackImages(0.8,([img,imgGray,imgCanny]))
    cv2.imshow("Gray-Canny",imgStack_canny)#"""Etrafındaki gürültülerden arınmış resmi ekrana veriyoruz Buraya kadar resmin biizm istediğimiz kısmını bulmayı başardık"""
    cv2.waitKey(1)

    """Görüntüyü güzelleştirmek için  genişleme fonksiyonuna sokmamız gerekliymiş bunun içinde 5 5 lik bir çekirdek tanımlamalıyız."""
    cekirdek = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, cekirdek, iterations=1)#dilate ile resmi genişleme fonksiyonua gönderiyoruz iterasyon 1 den daha yuksek sayı yaptığımızda kayığplar artmaktadır.
    imgStack_imgDil = stackImages(0.8,([imgGray,imgCanny],[imgDil,imgDil]))
    cv2.imshow("imgStack_imgDil",imgStack_imgDil)  # Etrafındaki gürültülerden arınmış resmi ekrana veriyoruz Buraya kadar resmin biizm istediğimiz kısmını bulmayı başardık
    cv2.waitKey(1)

    getContours(imgDil,imgContour)

    """resmimizde istediğimiz van gölünün işaaretlenmiş halini ekrana yazdırıyoruz"""
    imgStack_imgContour = stackImages(0.8, ([imgGray, imgCanny], [imgDil, imgContour]))
    cv2.imshow("imgStack_imgContour", imgStack_imgContour)
    cv2.waitKey(1)



    #stackImages fonksiyonu resimleri yan yana yazılmasını sağlamaktadır.
    imgStack = stackImages(0.8,([img,imgCanny],[imgDil,imgContour]))
    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Burada true yapılırsa tum adımlar geçildikten sonra kodum tek çıktısı verilmekte
if False:
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)  # resmimize 7 7 lik bir bluur ekliyoruz
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)# resme BGR2GRAY ile  gri tona dönüştürüyoruz
    """Net görüntü elde edilen threshold değerleri"""
    threshold1 =170
    threshold2 = 240
    """threshold1 ve 2 değerlerini aldıktan sonra  en son oluşturduğumuz gri resme bu değerlerle candy
       burada canny işleminde gürültüleri ortadan kaldırarak şeklin belirlenmesiamaçlanıyor bunuda thresold değerleriyle oynayarak buluyoruz"""
    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    cekirdek = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, cekirdek, iterations=1)
    getContours(imgDil,imgContour)
    imgStack_imgContour = stackImages(0.8, ( [imgContour]))
    cv2.imshow("imgStack_imgContour", imgStack_imgContour)
    cv2.waitKey(0)
