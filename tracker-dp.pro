TEMPLATE = app
CONFIG += console
CONFIG -= qt
CONFIG += c++11
CONFIG += -std=c++11
QMAKE_CXXFLAGS += -std=c++11


SOURCES += main.cpp \
    Common.cpp \
    myloader.cpp \
    mylocalization.cpp \
    mydetector.cpp \
    mytracker.cpp

HEADERS += \
    Common.h \
    myloader.h \
    mylocalization.h \
    mydetector.h \
    mytracker.h

INCLUDEPATH += /usr/local/include/opencv

LIBS += -L/usr/local/lib \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_highgui \
    -lopencv_ml \
    -lopencv_video \
    -lopencv_features2d \
    -lopencv_calib3d \
    -lopencv_objdetect \
    -lopencv_contrib \
    -lopencv_legacy \
    -lopencv_flann \
    -lopencv_nonfree
