cmake_minimum_required(VERSION 3.4)
project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( DisplayImage main.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
