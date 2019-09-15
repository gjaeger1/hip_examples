cmake_minimum_required(VERSION 2.8.3)
project(hip_examples)

# includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Thrust)

# packages
find_package(Boost)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

# setup HIP
if(NOT DEFINED HIP_PATH)
	if(NOT DEFINED ENV{HIP_PATH})
	    set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
	else()
	    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
	endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(HIP QUIET REQUIRED)
if(HIP_FOUND)
	message(STATUS "Found HIP: " ${HIP_VERSION})
else()
	message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

set(HIPCC_OPTIONS)
set(HCC_OPTIONS)
set(NVCC_OPTIONS)
	

# list sources and names for targets
set(HIP_EIGEN_SOURCE_FILES src/hip_eigen.cu) # we need the .cu file-endig to work with NVCC
set(HIP_EIGEN_TARGET_NAME hip_eigen)

set(HIP_CLASSES_SOURCE_FILES src/hip_classes.cu)
set(HIP_CLASSES_TARGET_NAME hip_classes)

set(HIP_CLASSES_CRTP_SOURCE_FILES src/hip_classes_crtp.cu)
set(HIP_CLASSES_CRTP_TARGET_NAME hip_classes_crtp)

set(THRUST_BOUNDING_BOX_SOURCE_FILES src/thrust_bounding_box.cu)
set(THRUST_BOUNDING_BOX_TARGET_NAME thrust_bounding_box)

set(THRUST_ABSTRACT_CLASS_SOURCE_FILES src/thrust_abstract_class.cu)
set(THRUST_ABSTRACT_CLASS_TARGET_NAME thrust_abstract_class)

set(BOOST_THRUST_ODEINT_SOURCE_FILES src/boost_thrust_odeint.cu)
set(BOOST_THRUST_ODEINT_TARGET_NAME boost_thrust_odeint)

# add HIP targets
set_source_files_properties(${HIP_EIGEN_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${HIP_EIGEN_TARGET_NAME} ${HIP_EIGEN_SOURCE_FILES} HIPCC_OPTIONS ${HIPCC_OPTIONS} HCC_OPTIONS ${HCC_OPTIONS} NVCC_OPTIONS ${NVCC_OPTIONS})

set_source_files_properties(${HIP_CLASSES_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${HIP_CLASSES_TARGET_NAME} ${HIP_CLASSES_SOURCE_FILES} HIPCC_OPTIONS ${HIPCC_OPTIONS} HCC_OPTIONS ${HCC_OPTIONS} NVCC_OPTIONS ${NVCC_OPTIONS})

set_source_files_properties(${HIP_CLASSES_CRTP_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${HIP_CLASSES_CRTP_TARGET_NAME} ${HIP_CLASSES_CRTP_SOURCE_FILES} HIPCC_OPTIONS ${HIPCC_OPTIONS} HCC_OPTIONS ${HCC_OPTIONS} NVCC_OPTIONS ${NVCC_OPTIONS})

set_source_files_properties(${THRUST_BOUNDING_BOX_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${THRUST_ABSTRACT_CLASS_TARGET_NAME} ${THRUST_BOUNDING_BOX_SOURCE_FILES} HIPCC_OPTIONS ${HIPCC_OPTIONS} HCC_OPTIONS ${HCC_OPTIONS} NVCC_OPTIONS ${NVCC_OPTIONS})

set_source_files_properties(${THRUST_ABSTRACT_CLASS_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${THRUST_ABSTRACT_CLASS_TARGET_NAME} ${THRUST_ABSTRACT_CLASS_SOURCE_FILES} HIPCC_OPTIONS ${HIPCC_OPTIONS} HCC_OPTIONS ${HCC_OPTIONS} NVCC_OPTIONS ${NVCC_OPTIONS})

set_source_files_properties(${BOOST_THRUST_ODEINT_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${BOOST_THRUST_ODEINT_TARGET_NAME} ${BOOST_THRUST_ODEINT_SOURCE_FILES} HIPCC_OPTIONS ${HIPCC_OPTIONS} HCC_OPTIONS ${HCC_OPTIONS} NVCC_OPTIONS ${NVCC_OPTIONS})


