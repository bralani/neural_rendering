/* DO NOT EDIT THIS FILE - it is machine generated */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif
#ifdef HAVE_JAVAVM_JNI_H
#  include <JavaVM/jni.h>
#elif defined(HAVE_JNI_H)
#  include <jni.h>
#else
#  error ERROR: jni.h could not be found
#endif

/* Header for class mil_army_arl_muves_rtserver_RtServerImpl */

#ifndef _Included_mil_army_arl_muves_rtserver_RtServerImpl
#define _Included_mil_army_arl_muves_rtserver_RtServerImpl
#ifdef __cplusplus
extern "C" {
#endif
#undef mil_army_arl_muves_rtserver_RtServerImpl_serialVersionUID
#define mil_army_arl_muves_rtserver_RtServerImpl_serialVersionUID -3215090123894869218LL
#undef mil_army_arl_muves_rtserver_RtServerImpl_serialVersionUID
#define mil_army_arl_muves_rtserver_RtServerImpl_serialVersionUID -4100238210092549637LL
/* Inaccessible static: logNull */
#undef mil_army_arl_muves_rtserver_RtServerImpl_serialVersionUID
#define mil_army_arl_muves_rtserver_RtServerImpl_serialVersionUID 4974527148936298033LL
/* Inaccessible static: portParamTypes */
/* Inaccessible static: portFactoryParamTypes */
/* Inaccessible static: class_00024java_00024rmi_00024server_00024RMIClientSocketFactory */
/* Inaccessible static: class_00024java_00024rmi_00024server_00024RMIServerSocketFactory */
/* Inaccessible static: class_00024java_00024rmi_00024server_00024ServerRef */
/* Inaccessible static: usage */
/*
 * Class:     mil_army_arl_muves_rtserver_RtServerImpl
 * Method:    rtsInit
 * Signature: ([Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_mil_army_arl_muves_rtserver_RtServerImpl_rtsInit
  (JNIEnv *, jobject, jobjectArray);

/*
 * Class:     mil_army_arl_muves_rtserver_RtServerImpl
 * Method:    getDbTitle
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_mil_army_arl_muves_rtserver_RtServerImpl_getDbTitle
  (JNIEnv *, jobject);

/*
 * Class:     mil_army_arl_muves_rtserver_RtServerImpl
 * Method:    openSession
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_mil_army_arl_muves_rtserver_RtServerImpl_openSession
  (JNIEnv *, jobject);

/*
 * Class:     mil_army_arl_muves_rtserver_RtServerImpl
 * Method:    closeSession
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_mil_army_arl_muves_rtserver_RtServerImpl_closeSession
  (JNIEnv *, jobject, jint);

/*
 * Class:     mil_army_arl_muves_rtserver_RtServerImpl
 * Method:    shutdownNative
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_mil_army_arl_muves_rtserver_RtServerImpl_shutdownNative
  (JNIEnv *, jobject);

/*
 * Class:     mil_army_arl_muves_rtserver_RtServerImpl
 * Method:    shootRay
 * Signature: (Lmil/army/arl/muves/math/Point;Lmil/army/arl/muves/math/Vector3;I)Lmil/army/arl/muves/rtserver/RayResult;
 */
JNIEXPORT jobject JNICALL Java_mil_army_arl_muves_rtserver_RtServerImpl_shootRay
  (JNIEnv *, jobject, jobject, jobject, jint);

/*
 * Class:     mil_army_arl_muves_rtserver_RtServerImpl
 * Method:    getItemTree
 * Signature: (I)Lmil/army/arl/muves/rtserver/ItemTree;
 */
JNIEXPORT jobject JNICALL Java_mil_army_arl_muves_rtserver_RtServerImpl_getItemTree
  (JNIEnv *, jobject, jint);

/*
 * Class:     mil_army_arl_muves_rtserver_RtServerImpl
 * Method:    getBoundingBox
 * Signature: (I)Lmil/army/arl/muves/math/BoundingBox;
 */
JNIEXPORT jobject JNICALL Java_mil_army_arl_muves_rtserver_RtServerImpl_getBoundingBox
  (JNIEnv *, jobject, jint);

#ifdef __cplusplus
}
#endif
#endif