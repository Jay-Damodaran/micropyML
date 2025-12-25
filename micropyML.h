//
// Created by Jay on 11/16/2025.
//

#ifndef MICROPYML_H
#define MICROPYML_H

// function that performs relu activation on ndarray
mp_obj_t relu(mp_obj_t x);

// function that performs softmax activation on integer ndarray. It is assumed that max ndarray value <= INT_MAX.
// returns fixed point ndarray with dtype uint16. max value is 10000, which represents 1.
// move decimal over between 0 and 2 places to get confidence percent that is displayed.
mp_obj_t softmax(mp_obj_t x);

// function that performs dropout on ndarray with probability p
// p is 0.5 by default
mp_obj_t dropout(size_t n_args, const mp_obj_t *args);

#define MIN_FOR_TYPE(TYPE) \
    _Generic((TYPE)0, \
        float: -INFINITY, \
        int8_t: INT8_MIN, \
        int16_t: INT16_MIN, \
        uint8_t: 0,             \
        uint16_t: 0             \
    )
#define MAX_FOR_TYPE(TYPE) \
    _Generic((TYPE)0, \
        float: INFINITY, \
        int8_t: INT8_MAX, \
        int16_t: INT16_MAX, \
        uint8_t: UINT8_MAX, \
        uint16_t: UINT16_MAX \
    )
//function that performs 1D max pool on input ndarray. Assumes stride=kernel and input ndarray is 3D
//default kernel size/stride is 2
mp_obj_t maxpool1d(size_t n_args, const mp_obj_t *args);

//function that performs 1D convolution on an input ndarray. Assumes stride=1, 'same' padding, and input ndarray is 3D
//bias is an optional argument
//x is shape (B, Cin, N)
//kernel is shape (Cout, Cin, K)
//bias is shape (Cout)
//output is shape (B, Cout, N)
mp_obj_t conv1d(size_t n_args, const mp_obj_t *args);

//function that performs 1D quantized convolution on ndarray of dtype 8/16-bit uint/int
//Assumes stride=1, 'same' padding, and both x and kernel are 3D.
//3-element indexable object is an optional argument that specifies the quantization multiplier, shift, and output zero_pt
//bias is an optional argument and is passed as 3rd argument
//supports
mp_obj_t qconvrelu1d(size_t n_args, const mp_obj_t *args);

MP_DECLARE_CONST_FUN_OBJ_1(relu_obj);
MP_DECLARE_CONST_FUN_OBJ_1(softmax_obj);
MP_DECLARE_CONST_FUN_OBJ_1(confidence_obj);
MP_DECLARE_CONST_FUN_OBJ_VAR_BETWEEN(dropout_obj);
MP_DECLARE_CONST_FUN_OBJ_VAR_BETWEEN(maxpool1d_obj);
MP_DECLARE_CONST_FUN_OBJ_VAR_BETWEEN(conv1d_obj);
MP_DECLARE_CONST_FUN_OBJ_VAR_BETWEEN(qconvrelu1d_obj);

#endif //MICROPYML_H
