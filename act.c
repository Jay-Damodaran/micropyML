#include "py/obj.h"
#include "py/runtime.h"

#include "ulab.h"
#include "ndarray.h"
#include "act.h"

#include <stdint.h>

#define RELU_LOOP(type) \
    type *arr = (type *) arr_pt->array;\
    out_arr_pt = ndarray_new_dense_ndarray(arr_pt->ndim, arr_pt->shape, dtype);\
    type *out_arr = (type *) out_arr_pt->array;\
    for (size_t i = 0; i < arr_pt->len; i++) {\
        out_arr[i] = arr[i] < 0 ? 0 : arr[i];\
    }

mp_obj_t relu(const mp_obj_t x){
    // raise error if input isn't ndarray
    if(!mp_obj_is_type(x, &ulab_ndarray_type)){
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray"));
    }
    ndarray_obj_t *arr_pt = MP_OBJ_TO_PTR(x);
    uint8_t dtype = arr_pt->dtype;

    ndarray_obj_t *out_arr_pt; // declaring new array to hold outputs
    switch(dtype){
        case NDARRAY_FLOAT: {
            RELU_LOOP(float)
            break;
        }
        case NDARRAY_INT16: {
            RELU_LOOP(int16_t)
            break;
        }
        case NDARRAY_INT8: {
            RELU_LOOP(int8_t)
            break;
        }
        default: {
            mp_raise_TypeError(MP_ERROR_TEXT("Expected one of the following dtypes: np.int8, np.int16, np.float"));
            break;
        }
    }
    return MP_OBJ_FROM_PTR(out_arr_pt);
}
MP_DEFINE_CONST_FUN_OBJ_1(relu_obj, relu);


mp_obj_t reluN(size_t n_args, const mp_obj_t *args){
    int8_t N = 6;
    if(!mp_obj_is_type(args[0], &ulab_ndarray_type)){
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray"));
    }
    ndarray_obj_t *arr_pt = MP_OBJ_TO_PTR(args[0]);
    if(arr_pt->dtype != NDARRAY_INT8)
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray of dtype int8"));
    if(n_args == 2 && !mp_obj_is_int(args[1]))
        mp_raise_TypeError(MP_ERROR_TEXT("Expected integer value for N"));
    else if(n_args == 2)
        N = mp_obj_get_int(args[1]);

    int8_t *arr = (int8_t *) arr_pt->array;
    ndarray_obj_t *out_arr_pt = ndarray_new_dense_ndarray(arr_pt->ndim, arr_pt->shape, arr_pt->dtype);
    int8_t *out_arr = (int8_t *) out_arr_pt->array;
    for(size_t i = 0; i < arr_pt->len; i++) {
        out_arr[i] = arr[i] < 0 ? 0 : (arr[i] > N ? N : arr[i]);
    }
    return MP_OBJ_FROM_PTR(out_arr_pt);
}
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(reluN_obj, 1, 2, reluN);

// 2^(x_i - max + 22) / sum(2^(x - max + 22))
// Outputs are Q7.7 on interval [0, 100]
#define MAXEXP 22
#define QUANTBITS 14
#define FRACBITS 7
#define SOFTMAX_BODY(TYPE) \
    TYPE *x_arr = (TYPE *)x_pt->array; \
    size_t start;          \
    size_t end;            \
    uint16_t scaled_reciprocal;        \
    uint8_t temp_exp;      \
    for(size_t b = 0; b < x_pt->shape[1]; b++) { \
        exp_sum = 0;       \
        argmax[b] = mp_obj_new_int(0);        \
        offset = INT16_MIN;\
        start = b * x_pt->shape[2];       \
        end = (b+1) * x_pt->shape[2];\
        for(size_t i = start; i < end; i++) { /* retrieve maximum array element*/ \
            if((int16_t)x_arr[i] > offset) { \
                offset = (int16_t)x_arr[i]; \
                argmax[b] = mp_obj_new_int(i-start); \
            } \
        } \
        offset -= MAXEXP; /* ensures max logit maps to 2^22, which is representable in 32-bits*/ \
        for(size_t i = start; i < end; i++) { \
            if(x_arr[i] < offset) continue; \
            exp_sum += 1 << (uint8_t)((int16_t)x_arr[i] - offset); /* exp_sum is on [2^22, 2^32 - 1] as long as len(x_arr) < 1024*/ \
        } \
        scaled_reciprocal = (uint16_t)(((uint64_t)1 << (MAXEXP+QUANTBITS)) / exp_sum); /* 2^36 / exp_sum, which is on [16, 2^14]*/ \
        for(size_t i = start; i < end; i++){ \
            if(x_arr[i] >= offset) { \
                temp_exp = MAXEXP + offset - x_arr[i]; /* max - x, which is on [0, 22]*/ \
                prob_arr[i] = scaled_reciprocal >> temp_exp; /* in interval [0, 2^14]*/ \
                prob_arr[i] = prob_arr[i] > ((uint16_t)(100) << FRACBITS) ? (uint16_t)(100) << FRACBITS : prob_arr[i]; /* clamp to 100 in Q7.7*/ \
            } \
            else { \
                prob_arr[i] = 0; \
            } \
        }   \
    }
mp_obj_t softmax(const mp_obj_t x){
    if(!mp_obj_is_type(x, &ulab_ndarray_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray as first argument"));
    }
    ndarray_obj_t *x_pt = MP_OBJ_TO_PTR(x);
    if(x_pt->ndim != 2) { // (Batch, logits)
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 2D ndarray as argument"));
    }
    ndarray_obj_t *prob_vec = ndarray_new_dense_ndarray(x_pt->ndim, x_pt->shape, NDARRAY_UINT16);
    uint32_t exp_sum = 0;
    mp_obj_t argmax[x_pt->shape[1]]; // 1 because for 2D ndarrays first actual dim is at index 1
    uint16_t *prob_arr = (uint16_t *)prob_vec->array; // holds 14-bit fixed pt prob vals (7 int bits and 7 frac bits)
    int16_t offset = INT16_MIN;
    switch(x_pt->dtype) {
        case NDARRAY_UINT8: {
            SOFTMAX_BODY(uint8_t)
            break;
        }
        case NDARRAY_INT8: {
            SOFTMAX_BODY(int8_t)
            break;
        }
        default: {
            mp_raise_TypeError(MP_ERROR_TEXT("Softmax expects ndarray with dtype uint8 or int8"));
            break;
        }
    }
    mp_obj_t return_vals[2];
    return_vals[0] = mp_obj_new_tuple(x_pt->shape[1], argmax);
    return_vals[1] = MP_OBJ_FROM_PTR(prob_vec);
    return mp_obj_new_tuple(2, return_vals); // returns index of max prob as well as full prob vector
}
MP_DEFINE_CONST_FUN_OBJ_1(softmax_obj, softmax);

// takes Q7.7 softmax val and converts to percent confidence in range [0.00, 100.00]
mp_obj_t confidence(const mp_obj_t x) {
    if(!(mp_obj_is_int(x) || mp_obj_is_type(x, &ulab_ndarray_type)))
        mp_raise_TypeError(MP_ERROR_TEXT("Expected integer or ndarray softmax output"));
    if(mp_obj_is_int(x)) {
        uint16_t quantized_conf = mp_obj_get_int(x);
        uint8_t intbits = quantized_conf >> FRACBITS;
        uint8_t floatbits = ((quantized_conf & 0x7F) * 100 + (1 << (FRACBITS - 1))) >> FRACBITS; // +64 rounds to nearest hundredths
        mp_obj_t return_vals[2] = {MP_OBJ_NEW_SMALL_INT(intbits), MP_OBJ_NEW_SMALL_INT(floatbits)};
        return mp_obj_new_tuple(2, return_vals);
    }
    ndarray_obj_t *x_pt = MP_OBJ_TO_PTR(x);
    if(!(x_pt->dtype == NDARRAY_UINT16 && x_pt->ndim == 1))
        mp_raise_TypeError(MP_ERROR_TEXT("Expected 1D ndarray of dtype uint16"));
    uint16_t *x_arr = (uint16_t *)x_pt->array;
    mp_obj_t int_float_bits[2];
    uint16_t quantized_conf;
    mp_obj_t return_vals[x_pt->len];
    for(size_t i = 0; i < x_pt->len; i++) {
        quantized_conf = x_arr[i];
        int_float_bits[0] = MP_OBJ_NEW_SMALL_INT(quantized_conf >> FRACBITS);
        int_float_bits[1] = MP_OBJ_NEW_SMALL_INT(((quantized_conf & 0x7F) * 100 + (1 << (FRACBITS - 1))) >> FRACBITS); // +64 rounds to nearest hundredths
        return_vals[i] = mp_obj_new_tuple(2, int_float_bits);
    }
    return mp_obj_new_tuple(x_pt->len, return_vals);
}
MP_DEFINE_CONST_FUN_OBJ_1(confidence_obj, confidence);