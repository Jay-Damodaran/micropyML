#include "py/obj.h"
#include "py/runtime.h"

#include "ulab.h"
#include "ndarray.h"
#include "ulabML.h"

#include "stdint.h"
#include "stdlib.h"
#include "time.h"
#include "limits.h"
#include "math.h"

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
    mp_obj_t argmax[x_pt->shape[1]];
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
mp_obj_t confidence(mp_obj_t x) {
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

#define DROPOUT_LOOP(type) \
    type *arr = (type *) arr_pt->array; \
    type *out_arr = (type *) out_arr_pt->array; \
    for (size_t i = 0; i < arr_pt->len; i++) {  \
        out_arr[i] = ((float) rand() / RAND_MAX) < rate ? 0 : arr[i]; \
    }

mp_obj_t dropout(size_t n_args, const mp_obj_t *args){
    // type check to ensure input is ndarray followed by float
    float rate = 0.5;
    if(!mp_obj_is_type(args[0], &ulab_ndarray_type)){
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray as first argument"));
    }
    if(n_args == 2 && !mp_obj_is_float(args[1])){
        mp_raise_TypeError(MP_ERROR_TEXT("Expected type float for dropout rate"));
    }
    else if(n_args == 2)
        rate = mp_obj_get_float(args[1]);
//    static int rand_seeded = 0;
//    if(!rand_seeded){
//        srand(time(NULL));
//        rand_seeded = 1;
//    }
    srand(time(NULL));
    ndarray_obj_t *arr_pt = MP_OBJ_TO_PTR(args[0]);
    uint8_t dtype = arr_pt->dtype;
    ndarray_obj_t *out_arr_pt = ndarray_new_dense_ndarray(arr_pt->ndim, arr_pt->shape, dtype);
    switch(dtype){
        case NDARRAY_FLOAT: {
            DROPOUT_LOOP(float)
            break;
        }
        case NDARRAY_INT16: {
            DROPOUT_LOOP(int16_t)
            break;
        }
        case NDARRAY_INT8: {
            DROPOUT_LOOP(int8_t)
            break;
        }
        case NDARRAY_UINT16: {
            DROPOUT_LOOP(uint16_t)
            break;
        }
        case NDARRAY_UINT8: {
            DROPOUT_LOOP(uint8_t)
            break;
        }
        default: {
            mp_raise_TypeError(MP_ERROR_TEXT("Unrecognized ndarray dtype"));
            break;
        }
    }
    return MP_OBJ_FROM_PTR(out_arr_pt);
}
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(dropout_obj, 1, 2, dropout);

#define MAXPOOL1D_LOOP(type) \
    type *arr = (type *) arr_pt->array; \
    type *out_arr = (type *) out_arr_pt->array;\
    type max = MIN_FOR_TYPE(type);       \
    uint32_t ref = 0;        \
    for(uint32_t i = 0; i < (arr_pt->len); i++){ \
        if(arr[i] > max) max = arr[i]; \
        if((i+1-ref) % kernel == 0){ \
            out_arr[(i-ref)/kernel] = max; \
            max = MIN_FOR_TYPE(type); \
        } \
        if((i+1-ref) % (result_shape[2] * kernel) == 0){ \
            ref += (uint32_t)((arr_pt->shape)[2] - result_shape[2] * kernel); \
            i += (uint32_t)((arr_pt->shape)[2] - result_shape[2] * kernel); \
        } \
    }

mp_obj_t maxpool1d(size_t n_args, const mp_obj_t *args){
    // preconditions ensuring valid ndarray input
    mp_obj_t x = args[0];
    if(!mp_obj_is_type(x, &ulab_ndarray_type)){
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray as first argument"));
    }
    ndarray_obj_t *arr_pt = MP_OBJ_TO_PTR(x);
    if(arr_pt->ndim != 3) mp_raise_ValueError(MP_ERROR_TEXT("Expected 3D array of form (batch, channels, time steps)"));

    //preconditions to ensure valid kernel input
    int32_t kernel;
    if(n_args == 1){
        kernel = 2;
    }
    else if(!mp_obj_is_int(args[1])){
        mp_raise_TypeError(MP_ERROR_TEXT("Expected integer kernel size"));
    }
    else{
        kernel = mp_obj_get_int(args[1]);
    }
    // extract shape from arr_pt
    size_t result_shape[3]; // Batch, channels, time-points
    result_shape[0] = (arr_pt->shape)[0];
    result_shape[1] = (arr_pt->shape)[1];
    if(kernel <= 0 || (size_t)kernel > (arr_pt->shape)[2]) mp_raise_ValueError(MP_ERROR_TEXT("Expected 0 < kernel size <= dim 2"));
    result_shape[2] = (arr_pt->shape)[2] / kernel;

    ndarray_obj_t *out_arr_pt = ndarray_new_dense_ndarray(3, result_shape, arr_pt->dtype);
    switch(arr_pt->dtype){
        case NDARRAY_FLOAT: {
            MAXPOOL1D_LOOP(float)
            break;
        }
        case NDARRAY_INT16: {
            MAXPOOL1D_LOOP(int16_t)
            break;
        }
        case NDARRAY_INT8: {
            MAXPOOL1D_LOOP(int8_t)
            break;
        }
        case NDARRAY_UINT16: {
            MAXPOOL1D_LOOP(uint16_t)
            break;
        }
        case NDARRAY_UINT8: {
            MAXPOOL1D_LOOP(uint8_t)
            break;
        }
        default: {
            mp_raise_TypeError(MP_ERROR_TEXT("Unrecognized or unsupported ndarray dtype"));
            break;
        }
    }
    return MP_OBJ_FROM_PTR(out_arr_pt);
}
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(maxpool1d_obj, 1, 2, maxpool1d);


static inline int32_t requantize(int32_t accum, int32_t multiplier, uint8_t shift, int32_t zero_pt){
    int64_t product = (int64_t)accum * (int64_t)multiplier;
    if(product > 0)
        product += ((int64_t)1 << (30 + shift));
    product >>= (31 + shift);
    return (int32_t)product + zero_pt; // saturating cast into lower precision range is calling function's responsibility
}
#define CONVBODY(type) \
    type *x_arr = (type *)x_pt->array; \
    type *kernel_arr = (type *)kernel_pt->array; \
    int32_t *bias_arr = NULL;                   \
    if(bias_pt){               \
        bias_arr = (int32_t *)bias_pt->array; \
    } \
    type *result_arr = (type *)result_pt->array; \
    int32_t dot_prod = 0;              \
    for(int32_t b = 0; b < x_pt->shape[0]; b++) {   \
        for(int32_t f = 0; f < kernel_pt->shape[0]; f++) {  \
            result_index = (b * kernel_pt->shape[0] + f) * channel_ref; \
            for(int32_t n = 0; n < result_pt->shape[2]; n++){           \
                dot_prod = bias_pt ? bias_arr[f] : 0;   \
                for(int32_t c = 0; c < x_pt->shape[1]; c++){\
                    x_index = b * batch_ref + c * channel_ref;      \
                    kernel_index = f * filter_ref + c * kernel_ref; \
                    for(int32_t k = 0; k < kernel_pt->shape[2]; k++){   \
                        xk = n + k - pad; \
                        if (xk < 0 || xk >= (int32_t)x_pt->shape[2]) continue; \
                        dot_prod += (int32_t)x_arr[x_index + xk] * (int32_t)kernel_arr[kernel_index + k];     \
                    }           \
                }               \
                dot_prod = requantize(dot_prod, m0, shift, output_zp); \
                dot_prod = dot_prod < output_zp ? output_zp : dot_prod; /*applying relu activation*/      \
                dot_prod = dot_prod > MAX_FOR_TYPE(type) ? MAX_FOR_TYPE(type) : dot_prod; \
                dot_prod = dot_prod < MIN_FOR_TYPE(type) ? MIN_FOR_TYPE(type) : dot_prod;            \
                result_arr[result_index + n] = (type)dot_prod;           \
            }                        \
        } \
    }

/* for(int32_t c = 0; c < x_pt->shape[1]; c++) { \
                x_index = b * batch_ref + c * channel_ref; \
                kernel_index = f * filter_ref + c * kernel_ref; \
                for (int32_t k = 0; k < result_pt->shape[2]; k++) { \
                    for (int32_t i = 0; i < kernel_pt->shape[2]; i++) { \
                        if ((i + k - pad) < 0 || (i + k - pad) >= (int32_t)x_pt->shape[2]) continue; \
                        result_arr[result_index + k] += x_arr[x_index + i + k - pad] * kernel_arr[kernel_index + i]; \
                    } \
                } \
            }    put a \ here if you paste it back */
mp_obj_t qconvrelu1d(size_t n_args, const mp_obj_t *args){
    ndarray_obj_t *bias_pt = NULL;
    mp_obj_t m0_obj;
    mp_obj_t shift_obj;
    mp_obj_t output_zp_obj;
    int32_t m0 = 1;
    int32_t output_zp = 0;
    uint8_t shift = 0;

    if(!mp_obj_is_type(args[0], &ulab_ndarray_type) || !mp_obj_is_type(args[1], &ulab_ndarray_type))
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarrays for input and kernel arguments"));
    ndarray_obj_t *x_pt = MP_OBJ_TO_PTR(args[0]);
    ndarray_obj_t *kernel_pt = MP_OBJ_TO_PTR(args[1]);
    if(x_pt->ndim != 3 || kernel_pt->ndim != 3)
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3D array for input and kernel"));
    if(n_args == 4){
        if(!mp_obj_is_type(args[2], &ulab_ndarray_type))
            mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray for bias argument"));
        bias_pt = MP_OBJ_TO_PTR(args[2]);
        if(bias_pt->ndim != 1)
            mp_raise_ValueError(MP_ERROR_TEXT("Expected 1D array for bias"));
        m0_obj = mp_obj_subscr(args[3], MP_OBJ_NEW_SMALL_INT(0), MP_OBJ_SENTINEL);
        shift_obj = mp_obj_subscr(args[3], MP_OBJ_NEW_SMALL_INT(1), MP_OBJ_SENTINEL);
        output_zp_obj = mp_obj_subscr(args[3], MP_OBJ_NEW_SMALL_INT(2), MP_OBJ_SENTINEL);
        if(!(mp_obj_is_int(m0_obj) && mp_obj_is_int(shift_obj) && mp_obj_is_int(output_zp_obj)))
            mp_raise_TypeError(MP_ERROR_TEXT("Expected integers for quantization params"));
        m0 = mp_obj_get_int(m0_obj);
        shift = mp_obj_get_int(shift_obj);
        output_zp = mp_obj_get_int(output_zp_obj);
    }
    else if(n_args == 3){
        m0_obj = mp_obj_subscr(args[2], MP_OBJ_NEW_SMALL_INT(0), MP_OBJ_SENTINEL);
        shift_obj = mp_obj_subscr(args[2], MP_OBJ_NEW_SMALL_INT(1), MP_OBJ_SENTINEL);
        output_zp_obj = mp_obj_subscr(args[3], MP_OBJ_NEW_SMALL_INT(2), MP_OBJ_SENTINEL);
        if(!(mp_obj_is_int(m0_obj) && mp_obj_is_int(shift_obj) && mp_obj_is_int(output_zp_obj)))
            mp_raise_TypeError(MP_ERROR_TEXT("Expected integers for quantization params"));
        m0 = mp_obj_get_int(m0_obj);
        shift = mp_obj_get_int(shift_obj);
        output_zp = mp_obj_get_int(output_zp_obj);
    }
    size_t result_shape[3];
    result_shape[0] = x_pt->shape[0];
    result_shape[1] = kernel_pt->shape[0];
    result_shape[2] = x_pt->shape[2];
    ndarray_obj_t *result_pt = ndarray_new_dense_ndarray(3, result_shape, x_pt->dtype);

    int32_t pad = (int32_t)(kernel_pt->shape[2] - 1) / 2; // padding formula for stride 1 and no dilation
    // inner loop for same padding
    int32_t channel_ref = x_pt->shape[2]; // how much index increments with each channel (N)
    int32_t kernel_ref = kernel_pt->shape[2];
    int32_t batch_ref = x_pt->shape[1] * channel_ref; // how much to increment index for each filter or batch (C)
    int32_t filter_ref = kernel_pt->shape[1] * kernel_pt->shape[2];
    int32_t x_index, kernel_index, result_index, xk;

    switch (x_pt->dtype) {
        case NDARRAY_INT16: {
            CONVBODY(int16_t)
            break;
        }
        case NDARRAY_INT8: {
            CONVBODY(int8_t)
            break;
        }
        case NDARRAY_UINT16: {
            CONVBODY(uint16_t)
            break;
        }
        case NDARRAY_UINT8: {
            CONVBODY(uint8_t)
            break;
        }
        default: {
            mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray dtype of 8 or 16 bit uint/int"));
            break;
        }
    }
    // inner loop for valid padding
    //    for(size_t k = 0; k <= (x_pt->shape[2]-kernel_pt->shape[2]); k++) {
    //        for (size_t i = 0; i < kernel_pt->shape[2]; i++) {
    //            result_arr[k] += x_arr[][][i+k] * kernel_arr[][][i];
    //        }
    //    }
    return MP_OBJ_FROM_PTR(result_pt);
}
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(qconvrelu1d_obj, 2, 4, qconvrelu1d);

mp_obj_t conv1d(size_t n_args, const mp_obj_t *args){
    ndarray_obj_t *bias_pt = NULL;

    if(!mp_obj_is_type(args[0], &ulab_ndarray_type) || !mp_obj_is_type(args[1], &ulab_ndarray_type))
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarrays for input and kernel arguments"));
    ndarray_obj_t *x_pt = MP_OBJ_TO_PTR(args[0]);
    ndarray_obj_t *kernel_pt = MP_OBJ_TO_PTR(args[1]);
    if(x_pt->dtype != NDARRAY_FLOAT || kernel_pt->dtype != NDARRAY_FLOAT)
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarrays of dtype float"));
    if(x_pt->ndim != 3 || kernel_pt->ndim != 3)
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3D array for input and kernel"));
    if(n_args == 3 && !mp_obj_is_type(args[2], &ulab_ndarray_type))
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray for bias argument"));
    else if(n_args == 3){
        bias_pt = MP_OBJ_TO_PTR(args[2]);
        if(bias_pt->ndim != 1) mp_raise_ValueError(MP_ERROR_TEXT("Expected 1D array for bias"));
        if(bias_pt->dtype != NDARRAY_FLOAT) mp_raise_TypeError(MP_ERROR_TEXT("Expected bias of dtype float"));
    }
    size_t result_shape[3];
    result_shape[0] = x_pt->shape[0];
    result_shape[1] = kernel_pt->shape[0];
    result_shape[2] = x_pt->shape[2];
    ndarray_obj_t *result_pt = ndarray_new_dense_ndarray(3, result_shape, x_pt->dtype);

    int32_t pad = (int32_t)(kernel_pt->shape[2] - 1) / 2; // padding formula for stride 1 and no dilation
    // inner loop for same padding
    int32_t channel_ref = x_pt->shape[2]; // how much index increments with each channel (N)
    int32_t kernel_ref = kernel_pt->shape[2];
    int32_t batch_ref = x_pt->shape[1] * channel_ref; // how much to increment index for each filter or batch (C)
    int32_t filter_ref = kernel_pt->shape[1] * kernel_pt->shape[2];
    int32_t x_index, kernel_index, result_index, xk;

    float *x_arr = (float *)x_pt->array;
    float *kernel_arr = (float *)kernel_pt->array;
    float *bias_arr = NULL;

    if (bias_pt) {
        bias_arr = (float *)bias_pt->array;
    }

    float *result_arr = (float *)result_pt->array;
    float dot_prod = 0;

    for(int32_t b = 0; b < x_pt->shape[0]; b++) {
        for(int32_t f = 0; f < kernel_pt->shape[0]; f++) {
            result_index = (b * kernel_pt->shape[0] + f) * channel_ref;
            for(int32_t n = 0; n < result_pt->shape[2]; n++) {
                dot_prod = bias_pt ? bias_arr[f] : 0;
                for(int32_t c = 0; c < x_pt->shape[1]; c++) {
                    x_index = b * batch_ref + c * channel_ref;
                    kernel_index = f * filter_ref + c * kernel_ref;
                    for(int32_t k = 0; k < kernel_pt->shape[2]; k++) {
                        xk = n + k - pad;
                        if(xk < 0 || xk >= (int32_t)x_pt->shape[2]) {
                            continue;
                        }
                        dot_prod += x_arr[x_index + xk] * kernel_arr[kernel_index + k];
                    }
                }
                result_arr[result_index + n] = dot_prod;
            }
        }
    }
    return MP_OBJ_FROM_PTR(result_pt);
}
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(conv1d_obj, 2, 3, conv1d);

// place functions in ROM globals table
static const mp_rom_map_elem_t ulabML_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR_relu), MP_ROM_PTR(&relu_obj) },
    { MP_ROM_QSTR(MP_QSTR_softmax), MP_ROM_PTR(&softmax_obj) },
    { MP_ROM_QSTR(MP_QSTR_confidence), MP_ROM_PTR(&confidence_obj) },
    { MP_ROM_QSTR(MP_QSTR_dropout), MP_ROM_PTR(&dropout_obj) },
    { MP_ROM_QSTR(MP_QSTR_maxpool1d), MP_ROM_PTR(&maxpool1d_obj) },
    { MP_ROM_QSTR(MP_QSTR_qconvrelu1d), MP_ROM_PTR(&qconvrelu1d_obj) },
    { MP_ROM_QSTR(MP_QSTR_conv1d), MP_ROM_PTR(&conv1d_obj) },
};
static MP_DEFINE_CONST_DICT(ulabML_module_globals, ulabML_globals_table);

//place address of globals table dict into module object
//In micropython, this shows up as a dict attribute of ulabML
//dict has function name as key and ptr to func as value, which was set up in globals table. This can be used to call func
//i.e. ulabML.__dict__['relu'](np.array([-2, 0, -1, 5, 6]) >> array([0, 0, 0, 5, 6])
const mp_obj_module_t mp_module_ulabML = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&ulabML_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_ulabML, mp_module_ulabML); // registering module w/ micropython