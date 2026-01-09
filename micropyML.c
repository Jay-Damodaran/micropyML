#include "py/obj.h"
#include "micropyML.h"

// place functions in ROM globals table
static const mp_rom_map_elem_t micropyML_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR_relu), MP_ROM_PTR(&relu_obj) },
    { MP_ROM_QSTR(MP_QSTR_reluN), MP_ROM_PTR(&reluN_obj) },
    { MP_ROM_QSTR(MP_QSTR_softmax), MP_ROM_PTR(&softmax_obj) },
    { MP_ROM_QSTR(MP_QSTR_confidence), MP_ROM_PTR(&confidence_obj) },
    { MP_ROM_QSTR(MP_QSTR_dropout), MP_ROM_PTR(&dropout_obj) },
    { MP_ROM_QSTR(MP_QSTR_maxpool1d), MP_ROM_PTR(&maxpool1d_obj) },
    { MP_ROM_QSTR(MP_QSTR_qconvrelu1d), MP_ROM_PTR(&qconvrelu1d_obj) },
    { MP_ROM_QSTR(MP_QSTR_conv1d), MP_ROM_PTR(&conv1d_obj) },
};
static MP_DEFINE_CONST_DICT(micropyML_module_globals, micropyML_globals_table);

//place address of globals table dict into module object
//In micropython, this shows up as a dict attribute of micropyML
//dict has function name as key and ptr to func as value, which was set up in globals table. This can be used to call func
//i.e. micropyML.__dict__['relu'](np.array([-2, 0, -1, 5, 6]) >> array([0, 0, 0, 5, 6])
const mp_obj_module_t mp_module_micropyML = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&micropyML_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_micropyML, mp_module_micropyML); // registering module w/ micropython