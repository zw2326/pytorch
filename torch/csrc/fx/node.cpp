#include <torch/csrc/fx/node.h>

#include <structmember.h>

struct NodeBase {
  PyObject_HEAD int x;
};

static PyObject* NodeBase_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  PyObject* self = type->tp_alloc(type, 0);
  if (!self)
    return nullptr;
  return self;
}

static int Nodebase_init(NodeBase* self, PyObject* args, PyObject* kwds) {
  self->x = 5;
  return 0;
}

static void NodeBase_dealloc(NodeBase* self) {
  // PyObject_GC_UnTrack(self);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static struct PyGetSetDef NodeBase_properties[] = {
    {nullptr} /* Sentinel */
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static struct PyMemberDef NodeBase_members[] = {
    {"x", offsetof(NodeBase, x), T_INT, 0, nullptr},
    {nullptr} /* Sentinel */
};

static PyTypeObject NodeBaseType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C.NodeBase", /* tp_name */
    sizeof(NodeBase), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)NodeBase_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    NodeBase_members, /* tp_members */
    NodeBase_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)NodeBase_init, /* tp_init */
    nullptr, /* tp_alloc */
    NodeBase_new, /* tp_new */
};

/*
static PyModuleDef _module = {
  PyModuleDef_HEAD_INIT,
  "torch._C.fx",
  "Module containing C++ implementations",
  -1,
  NULL
};*/

bool NodeBase_init(PyObject* module) {
  if (PyType_Ready(&NodeBaseType) < 0) {
    return false;
  }
  Py_INCREF(&NodeBaseType);
  if (PyModule_AddObject(module, "NodeBase", (PyObject*)&NodeBaseType) != 0) {
    return false;
  }
  return true;
}
