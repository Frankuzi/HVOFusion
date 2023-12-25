#include "../include/octree.h"

TORCH_LIBRARY(svo, m)
{
    m.class_<Octree>("Octree")
        .def(torch::init<>())
        .def("init", &Octree::init)
        .def("updateTree", &Octree::updateTree)
        .def("updateTreePcd", &Octree::updateTreePcd)
        .def("packOutput", &Octree::packOutput)
        .def("updateVerts", &Octree::updateVerts)
        .def("packVerts", &Octree::packVerts);
}