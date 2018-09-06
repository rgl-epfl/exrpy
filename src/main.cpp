#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <tuple> 
#include <thread>



#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImfPartType.h>
#include <ImfInputPart.h>
#include <ImfTiledInputPart.h>
#include <ImfNamespace.h>
#include <ImathNamespace.h>

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;

namespace py = pybind11;


py::array_t<float> loadExrFile(const std::string &fileName) {
    if (Imf::globalThreadCount() == 0)
        Imf::setGlobalThreadCount(std::thread::hardware_concurrency());

    InputFile file(fileName.c_str());
    Box2i dw = file.header().dataWindow();
    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;
    int nPixels = width * height;
    py::array_t<float> imgData({ height, width, 3 });
    std::vector<float> r(nPixels, 0.0f), g(nPixels, 0.0f), b(nPixels, 0.0f);
    FrameBuffer fb;
    fb.insert("R", Slice(FLOAT, (char*) &r.data()[-dw.min.x - dw.min.y * width], sizeof(float), sizeof(float) * width, 1, 1));
    fb.insert("G", Slice(FLOAT, (char*) &g.data()[-dw.min.x - dw.min.y * width], sizeof(float), sizeof(float) * width, 1, 1));
    fb.insert("B", Slice(FLOAT, (char*) &b.data()[-dw.min.x - dw.min.y * width], sizeof(float), sizeof(float) * width, 1, 1));
    file.setFrameBuffer(fb);
    file.readPixels(dw.min.y, dw.max.y);
    auto dataView = imgData.mutable_unchecked<3>();
    for (ssize_t i = 0; i < dataView.shape(0); i++) {
        for (ssize_t j = 0; j < dataView.shape(1); j++) {
            dataView(i, j, 0) = r[j + i * width];
            dataView(i, j, 1) = g[j + i * width];
            dataView(i, j, 2) = b[j + i * width];
        }
    }
    return imgData;
}


PYBIND11_MODULE(exrpy, m) {
    m.doc() = R"pbdoc(
        Simple EXR bindings.
    )pbdoc";

    m.def("read", &loadExrFile, R"pbdoc(
        Loads an exr image from disk.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}