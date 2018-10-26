#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#include <ImathNamespace.h>
#include <ImfChannelList.h>
#include <ImfInputFile.h>
#include <ImfInputPart.h>
#include <ImfNamespace.h>
#include <ImfPartType.h>
#include <ImfTiledInputPart.h>

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;

namespace py = pybind11;

bool isExrFile(const std::string &fileName) {
    try {
        InputFile(fileName.c_str()).header();
        return true;
    } catch (...) {
        return false;
    }
}

class ExrInputFile {

public:
    ExrInputFile(const std::string &fileName) {
        if (!isExrFile(fileName)) {
            throw std::invalid_argument("File is not EXR file!");
        }

        m_file   = std::make_unique<InputFile>(fileName.c_str());
        m_header = m_file->header();

        Box2i dw = m_header.dataWindow();
        m_width  = dw.max.x - dw.min.x + 1;
        m_height = dw.max.y - dw.min.y + 1;
        m_header.channels().layers(m_layers);

        const auto &channels = m_header.channels();
        for (auto i = channels.begin(); i != channels.end(); ++i) {
            m_channels.push_back(i.name());
        }
    }

    std::vector<std::string> getChannelNames() const { return m_channels; }

    std::vector<std::string> getLayerNames() const {
        return std::vector<std::string>(m_layers.begin(), m_layers.end());
    }

    py::array_t<float> get() { return getChannels({ "R", "G", "B" }); }

    py::array_t<float> get(const std::string &channelOrLayerName) {
        if (m_layers.find(channelOrLayerName) != m_layers.end()) {
            std::vector<std::string> channelNames;
            ChannelList::ConstIterator layerBegin, layerEnd;
            m_header.channels().channelsInLayer(channelOrLayerName, layerBegin,
                                                layerEnd);
            for (auto j = layerBegin; j != layerEnd; ++j)
                channelNames.push_back(j.name());

            return getChannels(channelNames);
        }

        if (m_header.channels().findChannel(channelOrLayerName))
            return getChannels({ channelOrLayerName });
        throw std::invalid_argument("Channel/Layer could not be found!");
    }

    py::array_t<float> getChannels(const std::vector<std::string> &channels) {
        py::array_t<float> imgData({ m_height, m_width, channels.size() });
        std::vector<std::vector<float>> pixelData;
        Box2i dw = m_header.dataWindow();
        FrameBuffer fb;
        for (auto &c : channels) {
            if (!m_header.channels().findChannel(c))
                throw std::invalid_argument(
                    "Channel/Layer could not be found!");

            pixelData.push_back(std::vector<float>(m_width * m_height, 0.0f));
            fb.insert(c, Slice(FLOAT,
                               (char *) &(pixelData.back())
                                   .data()[-dw.min.x - dw.min.y * m_width],
                               sizeof(float), sizeof(float) * m_width, 1, 1));
        }

        m_file->setFrameBuffer(fb);
        m_file->readPixels(dw.min.y, dw.max.y);
        auto dataView = imgData.mutable_unchecked<3>();
        for (ssize_t i = 0; i < dataView.shape(0); i++) {
            for (ssize_t j = 0; j < dataView.shape(1); j++) {
                for (size_t c = 0; c < channels.size(); ++c) {
                    dataView(i, j, c) = pixelData[c][j + i * m_width];
                }
            }
        }
        return imgData;
    }

private:
    std::unique_ptr<InputFile> m_file;
    Header m_header;
    size_t m_width, m_height;
    std::vector<std::string> m_channels;
    std::set<std::string> m_layers;
};

ExrInputFile open(const std::string &fileName) {
    return ExrInputFile(fileName);
}

py::array_t<float> loadExrFile(const std::string &fileName) {
    if (Imf::globalThreadCount() == 0)
        Imf::setGlobalThreadCount(std::thread::hardware_concurrency());

    ExrInputFile f(fileName);
    return f.get();
}

void saveExrFile(const std::string &fileName, const py::array_t<float> &data) {

    if (data.ndim() != 3) {
        throw std::invalid_argument("Input tensor must be threedimensional");
    }
    int height    = data.shape(0);
    int width     = data.shape(1);
    int nChannels = data.shape(2);
    if (nChannels < 3) {
        throw std::invalid_argument(
            "Too few color channels (must have at least RGB channels)");
    }

    std::vector<std::string> channels = { "R", "G", "B" };
    if (nChannels > 3)
        channels.push_back("A");

    Header header(width, height);
    for (auto &c : channels)
        header.channels().insert(c.c_str(), Channel(FLOAT));

    OutputFile file(fileName.c_str(), header);
    FrameBuffer frameBuffer;

    std::vector<std::vector<float>> dataArrays;
    for (auto &c : channels)
        dataArrays.push_back(std::vector<float>(width * height));

    auto dataView = data.unchecked<3>();
    for (ssize_t i = 0; i < dataView.shape(0); i++) {
        for (ssize_t j = 0; j < dataView.shape(1); j++) {
            for (size_t c = 0; c < dataArrays.size(); ++c) {
                dataArrays[c][j + i * width] = dataView(i, j, c);
            }
        }
    }

    for (size_t i = 0; i < dataArrays.size(); ++i)
        frameBuffer.insert(channels[i].c_str(),
                           Slice(FLOAT, (char *) dataArrays[i].data(),
                                 sizeof(float) * 1, sizeof(float) * width));

    file.setFrameBuffer(frameBuffer);
    file.writePixels(height);
}

PYBIND11_MODULE(exrpy, m) {
    m.doc() = R"pbdoc(
        Simple EXR bindings.
    )pbdoc";

    m.def("is_exr_file", &isExrFile,
          R"pbdoc(
        Checks if a file on disk is a valid OpenEXR file.
    )pbdoc")
        .def("read", &loadExrFile, R"pbdoc(
        Loads an exr image from disk.
    )pbdoc")
        .def("write", &saveExrFile, R"pbdoc(
        Save an exr image to disk.
    )pbdoc");

    py::class_<ExrInputFile>(m, "InputFile")
        .def(py::init<const std::string &>())
        .def("get", py::overload_cast<>(&ExrInputFile::get),
             "Get the default layer")
        .def("get", py::overload_cast<const std::string &>(&ExrInputFile::get),
             "Get a specific layer or channel")
        .def("get_channels", &ExrInputFile::getChannels)
        .def_property_readonly("channels", &ExrInputFile::getChannelNames)
        .def_property_readonly("layers", &ExrInputFile::getLayerNames);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}