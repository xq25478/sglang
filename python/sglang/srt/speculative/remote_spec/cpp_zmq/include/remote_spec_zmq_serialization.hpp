#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <msgpack.hpp>
#include <pybind11/pybind11.h>

#include "remote_spec_protocol.hpp"

std::string pack_remote_spec_batch_payload(
    const std::vector<remote_spec::RemoteSpecRequest>& objs);

bool unpack_remote_spec_batch_payload(
    msgpack::unpacker& unpacker,
    const void* data,
    size_t len,
    std::vector<remote_spec::RemoteSpecRequest>& objs);

std::vector<remote_spec::RemoteSpecRequest> from_py_list(
    const pybind11::list& objs);
