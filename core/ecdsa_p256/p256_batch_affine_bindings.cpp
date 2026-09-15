// p256_batch_affine_bindings.cpp
//
// Binds the CPU golden batch-affine function into the existing
// p256_sign_persistent pybind11 module.
//
// Exposes:
//   batch_affine_from_jacobian(jacobians)
// where:
//   jacobians: list[ (X, Y, Z), ... ]
//   X, Y, Z: np.ndarray shape (8,), dtype=uint32
// returns:
//   list[ (x, y), ... ] with x,y same shape/dtype.

#include <cstdint>
#include <vector>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declarations of structures and functions from p256_batch_affine_ref.cuh
// We define them here to avoid including CUDA headers in C++ compilation

struct P256_Jacobian_Scalar {
    uint32_t X[8];
    uint32_t Y[8];
    uint32_t Z[8];
};

struct P256_Affine_Scalar {
    uint32_t x[8];
    uint32_t y[8];
};

// Forward declaration - will be provided by the .cu compilation unit
// which includes p256_batch_affine_ref.cuh
void batch_affine_from_jacobian_host(
    const P256_Jacobian_Scalar* in_points,
    P256_Affine_Scalar* out_points,
    int N
);

namespace {

py::list batch_affine_from_jacobian_py(py::sequence py_jacobians) {
    const std::size_t N = py::len(py_jacobians);
    py::list result;

    if (N == 0) {
        return result;  // empty list
    }

    // 1) Copy input from Python into P256_Jacobian_Scalar vector
    std::vector<P256_Jacobian_Scalar> in_points;
    in_points.resize(N);

    for (std::size_t i = 0; i < N; ++i) {
        py::object item = py_jacobians[i];
        py::tuple tup = item.cast<py::tuple>();
        if (py::len(tup) != 3) {
            throw std::runtime_error(
                "Each jacobian must be a tuple (X, Y, Z) of numpy arrays"
            );
        }

        auto x_arr = tup[0].cast<py::array_t<std::uint32_t, py::array::c_style>>();
        auto y_arr = tup[1].cast<py::array_t<std::uint32_t, py::array::c_style>>();
        auto z_arr = tup[2].cast<py::array_t<std::uint32_t, py::array::c_style>>();

        if (x_arr.ndim() != 1 || x_arr.shape(0) != 8 ||
            y_arr.ndim() != 1 || y_arr.shape(0) != 8 ||
            z_arr.ndim() != 1 || z_arr.shape(0) != 8) {
            throw std::runtime_error(
                "X, Y, Z must each be 1D np.ndarray of length 8 (uint32)"
            );
        }

        auto x = x_arr.unchecked<1>();
        auto y = y_arr.unchecked<1>();
        auto z = z_arr.unchecked<1>();

        P256_Jacobian_Scalar &P = in_points[i];
        for (int limb = 0; limb < 8; ++limb) {
            P.X[limb] = static_cast<std::uint32_t>(x(limb));
            P.Y[limb] = static_cast<std::uint32_t>(y(limb));
            P.Z[limb] = static_cast<std::uint32_t>(z(limb));
        }
    }

    // 2) Allocate output vector and call the golden batch function
    std::vector<P256_Affine_Scalar> out_points;
    out_points.resize(N);

    batch_affine_from_jacobian_host(
        in_points.data(),
        out_points.data(),
        N
    );

    // 3) Convert results back to Python list[(x, y)]
    for (std::size_t i = 0; i < N; ++i) {
        const P256_Affine_Scalar &A = out_points[i];

        // Create np.ndarray for x and y (shape (8,), uint32)
        py::array_t<std::uint32_t> x_arr({8});
        py::array_t<std::uint32_t> y_arr({8});

        auto x = x_arr.mutable_unchecked<1>();
        auto y = y_arr.mutable_unchecked<1>();

        for (int limb = 0; limb < 8; ++limb) {
            x(limb) = A.x[limb];
            y(limb) = A.y[limb];
        }

        result.append(py::make_tuple(std::move(x_arr), std::move(y_arr)));
    }

    return result;
}

} // anonymous namespace

// This is the hook you will call from the existing module definition
// in p256_sign_persistent.cu / .cpp.
void register_batch_affine_bindings(py::module_ &m) {
    m.def(
        "batch_affine_from_jacobian",
        &batch_affine_from_jacobian_py,
        py::arg("jacobians"),
        R"doc(
Batch-convert Jacobian points (in Montgomery domain) to affine (also Montgomery).

Args:
    jacobians: list of (X, Y, Z), where each of X, Y, Z is a 1D np.ndarray
               of length 8, dtype=np.uint32. Coordinates are P-256 field
               elements in Montgomery domain.

Returns:
    list of (x, y) affine points, each as 1D np.ndarray of length 8,
    dtype=np.uint32 (still in Montgomery domain).
)doc"
    );
}
