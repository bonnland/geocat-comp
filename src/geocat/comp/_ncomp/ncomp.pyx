# cython: language_level=3, boundscheck=False, embedsignature=True
cimport ncomp
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

import cython
import numpy as np
cimport numpy as np
import functools
import warnings

class NcompWarning(Warning):
    pass

class NcompError(Exception):
    pass

def carrayify(f):
    """
    A decorator that ensures that :class:`numpy.ndarray` arguments are
    C-contiguous in memory. The decorator function takes no arguments.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        new_args = list(args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, np.ndarray) and not arg.flags.carray:
                new_args[i] = np.ascontiguousarray(arg)
        return f(*new_args, **kwargs)
    return wrapper


cdef class Array:
    cdef ncomp.ncomp_array* ncomp_array
    cdef np.ndarray         numpy_array
    cdef int                ndim
    cdef int                type
    cdef void*              addr
    cdef size_t*            shape

    cdef ncomp.ncomp_array* np_to_ncomp_array(self):
        return <ncomp.ncomp_array*> ncomp.ncomp_array_alloc(self.addr, self.type, self.ndim, self.shape)

    cdef np.ndarray ncomp_to_np_array(self):
        np.import_array()
        nparr = np.PyArray_SimpleNewFromData(self.ndim, <np.npy_intp *> self.shape, self.type, self.addr)
        cdef extern from "numpy/arrayobject.h":
            void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
        PyArray_ENABLEFLAGS(nparr, np.NPY_OWNDATA)
        return nparr

    @staticmethod
    cdef Array from_np(np.ndarray nparr):
        cdef Array a = Array.__new__(Array)
        a.numpy_array = nparr
        a.ndim = nparr.ndim
        a.shape = <size_t*>nparr.shape
        a.type = nparr.dtype.num
        a.addr = <void*> (<unsigned long> nparr.__array_interface__['data'][0])
        a.ncomp_array = a.np_to_ncomp_array()
        return a

    @staticmethod
    cdef Array from_ncomp(ncomp.ncomp_array* ncarr):
        cdef Array a = Array.__new__(Array)
        a.ncomp_array = ncarr
        a.ndim = ncarr.ndim
        a.shape = ncarr.shape
        a.type = ncarr.type
        a.numpy_array = a.ncomp_to_np_array()
        return a

    def __dealloc__(self):
        ncomp.ncomp_array_free(self.ncomp_array, 1)


dtype_default_fill = {
             "DEFAULT_FILL":       ncomp.DEFAULT_FILL_DOUBLE,
             np.dtype(np.int8):    np.int8(ncomp.DEFAULT_FILL_INT8),
             np.dtype(np.uint8):   np.uint8(ncomp.DEFAULT_FILL_UINT8),
             np.dtype(np.int16):   np.int16(ncomp.DEFAULT_FILL_INT16),
             np.dtype(np.uint16):  np.uint16(ncomp.DEFAULT_FILL_UINT16),
             np.dtype(np.int32):   np.int32(ncomp.DEFAULT_FILL_INT32),
             np.dtype(np.uint32):  np.uint32(ncomp.DEFAULT_FILL_UINT32),
             np.dtype(np.int64):   np.int64(ncomp.DEFAULT_FILL_INT64),
             np.dtype(np.uint64):  np.uint64(ncomp.DEFAULT_FILL_UINT64),
             np.dtype(np.float32): np.float32(ncomp.DEFAULT_FILL_FLOAT),
             np.dtype(np.float64): np.float64(ncomp.DEFAULT_FILL_DOUBLE),
            }


dtype_to_ncomp = {np.dtype(np.bool):       ncomp.NCOMP_BOOL,
                  np.dtype(np.int8):       ncomp.NCOMP_BYTE,
                  np.dtype(np.uint8):      ncomp.NCOMP_UBYTE,
                  np.dtype(np.int16):      ncomp.NCOMP_SHORT,
                  np.dtype(np.uint16):     ncomp.NCOMP_USHORT,
                  np.dtype(np.int32):      ncomp.NCOMP_INT,
                  np.dtype(np.uint32):     ncomp.NCOMP_UINT,
                  np.dtype(np.int64):      ncomp.NCOMP_LONG,
                  np.dtype(np.uint64):     ncomp.NCOMP_ULONG,
                  np.dtype(np.longlong):   ncomp.NCOMP_LONGLONG,
                  np.dtype(np.ulonglong):  ncomp.NCOMP_ULONGLONG,
                  np.dtype(np.float32):    ncomp.NCOMP_FLOAT,
                  np.dtype(np.float64):    ncomp.NCOMP_DOUBLE,
                  np.dtype(np.float128):   ncomp.NCOMP_LONGDOUBLE,
                 }


ncomp_to_dtype = {ncomp.NCOMP_BOOL:         np.bool,
                  ncomp.NCOMP_BYTE:         np.int8,
                  ncomp.NCOMP_UBYTE:        np.uint8,
                  ncomp.NCOMP_SHORT:        np.int16,
                  ncomp.NCOMP_USHORT:       np.uint16,
                  ncomp.NCOMP_INT:          np.int32,
                  ncomp.NCOMP_UINT:         np.uint32,
                  ncomp.NCOMP_LONG:         np.int64,
                  ncomp.NCOMP_ULONG:        np.uint64,
                  ncomp.NCOMP_LONGLONG:     np.longlong,
                  ncomp.NCOMP_ULONGLONG:    np.ulonglong,
                  ncomp.NCOMP_FLOAT:        np.float32,
                  ncomp.NCOMP_DOUBLE:       np.float64,
                  ncomp.NCOMP_LONGDOUBLE:   np.float128,
                 }


def get_default_fill(arr):
    if isinstance(arr, type(np.dtype)):
        dtype = arr
    else:
        dtype = arr.dtype

    try:
        return dtype_default_fill[dtype]
    except KeyError:
        return dtype_default_fill['DEFAULT_FILL']


def get_ncomp_type(arr):
    try:
        return dtype_to_ncomp[arr.dtype]
    except KeyError:
        raise KeyError("dtype('{}') is not a valid NCOMP type".format(arr.dtype)) from None


cdef ncomp.ncomp_array* np_to_ncomp_array(np.ndarray nparr):
    cdef int np_type = nparr.dtype.num
    cdef void* addr = <void*> (<unsigned long> nparr.__array_interface__['data'][0])
    cdef int ndim = nparr.ndim
    cdef size_t* shape = <size_t*> nparr.shape
    return <ncomp.ncomp_array*> ncomp.ncomp_array_alloc(addr, np_type, ndim, shape)


cdef np.ndarray ncomp_to_np_array(ncomp.ncomp_array* ncarr):
    np.import_array()
    nparr = np.PyArray_SimpleNewFromData(ncarr.ndim, <np.npy_intp *> ncarr.shape, ncarr.type, ncarr.addr)
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    PyArray_ENABLEFLAGS(nparr, np.NPY_OWNDATA)
    return nparr


cdef set_ncomp_msg(ncomp.ncomp_missing* ncomp_msg, num):
    ncomp_type = num.dtype.num
    if ncomp_type == ncomp.NCOMP_FLOAT:
        ncomp_msg.msg_float = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_DOUBLE:
        ncomp_msg.msg_double = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_BOOL:
        ncomp_msg.msg_bool = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_BYTE:
        ncomp_msg.msg_byte = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_UBYTE:
        ncomp_msg.msg_ubyte = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_SHORT:
        ncomp_msg.msg_short = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_USHORT:
        ncomp_msg.msg_ushort = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_INT:
        ncomp_msg.msg_int = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_UINT:
        ncomp_msg.msg_uint = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_LONG:
        ncomp_msg.msg_long = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_ULONG:
        ncomp_msg.msg_ulong = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_LONGLONG:
        ncomp_msg.msg_longlong = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_ULONGLONG:
        ncomp_msg.msg_ulonglong = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_LONGDOUBLE:
        ncomp_msg.msg_longdouble = ncomp_to_dtype[ncomp_type](num)

@carrayify
def _linint2(np.ndarray xi_np, np.ndarray yi_np, np.ndarray fi_np, np.ndarray xo_np, np.ndarray yo_np, int icycx, msg=None):
    """_linint2(xi, yi, fi, xo, yo, icycx, msg=None)

    Interpolates a regular grid to a rectilinear one using bi-linear
    interpolation.

    linint2 uses bilinear interpolation to interpolate from one
    rectilinear grid to another. The input grid may be cyclic in the x
    direction. The interpolation is first performed in the x direction,
    and then in the y direction.

    Args:

        xi (:class:`numpy.ndarray`):
            An array that specifies the X coordinates of the fi array.
            Most frequently, this is a 1D strictly monotonically
            increasing array that may be unequally spaced. In some
            cases, xi can be a multi-dimensional array (see next
            paragraph). The rightmost dimension (call it nxi) must have
            at least two elements, and is the last (fastest varying)
            dimension of fi.

            If xi is a multi-dimensional array, then each nxi subsection
            of xi must be strictly monotonically increasing, but may be
            unequally spaced. All but its rightmost dimension must be
            the same size as all but fi's rightmost two dimensions.

            For geo-referenced data, xi is generally the longitude
            array.

        yi (:class:`numpy.ndarray`):
            An array that specifies the Y coordinates of the fi array.
            Most frequently, this is a 1D strictly monotonically
            increasing array that may be unequally spaced. In some
            cases, yi can be a multi-dimensional array (see next
            paragraph). The rightmost dimension (call it nyi) must have
            at least two elements, and is the second-to-last dimension
            of fi.

            If yi is a multi-dimensional array, then each nyi subsection
            of yi must be strictly monotonically increasing, but may be
            unequally spaced. All but its rightmost dimension must be
            the same size as all but fi's rightmost two dimensions.

            For geo-referenced data, yi is generally the latitude array.

        fi (:class:`numpy.ndarray`):
            An array of two or more dimensions. If xi is passed in as an
            argument, then the size of the rightmost dimension of fi
            must match the rightmost dimension of xi. Similarly, if yi
            is passed in as an argument, then the size of the second-
            rightmost dimension of fi must match the rightmost dimension
            of yi.

            If missing values are present, then linint2 will perform the
            bilinear interpolation at all points possible, but will
            return missing values at coordinates which could not be
            used.

        xo (:class:`numpy.ndarray`):
            A one-dimensional array that specifies the X coordinates of
            the return array. It must be strictly monotonically
            increasing, but may be unequally spaced.

            For geo-referenced data, xo is generally the longitude
            array.

            If the output coordinates (xo) are outside those of the
            input coordinates (xi), then the fo values at those
            coordinates will be set to missing (i.e. no extrapolation is
            performed).

        yo (:class:`numpy.ndarray`):
            A one-dimensional array that specifies the Y coordinates of
            the return array. It must be strictly monotonically
            increasing, but may be unequally spaced.

            For geo-referenced data, yo is generally the latitude array.

            If the output coordinates (yo) are outside those of the
            input coordinates (yi), then the fo values at those
            coordinates will be set to missing (i.e. no extrapolation is
            performed).

        icycx (:obj:`bool`):
            An option to indicate whether the rightmost dimension of fi
            is cyclic. This should be set to True only if you have
            global data, but your longitude values don't quite wrap all
            the way around the globe. For example, if your longitude
            values go from, say, -179.75 to 179.75, or 0.5 to 359.5,
            then you would set this to True.

        msg (:obj:`numpy.number`):
            A numpy scalar value that represent a missing value in fi.
            This argument allows a user to use a missing value scheme
            other than NaN or masked arrays, similar to what NCL allows.

    Returns:
        :class:`numpy.ndarray`: The interpolated grid. The returned
        value will have the same dimensions as fi, except for the
        rightmost two dimensions which will have the same dimension
        sizes as the lengths of yo and xo. The return type will be
        double if fi is double, and float otherwise.

    """

    xi = Array.from_np(xi_np)
    yi = Array.from_np(yi_np)
    fi = Array.from_np(fi_np)
    xo = Array.from_np(xo_np)
    yo = Array.from_np(yo_np)


    cdef int iopt = 0
    cdef long i
    if fi.type == ncomp.NCOMP_DOUBLE:
        fo_dtype = np.float64
    else:
        fo_dtype = np.float32
    cdef np.ndarray fo_np = np.zeros(tuple([fi.shape[i] for i in range(fi.ndim - 2)] + [yo.shape[0], xo.shape[0]]), dtype=fo_dtype)

    missing_inds_fi = None

    if msg is None or np.isnan(msg): # if no missing value specified, assume NaNs
        missing_inds_fi = np.isnan(fi.numpy_array)
        msg = get_default_fill(fi.numpy_array)
    else:
        missing_inds_fi = (fi.numpy_array == msg)

    set_ncomp_msg(&(fi.ncomp_array.msg), msg) # always set missing on fi.ncomp_array

    if missing_inds_fi.any():
        fi.ncomp_array.has_missing = 1
        fi.numpy_array[missing_inds_fi] = msg

    fo = Array.from_np(fo_np)

#   release global interpreter lock
    cdef int ier
    with nogil:
        ier = ncomp.linint2(
            xi.ncomp_array, yi.ncomp_array, fi.ncomp_array,
            xo.ncomp_array, yo.ncomp_array, fo.ncomp_array,
            icycx, iopt)
#   re-acquire interpreter lock
#   check errors ier
    if ier:
        warnings.warn("linint2: {}: xi, yi, xo, and yo must be monotonically increasing".format(ier),
                      NcompWarning)

    if missing_inds_fi is not None and missing_inds_fi.any():
        fi.numpy_array[missing_inds_fi] = np.nan

    if fo.type == ncomp.NCOMP_DOUBLE:
        fo_msg = fo.ncomp_array.msg.msg_double
    else:
        fo_msg = fo.ncomp_array.msg.msg_float

    fo.numpy_array[fo.numpy_array == fo_msg] = np.nan

    return fo.numpy_array
