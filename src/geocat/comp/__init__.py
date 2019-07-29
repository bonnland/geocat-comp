from . import _ncomp
import numpy as np
import xarray as xr
from dask.array.core import map_blocks

def linint2(fi, xo, yo, icycx, xmsg=None, iopt=0, meta=False):
    if xmsg is None:
        xmsg = _ncomp.dtype_default_fill[fi.dtype]

    xi = fi.coords[fi.dims[-1]].values
    yi = fi.coords[fi.dims[-2]].values
    fi_data = fi.data
    fo_chunks = list(fi.chunks)
    fo_chunks[-2:] = (yo.shape, xo.shape)
    fo = map_blocks(_ncomp._linint2, xi, yi, fi_data, xo, yo, icycx, xmsg, iopt, chunks=fo_chunks, dtype=fi.dtype, drop_axis=[fi.ndim-2, fi.ndim-1], new_axis=[fi.ndim-2, fi.ndim-1])

    result = fo.compute()

    if meta:
        coords = {k:v if k not in fi.dims[-2:] else (xo if k == fi.dims[-1] else yo) for (k, v) in fi.coords.items()}
        result = xr.DataArray(result, attrs=fi.attrs, dims=fi.dims, coords=coords)
    else:
        result = xr.DataArray(result)

    return result

def mjo_cross_segment(x, y):
  '''
  Calculate space-time cross spectrum for a single time segment.
  Prototype

	function mjo_cross_segment (
		x [*][*][*] : numeric,  
		y [*][*][*] : numeric,  
		opt     [1] : integer   
	)

	return_val  :  

  Arguments
      x
      y

  Three dimensional variable arrays which nominally are dimensioned: (time,lat,lon).
  The longitudes should be global while the latitudes should only span the south-north region of interest. The size of the 'time' dimension should be the size of the desired segment length. EG: for daily mean data, the size of the time dimension is typically 96, 128, 256, etc.
opt

  Currently, not used. Set to 0.
  Return value

  The return variable will be a three-dimensional array (16,wavenumber,frequency) containing the 16 cross spectral quantities associated with the specific time segment.

         ( 0,:,:)  -  symmetric power spectrum of x
         ( 1,:,:)  -  asymmetric power spectrum of x
         ( 2,:,:)  -  symmetric power spectrum of y
         ( 3,:,:)  -  asymmetric power spectrum of y
         ( 4,:,:)  -  symmetric cospectrum 
         ( 5,:,:)  -  asymmetric cospectrum 
         ( 6,:,:)  -  symmetric quadrature spectrum 
         ( 7,:,:)  -  asymmetric quadrature spectrum 
         ( 8,:,:)  -  symmetric coherence-squared spectrum 
         ( 9,:,:)  -  asymmetric coherence-squared spectrum 
         (10,:,:)  -  symmetric phase spectrum 
         (11,:,:)  -  asymmetric phase spectrum 
         (12,:,:)  -  symmetric component-1 phase spectrum 
         (13,:,:)  -  asymmetric component-1 phase spectrum 
         (14,:,:)  -  symmetric component-2 phase spectrum 
         (15,:,:)  -  asymmetric component-2 phase spectrum 

  The coordinate frequencies span 0.0 to 0.5. The coordinate wavenumbers span -M/2 to M/2 where M is the number of longitudes. 
  '''
  
  cross_spec = _ncomp.mjo_cross_segment(x.data, y.data)
  return xr.DataArray(cross_spec)

  
@xr.register_dataarray_accessor('ncomp')
class Ncomp(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def linint2(self, xo, yo, icycx, xmsg=None, iopt=0, meta=False):
        return linint2(self._obj, xo, yo, icycx, xmsg, iopt, meta)
