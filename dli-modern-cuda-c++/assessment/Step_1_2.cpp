%%writefile Sources/maxwell.cu
#include "dli.h"

// FIXME(Step 1):
// accept device containers instead of `std::vector<float>`
void update_hx(int n, float dx, float dy, float dt, thrust::device_vector<float> &hx, thrust::device_vector<float> &ez, thrust::device_vector<float> &buffer)
{    
  // FIXME(Step 2):
  // Use zip and transform iterators to avoid materializing `ez[i + n] - ez[i]`
  // FIXME(Step 1):
  // compute transformation on GPU
  // std::transform(ez.begin() + n, ez.end(), ez.begin(), buffer.begin(), [](float x, float y) { return x - y; });
//thrust::transform(thrust::device, ez.begin() + n, ez.end(), ez.begin(), buffer.begin(), [] __host__ __device__(float x, float y) { return x - y; });
auto zip = thrust::make_zip_iterator(ez.begin() + n, ez.begin());
auto buffer2 = thrust::make_transform_iterator(zip, []__host__ __device__(thrust::tuple<float, float> t) { return thrust::get<0>(t) - thrust::get<1>(t); });
thrust::transform(thrust::device, hx.begin(), hx.end() - n, buffer2, hx.begin(), [dt, dx, dy] __host__ __device__(float h, float cex) { return h - dli::C0 * dt / 1.3f * cex / dy; });

  // FIXME(Step 1):
  // compute transformation on GPU
  // std::transform(hx.begin(), hx.end() - n, buffer.begin(), hx.begin(), [dt, dx, dy](float h, float cex) { return h - dli::C0 * dt / 1.3f * cex / dy; });
  // thrust::transform(thrust::device, hx.begin(), hx.end() - n, buffer.begin(), hx.begin(), [dt, dx, dy]__host__ __device__(float h, float cex) { return h - dli::C0 * dt / 1.3f * cex / dy; });

//thrust::device_vector<float> buffer(n*n);
//thrust::transform(thrust::device, hx.begin(), hx.end() - n, buffer.begin(), hx.begin(), [dt, dx, dy] __host__ __device__(float h, float cex) { return h - dli::C0 * dt / 1.3f * cex / dy; });
// thrust::transform(thrust::device, hx.begin(), hx.end() - n, buffer.begin(), hx.begin(), [dt, dx, dy] __host__ __device__(float h, float cex) { return h - dli::C0 * dt / 1.3f * cex / dy; });

}

// FIXME(Step 1):
// accept device containers instead of `std::vector<float>`
void update_hy(int n, float dx, float dy, float dt, thrust::device_vector<float> &hy, thrust::device_vector<float> &ez, thrust::device_vector<float> &buffer)
{
  // FIXME(Step 2):
  // Use zip and transform iterators to avoid materializing `ez[i] - ez[i + 1]`
  // FIXME(Step 1):
  // compute transformation on GPU
  // std::transform(ez.begin(), ez.end() - 1, ez.begin() + 1, buffer.begin(), [](float x, float y) { return x - y; });
  // thrust::transform(thrust::device, ez.begin(), ez.end() - 1, ez.begin() + 1, buffer.begin(), []__host__ __device__(float x, float y) { return x - y; });
 auto zip = thrust::make_zip_iterator(ez.begin(), ez.begin() + 1);
 auto buffer2 = thrust::make_transform_iterator(zip, []__host__ __device__(thrust::tuple<float, float> t) { return thrust::get<0>(t) - thrust::get<1>(t); });   
 thrust::transform(thrust::device, hy.begin(), hy.end() - 1, buffer2, hy.begin(), [dt, dx, dy]__host__ __device__(float h, float cey) { return h - dli::C0 * dt / 1.3f * cey / dx; });

//thrust::transform(thrust::device, ez.begin(), ez.end() - 1, ez.begin() + 1, buffer.begin(), []__host__ __device__(float x, float y) { return x - y; });
  
    // FIXME(Step 1):
  // compute transformation on GPU
  // std::transform(hy.begin(), hy.end() - 1, buffer.begin(), hy.begin(), [dt, dx, dy](float h, float cey) { return h - dli::C0 * dt / 1.3f * cey / dx; });
  // thrust::transform(thrust::device, hy.begin(), hy.end() - 1, buffer.begin(), hy.begin(), [dt, dx, dy]__host__ __device__(float h, float cey) { return h - dli::C0 * dt / 1.3f * cey / dx; });
//thrust::device_vector<float> buffer(n*n);
//thrust::transform(thrust::device, hy.begin(), hy.end() - 1, buffer.begin(), hy.begin(), [dt, dx, dy]__host__ __device__(float h, float cey) { return h - dli::C0 * dt / 1.3f * cey / dx; });
}

// FIXME(Step 1):
// accept device containers instead of `std::vector<float>`
//void update_dz(int n, float dx, float dy, float dt, thrust::device_vector<float> &hx_vec, thrust::device_vector<float> &hy_vec, thrust::device_vector<float> &dz_vec, thrust::device_vector<int> &cell_ids)
void update_dz(int n, float dx, float dy, float dt, thrust::device_vector<float> &hx_vec, thrust::device_vector<float> &hy_vec, thrust::device_vector<float> &dz_vec)
{
  auto hx = hx_vec.begin();
  auto hy = hy_vec.begin();
  auto dz = dz_vec.begin();

  // FIXME(Step 1):
  // compute for each on GPU
    /*
      std::for_each(cell_ids.begin(), cell_ids.end(), [n, dx, dy, dt, hx, hy, dz] (int cell_id)
        {
          if (cell_id > n)
            {
                float hx_diff = hx[cell_id - n] - hx[cell_id];
                float hy_diff = hy[cell_id] - hy[cell_id - 1];
                dz[cell_id] += dli::C0 * dt * (hx_diff / dx + hy_diff / dy);
            }
        });    
    std::for_each(cell_ids.begin(), cell_ids.end(), [n, dx, dy, dt, hx, hy, dz] __host__ __device__ (int cell_id)
    {
      if (cell_id > n)
        {
            float hx_diff = hx[cell_id - n] - hx[cell_id];
            float hy_diff = hy[cell_id] - hy[cell_id - 1];
            dz[cell_id] += dli::C0 * dt * (hx_diff / dx + hy_diff / dy);
        }
    });
    */
//thrust::device_vector<int> cell_ids(n*n);
//thrust::for_each(thrust::device, cell_ids.begin(), cell_ids.end(), [n, dx, dy, dt, hx, hy, dz] __host__ __device__ (int cell_id)
thrust::for_each(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(n*n), [n, dx, dy, dt, hx, hy, dz] __host__ __device__ (int cell_id)
    {
          if (cell_id > n)
            {
                float hx_diff = hx[cell_id - n] - hx[cell_id];
                float hy_diff = hy[cell_id] - hy[cell_id - 1];
                dz[cell_id] += dli::C0 * dt * (hx_diff / dx + hy_diff / dy);
            }
    } );
}

// FIXME(Step 1):
// accept device containers instead of `std::vector<float>`
void update_ez(thrust::device_vector<float> &ez, thrust::device_vector<float> &dz)
{
  // FIXME(Step 1):
  // compute transformation on GPU
  // std::transform(dz.begin(), dz.end(), ez.begin(), [](float d) { return d / 1.3f; });
thrust::transform(thrust::device, dz.begin(), dz.end(), ez.begin(), [] __host__ __device__(float d) { return d / 1.3f; });
}

// FIXME(Step 1):
// remove this function
/*
    std::vector<float> copy_to_host(const thrust::device_vector<float> &d_vec) {
    std::vector<float> vec(d_vec.size());
    thrust::copy(d_vec.begin(), d_vec.end(), vec.begin());
    return vec;
}
*/
// Do not change the signature of this function
void simulate(int cells_along_dimension, float dx, float dy, float dt,
              thrust::device_vector<float> &d_hx,
              thrust::device_vector<float> &d_hy,
              thrust::device_vector<float> &d_dz,
              thrust::device_vector<float> &d_ez) {
  // FIXME(Step 1):
  // remove host containers and compute in the incoming device containers
    /*
      std::vector<float> hx = copy_to_host(d_hx);
      std::vector<float> hy = copy_to_host(d_hy);
      std::vector<float> dz = copy_to_host(d_dz);
      std::vector<float> ez = copy_to_host(d_ez);
    */
  // FIXME(Step 2):
    // int cells = cells_along_dimension * cells_along_dimension;   
  // Remove `cell_ids` vector and use counting iterator instead    
    /*
        thrust::device_vector<int> cell_ids(cells);
        thrust::sequence(cell_ids.begin(), cell_ids.end());    
        thrust::device_vector<int> cell_ids(cells);
        for (int i = 0; i < cells; i++)
        {
            cell_ids[i] = i;
        }
    */
                         
  // FIXME(Step 2):
  // Remove `buffer` vector and use fancy iterators instead
  thrust::device_vector<float> buffer(cells_along_dimension * cells_along_dimension);
  
    for (int step = 0; step < dli::steps; step++)
    {
        update_hx(cells_along_dimension, dx, dy, dt, d_hx, d_ez, buffer);
        update_hy(cells_along_dimension, dx, dy, dt, d_hy, d_ez, buffer);
        //update_dz(cells_along_dimension, dx, dy, dt, d_hx, d_hy, d_dz, cell_ids);
        update_dz(cells_along_dimension, dx, dy, dt, d_hx, d_hy, d_dz); 
        update_ez(d_ez, d_dz);
    }

  // FIXME(Step 1):
  // remove copy to host containers, compute in the incoming device containers
/*
  d_hx = hx;
  d_hy = hy;
  d_dz = dz;
  d_ez = ez;
*/
}