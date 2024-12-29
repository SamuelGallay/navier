__kernel void add(__global float2* buffer, float scalar) {
    buffer[get_global_id(0)].x += scalar;
}
        
// diff in first coord, scalar is 2*pi/L
__kernel void mdiff_x(__global float2* buffer_in, __global float2* buffer_out, int N, float scalar) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float x = buffer_in[i*N +j].x;
    float y = buffer_in[i*N +j].y;
    float freq = scalar * ((float)i - (float)N * (2*i >= N));
    buffer_out[i*N +j].x =  y * freq;
    buffer_out[i*N +j].y = -x * freq;
}
__kernel void diff_y(__global float2* buffer_in, __global float2* buffer_out, int N, float scalar) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float x = buffer_in[i*N +j].x;
    float y = buffer_in[i*N +j].y;
    float freq = scalar * ((float)j - (float)N * (2*j >= N));
    buffer_out[i*N +j].x = -y * freq;
    buffer_out[i*N +j].y =  x * freq;
}
__kernel void inv_mlap(__global float2* buffer_in, __global float2* buffer_out, int N, float scalar) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float re = buffer_in[i*N +j].x;
    float im = buffer_in[i*N +j].y;
    float freqi = scalar * ((float)i - (float)N * (2*i >= N));
    float freqj = scalar * ((float)j - (float)N * (2*j >= N));
    float s = freqi*freqi + freqj*freqj + ((i==0) && (j==0));
    buffer_out[i*N +j].x = re / s;
    buffer_out[i*N +j].y = im / s;
}
__kernel void advection(__global float2* w_in, __global float2* w_out, __global float2* ux, __global float2* uy, int N, float L, float dt) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    float ci = (float)i - dt*ux[i*N+j].x*(float)N/L;  
    float cj = (float)j - dt*uy[i*N+j].x*(float)N/L;
    int ei = (int) floor(ci);
    int ej = (int) floor(cj);
    float di = ci - (float)ei;
    float dj = cj - (float)ej;
    ei = ((ei % N) + N) % N;
    ej = ((ej % N) + N) % N;

    float s = 0;
    //s += w_in[i*N + ((j+250)%N)].x;
    //s += w_in[((i+0)%N)*N + ((j+256)%N)].x;
    s += (1-di)*(1-dj) * w_in[( ei    % N)*N +( ej    % N)].x;
    s += (1-di)*   dj  * w_in[( ei    % N)*N +((ej+1) % N)].x;
    s +=    di *(1-dj) * w_in[((ei+1) % N)*N +( ej    % N)].x;
    s +=    di *   dj  * w_in[((ei+1) % N)*N +((ej+1) % N)].x;
    
    w_out[i*N +j].x = s;
    w_out[i*N +j].y = 0;
}
