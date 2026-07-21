// naxbench — IQ2_XXS routed-expert GEMM: Neural-Accelerator (matmul2d) vs
// simdgroup, at the compute-bound expert shape (out=2048, in=4096), sweeping
// token count N. Both kernels dequant IQ2 identically; only the matmul primitive
// differs. Answers res1.md §6/§7: does matmul2d beat simdgroup once the GEMM is
// compute-bound (IQ2 weights, AI~400-670 >> machine balance ~114)?
//
// build: clang -fobjc-arc -O2 misc/naxbench.m -framework Metal -framework Foundation -framework QuartzCore -o misc/naxbench
//
// C[m,n] = sum_k dequant(A_iq2)[m,k] * B[n,k]   (A weights [M x K] iq2, B acts [N x K] f32)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#import <stdio.h>
#import <stdlib.h>
#import <math.h>

// IQ2 dequant machinery copied verbatim from metal/moe.metal (tables + block + fn).
static const char *kSrc =
"#include <metal_stdlib>\n"
"#ifdef HAS_TENSOR\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"#endif\n"
"using namespace metal;\n"
"#ifdef HAS_TENSOR\n"
"using namespace mpp::tensor_ops;\n"
"#endif\n"
"#define QK_K 256\n"
"struct block_iq2_xxs { half d; ushort qs[QK_K/8]; };\n"
"struct Dims { uint M; uint N; uint K; };\n"
"static constant uchar kmask_iq2xs[8] = {1,2,4,8,16,32,64,128};\n"
"static constant uchar ksigns_iq2xs[128] = {\n"
"  0,129,130,3,132,5,6,135,136,9,10,139,12,141,142,15,144,17,18,147,20,149,150,23,24,153,154,27,156,29,30,159,\n"
"  160,33,34,163,36,165,166,39,40,169,170,43,172,45,46,175,48,177,178,51,180,53,54,183,184,57,58,187,60,189,190,63,\n"
"  192,65,66,195,68,197,198,71,72,201,202,75,204,77,78,207,80,209,210,83,212,85,86,215,216,89,90,219,92,221,222,95,\n"
"  96,225,226,99,228,101,102,231,232,105,106,235,108,237,238,111,240,113,114,243,116,245,246,119,120,249,250,123,252,125,126,255};\n"
"static constant ulong iq2xxs_grid[256] = {\n"
"0x0808080808080808,0x080808080808082b,0x0808080808081919,0x0808080808082b08,0x0808080808082b2b,0x0808080808190819,0x0808080808191908,0x08080808082b0808,\n"
"0x08080808082b082b,0x08080808082b2b08,0x08080808082b2b2b,0x0808080819080819,0x0808080819081908,0x0808080819190808,0x0808080819192b08,0x08080808192b0819,\n"
"0x08080808192b1908,0x080808082b080808,0x080808082b08082b,0x080808082b082b2b,0x080808082b2b082b,0x0808081908080819,0x0808081908081908,0x0808081908190808,\n"
"0x0808081908191919,0x0808081919080808,0x080808192b081908,0x080808192b192b08,0x0808082b08080808,0x0808082b0808082b,0x0808082b082b082b,0x0808082b2b08082b,\n"
"0x0808190808080819,0x0808190808081908,0x0808190808190808,0x08081908082b0819,0x08081908082b1908,0x0808190819080808,0x080819081908082b,0x0808190819082b08,\n"
"0x08081908192b0808,0x080819082b080819,0x080819082b081908,0x080819082b190808,0x080819082b2b1908,0x0808191908080808,0x080819190808082b,0x0808191908082b08,\n"
"0x08081919082b0808,0x080819191908192b,0x08081919192b2b19,0x080819192b080808,0x080819192b190819,0x0808192b08082b19,0x0808192b08190808,0x0808192b19080808,\n"
"0x0808192b2b081908,0x0808192b2b2b1908,0x08082b0808080808,0x08082b0808081919,0x08082b0808082b08,0x08082b0808191908,0x08082b08082b2b08,0x08082b0819080819,\n"
"0x08082b0819081908,0x08082b0819190808,0x08082b081919082b,0x08082b082b082b08,0x08082b1908081908,0x08082b1919080808,0x08082b2b0808082b,0x08082b2b08191908,\n"
"0x0819080808080819,0x0819080808081908,0x0819080808190808,0x08190808082b0819,0x0819080819080808,0x08190808192b0808,0x081908082b081908,0x081908082b190808,\n"
"0x081908082b191919,0x0819081908080808,0x0819081908082b08,0x08190819082b0808,0x0819081919190808,0x0819081919192b2b,0x081908192b080808,0x0819082b082b1908,\n"
"0x0819082b19081919,0x0819190808080808,0x0819190808082b08,0x08191908082b0808,0x08191908082b1919,0x0819190819082b19,0x081919082b080808,0x0819191908192b08,\n"
"0x08191919192b082b,0x0819192b08080808,0x0819192b0819192b,0x08192b0808080819,0x08192b0808081908,0x08192b0808190808,0x08192b0819080808,0x08192b082b080819,\n"
"0x08192b1908080808,0x08192b1908081919,0x08192b192b2b0808,0x08192b2b19190819,0x082b080808080808,0x082b08080808082b,0x082b080808082b2b,0x082b080819081908,\n"
"0x082b0808192b0819,0x082b08082b080808,0x082b08082b08082b,0x082b0819082b2b19,0x082b081919082b08,0x082b082b08080808,0x082b082b0808082b,0x082b190808080819,\n"
"0x082b190808081908,0x082b190808190808,0x082b190819080808,0x082b19081919192b,0x082b191908080808,0x082b191919080819,0x082b1919192b1908,0x082b192b2b190808,\n"
"0x082b2b0808082b08,0x082b2b08082b0808,0x082b2b082b191908,0x082b2b2b19081908,0x1908080808080819,0x1908080808081908,0x1908080808190808,0x1908080808192b08,\n"
"0x19080808082b0819,0x19080808082b1908,0x1908080819080808,0x1908080819082b08,0x190808081919192b,0x19080808192b0808,0x190808082b080819,0x190808082b081908,\n"
"0x190808082b190808,0x1908081908080808,0x19080819082b0808,0x19080819192b0819,0x190808192b080808,0x190808192b081919,0x1908082b08080819,0x1908082b08190808,\n"
"0x1908082b19082b08,0x1908082b1919192b,0x1908082b192b2b08,0x1908190808080808,0x1908190808082b08,0x19081908082b0808,0x190819082b080808,0x190819082b192b19,\n"
"0x190819190819082b,0x19081919082b1908,0x1908192b08080808,0x19082b0808080819,0x19082b0808081908,0x19082b0808190808,0x19082b0819080808,0x19082b0819081919,\n"
"0x19082b1908080808,0x19082b1919192b08,0x19082b19192b0819,0x19082b192b08082b,0x19082b2b19081919,0x19082b2b2b190808,0x1919080808080808,0x1919080808082b08,\n"
"0x1919080808190819,0x1919080808192b19,0x19190808082b0808,0x191908082b080808,0x191908082b082b08,0x1919081908081908,0x191908191908082b,0x191908192b2b1908,\n"
"0x1919082b2b190819,0x191919082b190808,0x191919082b19082b,0x1919191908082b2b,0x1919192b08080819,0x1919192b19191908,0x19192b0808080808,0x19192b0808190819,\n"
"0x19192b0808192b19,0x19192b08192b1908,0x19192b1919080808,0x19192b2b08082b08,0x192b080808081908,0x192b080808190808,0x192b080819080808,0x192b0808192b2b08,\n"
"0x192b081908080808,0x192b081919191919,0x192b082b08192b08,0x192b082b192b0808,0x192b190808080808,0x192b190808081919,0x192b191908190808,0x192b19190819082b,\n"
"0x192b19192b081908,0x192b2b081908082b,0x2b08080808080808,0x2b0808080808082b,0x2b08080808082b2b,0x2b08080819080819,0x2b0808082b08082b,0x2b08081908081908,\n"
"0x2b08081908192b08,0x2b08081919080808,0x2b08082b08190819,0x2b08190808080819,0x2b08190808081908,0x2b08190808190808,0x2b08190808191919,0x2b08190819080808,\n"
"0x2b081908192b0808,0x2b08191908080808,0x2b0819191908192b,0x2b0819192b191908,0x2b08192b08082b19,0x2b08192b19080808,0x2b08192b192b0808,0x2b082b080808082b,\n"
"0x2b082b1908081908,0x2b082b2b08190819,0x2b19080808081908,0x2b19080808190808,0x2b190808082b1908,0x2b19080819080808,0x2b1908082b2b0819,0x2b1908190819192b,\n"
"0x2b1908192b080808,0x2b19082b19081919,0x2b19190808080808,0x2b191908082b082b,0x2b19190819081908,0x2b19191919190819,0x2b192b082b080819,0x2b192b19082b0808,\n"
"0x2b2b08080808082b,0x2b2b080819190808,0x2b2b08082b081919,0x2b2b081908082b19,0x2b2b082b08080808,0x2b2b190808192b08,0x2b2b2b0819190808,0x2b2b2b1908081908};\n"
"static inline void deq_iq2(device const block_iq2_xxs *xb, short il, thread half4x4 &reg) {\n"
"    const float d = xb->d; const int ib32 = il/2; il = il%2;\n"
"    device const uint16_t *q2 = xb->qs + 4*ib32;\n"
"    const uint32_t g = q2[0]|(q2[1]<<16); const uint32_t sgn = q2[2]|(q2[3]<<16);\n"
"    thread const uint8_t *a8 = (thread const uint8_t*)&g;\n"
"    const float dl = d*(0.5f+(sgn>>28))*0.25f;\n"
"    constant uint8_t *gr = (constant uint8_t*)(iq2xxs_grid + a8[2*il+0]);\n"
"    uint8_t s = ksigns_iq2xs[(sgn>>(14*il))&127];\n"
"    for (int i=0;i<8;++i) reg[i/4][i%4] = dl*gr[i]*(s&kmask_iq2xs[i]?-1.f:1.f);\n"
"    gr = (constant uint8_t*)(iq2xxs_grid + a8[2*il+1]);\n"
"    s = ksigns_iq2xs[(sgn>>(14*il+7))&127];\n"
"    for (int i=0;i<8;++i) reg[2+i/4][i%4] = dl*gr[i]*(s&kmask_iq2xs[i]?-1.f:1.f);\n"
"}\n"
"\n"
"// ---- simdgroup IQ2 baseline: stage A(dequant) + B into tg, simdgroup mma ----\n"
"kernel void gemm_simd(device const char *srcA [[buffer(0)]],\n"
"                      device const float *B  [[buffer(1)]],\n"
"                      device float *C        [[buffer(2)]],\n"
"                      constant Dims &d       [[buffer(3)]],\n"
"                      threadgroup char *shmem[[threadgroup(0)]],\n"
"                      uint3 tgpig [[threadgroup_position_in_grid]],\n"
"                      ushort tiitg [[thread_index_in_threadgroup]],\n"
"                      ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    constexpr int NR0=64,NR1=32,NK=32,NL=NK/16,NT=128;\n"
"    const int M=d.M,N=d.N,K=d.K;\n"
"    const int r0=tgpig.y*NR0, r1=tgpig.x*NR1;\n"
"    const uint64_t rowb=(K/256)*sizeof(block_iq2_xxs);\n"
"    threadgroup half *sa=(threadgroup half*)shmem;      // [NR0 x NK]\n"
"    threadgroup half *sb=sa+NR0*NK;                     // [NR1 x NK]\n"
"    simdgroup_float8x8 acc[8]; for(int i=0;i<8;i++) acc[i]=make_filled_simdgroup_matrix<float,8,8>(0.f);\n"
"    for(int lk=0; lk<K; lk+=NK){\n"
"        for(int w=tiitg; w<NR0*NL; w+=NT){ int row=w/NL,kc=w%NL,kp=lk+kc*16,kb=kc*16;\n"
"            if(r0+row<M){ device const block_iq2_xxs *rp=(device const block_iq2_xxs*)(srcA+rowb*(r0+row));\n"
"                half4x4 t; deq_iq2(rp+kp/256,(kp/16)%16,t);\n"
"                for(short i=0;i<16;i++) sa[row*NK+kb+i]=t[i/4][i%4]; }\n"
"            else for(short i=0;i<16;i++) sa[row*NK+kb+i]=(half)0; }\n"
"        for(int w=tiitg; w<NR1*NK; w+=NT){ int col=w/NK,k=w%NK;\n"
"            sb[col*NK+k]=(r1+col<N&&lk+k<K)?(half)B[(uint64_t)(r1+col)*K+lk+k]:(half)0; }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        // each simdgroup owns an 8-col N stripe (sgitg*8), loops 8 M-subtiles\n"
"        simdgroup_half8x8 ma, mb;\n"
"        for(int k8=0;k8<NK;k8+=8){\n"
"            simdgroup_load(mb, sb + sgitg*8*NK + k8, NK, 0, true);   // [8K x 8N] from sb[col*NK+k]\n"
"            for(int t=0;t<8;t++){\n"
"                simdgroup_load(ma, sa + t*8*NK + k8, NK, 0, false);  // [8M x 8K] from sa[row*NK+k]\n"
"                simdgroup_multiply_accumulate(acc[t], ma, mb, acc[t]);\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    for(int t=0;t<8;t++){ int m=r0+t*8, n=r1+sgitg*8;\n"
"        if(m<M&&n<N) simdgroup_store(acc[t], C + (uint64_t)m*N + n, N, 0, false); }\n"
"}\n"
"\n"
"#ifdef HAS_TENSOR\n"
"// ---- matmul2d IQ2 (blueprint: double-buffered A dequant, direct-rhs B) ----\n"
"kernel void gemm_nax(device const char *srcA [[buffer(0)]],\n"
"                     device const float *B  [[buffer(1)]],\n"
"                     device float *C        [[buffer(2)]],\n"
"                     constant Dims &d       [[buffer(3)]],\n"
"                     threadgroup char *shmem[[threadgroup(0)]],\n"
"                     uint3 tgpig [[threadgroup_position_in_grid]],\n"
"                     ushort tiitg [[thread_index_in_threadgroup]]) {\n"
"    constexpr int NR0=64,NR1=64,NK=32,NL=NK/16,NT=128;\n"
"    const int M=d.M,N=d.N,K=d.K;\n"
"    const int r0=tgpig.y*NR0, r1=tgpig.x*NR1;\n"
"    const uint64_t rowb=(K/256)*sizeof(block_iq2_xxs);\n"
"    threadgroup half *sa=(threadgroup half*)shmem;\n"
"    auto tA0=tensor(sa, dextents<int32_t,2>(NK,NR0));\n"
"    auto tA1=tensor(sa+NR0*NK, dextents<int32_t,2>(NK,NR0));\n"
"    device float *Bnc=(device float*)B;\n"
"    auto tB=tensor(Bnc, dextents<int32_t,2>(K,N), array<int,2>({1,(int)K}));\n"
"    matmul2d<matmul2d_descriptor(NR1,NR0,NK,false,true,true,matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<4>> mm;\n"
"    auto cT=mm.template get_destination_cooperative_tensor<decltype(tB),decltype(tA0),float>();\n"
"    for(uint16_t i=0;i<cT.get_capacity();++i) if(cT.is_valid_element(i)) cT[i]=0.f;\n"
"    auto stage=[&](int lk, threadgroup half*buf){\n"
"        for(int w=tiitg; w<NR0*NL; w+=NT){ int row=w/NL,kc=w%NL,kp=lk+kc*16,kb=kc*16;\n"
"            if(r0+row<M){ device const block_iq2_xxs *rp=(device const block_iq2_xxs*)(srcA+rowb*(r0+row));\n"
"                half4x4 t; deq_iq2(rp+kp/256,(kp/16)%16,t);\n"
"                threadgroup half4 *d4=(threadgroup half4*)(buf+row*NK+kb); d4[0]=t[0];d4[1]=t[1];d4[2]=t[2];d4[3]=t[3]; }\n"
"            else for(short i=0;i<16;i++) buf[row*NK+kb+i]=(half)0; } };\n"
"    stage(0,sa); threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    uint sel=0;\n"
"    for(int lk=0; lk<K; lk+=NK){\n"
"        auto mA=(sel?tA1:tA0).slice(0,0); auto mB=tB.slice(lk,r1);\n"
"        mm.run(mB,mA,cT);\n"
"        int nk=lk+NK; if(nk<K){ sel^=1u; stage(nk, sel?sa+NR0*NK:sa); }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    device float *dt=C + r0 + (uint64_t)r1*M;\n"
"    auto tD=tensor(dt, dextents<int32_t,2>(NR0,NR1), array<int,2>({1,(int)M}));\n"
"    cT.store(tD);\n"
"}\n"
"#endif\n";

typedef struct { uint32_t M,N,K; } Dims;

// host copies for CPU golden reference
static const unsigned char h_kmask[8]={1,2,4,8,16,32,64,128};
static const unsigned char h_ksigns[128]={0,129,130,3,132,5,6,135,136,9,10,139,12,141,142,15,144,17,18,147,20,149,150,23,24,153,154,27,156,29,30,159,160,33,34,163,36,165,166,39,40,169,170,43,172,45,46,175,48,177,178,51,180,53,54,183,184,57,58,187,60,189,190,63,192,65,66,195,68,197,198,71,72,201,202,75,204,77,78,207,80,209,210,83,212,85,86,215,216,89,90,219,92,221,222,95,96,225,226,99,228,101,102,231,232,105,106,235,108,237,238,111,240,113,114,243,116,245,246,119,120,249,250,123,252,125,126,255};
static unsigned long long h_grid[256];
static int h_grid_init=0;
static void grid_from_src(void){ // parse the 256 hex ulongs out of kSrc's iq2xxs_grid block
  const char *p=strstr(kSrc,"iq2xxs_grid[256]"); p=strchr(p,'{'); int n=0;
  while(n<256){ const char*h=strstr(p,"0x"); if(!h)break; h_grid[n++]=strtoull(h,(char**)&p,16);} h_grid_init=1;
}
static float cpu_A(const unsigned char*A,int m,int k,int nblk,int blkb){
  if(!h_grid_init) grid_from_src();
  int blk=k/256, il=(k%256)/16, within=k%16;
  const unsigned char*bp=A+((unsigned long long)m*nblk+blk)*blkb;
  _Float16 dh; memcpy(&dh,bp,2); float dd=(float)dh;
  unsigned short qs[32]; memcpy(qs,bp+2,64);
  int ib32=il/2, ill=il%2; unsigned short*q2=qs+4*ib32;
  unsigned int g=q2[0]|(q2[1]<<16), sgn=q2[2]|(q2[3]<<16);
  unsigned char*a8=(unsigned char*)&g; float dl=dd*(0.5f+(sgn>>28))*0.25f;
  int gi= within<8 ? a8[2*ill+0] : a8[2*ill+1]; int wi=within%8;
  unsigned char*gr=(unsigned char*)&h_grid[gi];
  unsigned char sg=h_ksigns[ within<8 ? (sgn>>(14*ill))&127 : (sgn>>(14*ill+7))&127 ];
  return dl*gr[wi]*((sg&h_kmask[wi])?-1.f:1.f);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("device: %s\n", [[dev name] UTF8String]);
    int hasT=0; if (@available(macOS 26.0,*)) hasT=[dev supportsFamily:MTLGPUFamilyApple10];
    printf("Apple10(nax): %d\n\n", hasT);
    MTLCompileOptions *opt=[MTLCompileOptions new];
    if (@available(macOS 26.0,*)) opt.languageVersion=MTLLanguageVersion4_0;
    if (hasT) opt.preprocessorMacros=@{@"HAS_TENSOR":@"1"};
    NSError *e=nil;
    id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&e];
    if(!lib){printf("compile: %s\n",[[e description] UTF8String]);return 1;}
    id<MTLComputePipelineState> pS=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemm_simd"] error:&e];
    id<MTLComputePipelineState> pN=hasT?[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemm_nax"] error:&e]:nil;
    if(!pS||(hasT&&!pN)){printf("pipeline: %s\n",[[e description] UTF8String]);return 1;}
    id<MTLCommandQueue> q=[dev newCommandQueue];
    const uint32_t M=2048,K=4096,maxN=512;
    const uint32_t Ns[]={16,32,64,96,192,512};
    const uint32_t nblk=K/256, blkb=2+2*(256/8); // half d + ushort qs[32] = 66
    id<MTLBuffer> bA=[dev newBufferWithLength:(NSUInteger)M*nblk*blkb options:MTLResourceStorageModeShared];
    id<MTLBuffer> bB=[dev newBufferWithLength:(NSUInteger)maxN*K*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bC=[dev newBufferWithLength:(NSUInteger)M*maxN*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bC2=[dev newBufferWithLength:(NSUInteger)M*maxN*sizeof(float) options:MTLResourceStorageModeShared];
    unsigned char *A=(unsigned char*)bA.contents; float *B=(float*)bB.contents;
    unsigned s=777u;
    for(uint64_t i=0;i<(uint64_t)M*nblk;i++){ unsigned char*blk=A+i*blkb; _Float16 dd=(_Float16)0.05f; memcpy(blk,&dd,2);
        for(int j=0;j<32;j++){ s=s*1103515245u+12345u; unsigned short v=(s>>16)&0xffff; memcpy(blk+2+j*2,&v,2);} }
    for(uint64_t i=0;i<(uint64_t)maxN*K;i++){ s=s*1103515245u+12345u; B[i]=((int)((s>>16)&255)-128)/256.0f; }

    printf("%-6s %12s %12s %8s %10s\n","N","simd GF/s","nax GF/s","nax/simd","ckdiff");
    for(int ni=0;ni<(int)(sizeof(Ns)/sizeof(Ns[0]));ni++){
      uint32_t N=Ns[ni]; Dims dm={M,N,K}; double fl=2.0*(double)M*N*K;
      const int iters=100,warm=10; double best[2]={1e30,1e30};
      for(int wi=0;wi<(hasT?2:1);wi++){ id<MTLComputePipelineState> p=wi?pN:pS; id<MTLBuffer> out=wi?bC2:bC;
        for(int rep=0;rep<3;rep++){ id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
          [en setComputePipelineState:p]; [en setBuffer:bA offset:0 atIndex:0]; [en setBuffer:bB offset:0 atIndex:1];
          [en setBuffer:out offset:0 atIndex:2]; [en setBytes:&dm length:sizeof(dm) atIndex:3];
          int nr1 = wi?64:32;  // nax uses square 64x64 tile, simd 64x32
          [en setThreadgroupMemoryLength:(wi?(2*64*32):(64*32+32*32))*sizeof(uint16_t) atIndex:0];
          MTLSize tg=MTLSizeMake(128,1,1), gr=MTLSizeMake((N+nr1-1)/nr1,(M+63)/64,1);
          for(int it=0;it<iters+warm;it++)[en dispatchThreadgroups:gr threadsPerThreadgroup:tg];
          [en endEncoding]; double t0=CACurrentMediaTime(); [cb commit]; [cb waitUntilCompleted];
          double dt=(CACurrentMediaTime()-t0)/(iters+warm); if(dt<best[wi])best[wi]=dt; } }
      // checksum: max abs diff between the two outputs (first N cols)
      double ck=0; if(hasT){ float*c1=(float*)bC.contents,*c2=(float*)bC2.contents; // simd row-major vs nax M-major
        for(uint32_t m=0;m<M;m++)for(uint32_t n=0;n<N;n++){ double dd=fabs((double)c1[(uint64_t)m*N+n]-(double)c2[(uint64_t)m+(uint64_t)n*M]); if(dd>ck)ck=dd;} }
      double gS=fl/best[0]/1e9, gN=hasT?fl/best[1]/1e9:0;
      printf("%-6u %12.1f %12.1f %8.2f %10.4f\n",N,gS,gN,hasT?gN/gS:0,ck);
      if(ni==0){ // full CPU golden over all (m,n<N): true per-kernel error
        float*c1=(float*)bC.contents,*c2=(float*)bC2.contents;
        double eS=0,eN=0; int wsM=0,wsN=0,wnM=0,wnN=0;
        for(uint32_t m=0;m<M;m++)for(uint32_t n=0;n<N;n++){ double ref=0;
          for(uint32_t k=0;k<K;k++) ref+=(double)cpu_A(A,m,k,nblk,blkb)*(double)B[(uint64_t)n*K+k];
          double ds=fabs((double)c1[(uint64_t)m*N+n]-ref), dn=fabs((double)c2[(uint64_t)m+(uint64_t)n*M]-ref);
          if(ds>eS){eS=ds;wsM=m;wsN=n;} if(dn>eN){eN=dn;wnM=m;wnN=n;} }
        printf("  -- CPU golden (N=%u): simd max|err|=%.3f @(%d,%d)  nax max|err|=%.3f @(%d,%d) --\n",N,eS,wsM,wsN,eN,wnM,wnN);
      }
    }
  }
  return 0;
}
