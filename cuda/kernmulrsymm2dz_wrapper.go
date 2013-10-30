package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var kernmulRSymm2Dz_code cu.Function

type kernmulRSymm2Dz_args struct {
	arg_fftMz  unsafe.Pointer
	arg_fftKzz unsafe.Pointer
	arg_Nx     int
	arg_Ny     int
	argptr     [4]unsafe.Pointer
}

// Wrapper for kernmulRSymm2Dz CUDA kernel, asynchronous.
func k_kernmulRSymm2Dz_async(fftMz unsafe.Pointer, fftKzz unsafe.Pointer, Nx int, Ny int, cfg *config, str int) {
	if kernmulRSymm2Dz_code == 0 {
		kernmulRSymm2Dz_code = fatbinLoad(kernmulRSymm2Dz_map, "kernmulRSymm2Dz")
	}

	var _a_ kernmulRSymm2Dz_args

	_a_.arg_fftMz = fftMz
	_a_.argptr[0] = unsafe.Pointer(&_a_.arg_fftMz)
	_a_.arg_fftKzz = fftKzz
	_a_.argptr[1] = unsafe.Pointer(&_a_.arg_fftKzz)
	_a_.arg_Nx = Nx
	_a_.argptr[2] = unsafe.Pointer(&_a_.arg_Nx)
	_a_.arg_Ny = Ny
	_a_.argptr[3] = unsafe.Pointer(&_a_.arg_Ny)

	args := _a_.argptr[:]
	cu.LaunchKernel(kernmulRSymm2Dz_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream[str], args)
}

// Wrapper for kernmulRSymm2Dz CUDA kernel, synchronized.
func k_kernmulRSymm2Dz(fftMz unsafe.Pointer, fftKzz unsafe.Pointer, Nx int, Ny int, cfg *config) {
	const stream = 0
	k_kernmulRSymm2Dz_async(fftMz, fftKzz, Nx, Ny, cfg, stream)
	Sync(stream)
}

var kernmulRSymm2Dz_map = map[int]string{0: "",
	20: kernmulRSymm2Dz_ptx_20,
	30: kernmulRSymm2Dz_ptx_30,
	35: kernmulRSymm2Dz_ptx_35}

const (
	kernmulRSymm2Dz_ptx_20 = `
.version 3.1
.target sm_20
.address_size 64


.visible .entry kernmulRSymm2Dz(
	.param .u64 kernmulRSymm2Dz_param_0,
	.param .u64 kernmulRSymm2Dz_param_1,
	.param .u32 kernmulRSymm2Dz_param_2,
	.param .u32 kernmulRSymm2Dz_param_3
)
{
	.reg .pred 	%p<5>;
	.reg .s32 	%r<26>;
	.reg .f32 	%f<6>;
	.reg .s64 	%rd<11>;


	ld.param.u64 	%rd3, [kernmulRSymm2Dz_param_0];
	ld.param.u64 	%rd4, [kernmulRSymm2Dz_param_1];
	ld.param.u32 	%r3, [kernmulRSymm2Dz_param_2];
	ld.param.u32 	%r4, [kernmulRSymm2Dz_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd3;
	.loc 2 19 1
	mov.u32 	%r5, %ntid.y;
	mov.u32 	%r6, %ctaid.y;
	mov.u32 	%r7, %tid.y;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	.loc 2 20 1
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r2, %r8, %r9, %r10;
	.loc 2 22 1
	setp.ge.s32 	%p1, %r2, %r3;
	setp.ge.s32 	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	@%p3 bra 	BB0_2;

	.loc 2 26 1
	mad.lo.s32 	%r11, %r1, %r4, %r2;
	.loc 2 27 1
	sub.s32 	%r12, %r4, %r1;
	mad.lo.s32 	%r13, %r12, %r3, %r2;
	.loc 2 29 1
	shl.b32 	%r14, %r11, 1;
	.loc 2 31 1
	mul.wide.s32 	%rd5, %r14, 4;
	add.s64 	%rd6, %rd2, %rd5;
	.loc 2 32 1
	add.s32 	%r15, %r14, 1;
	mul.wide.s32 	%rd7, %r15, 4;
	add.s64 	%rd8, %rd2, %rd7;
	ld.global.f32 	%f1, [%rd8];
	.loc 2 35 1
	shr.u32 	%r17, %r4, 31;
	add.s32 	%r18, %r4, %r17;
	shr.s32 	%r19, %r18, 1;
	add.s32 	%r20, %r19, 1;
	setp.lt.s32 	%p4, %r1, %r20;
	.loc 2 36 1
	selp.b32 	%r21, %r11, %r13, %p4;
	mul.wide.s32 	%rd9, %r21, 4;
	add.s64 	%rd10, %rd1, %rd9;
	.loc 2 41 1
	ld.global.f32 	%f2, [%rd10];
	.loc 2 31 1
	ld.global.f32 	%f3, [%rd6];
	.loc 2 41 1
	mul.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd6], %f4;
	.loc 2 42 1
	mul.f32 	%f5, %f1, %f2;
	st.global.f32 	[%rd8], %f5;

BB0_2:
	.loc 2 43 2
	ret;
}


`
	kernmulRSymm2Dz_ptx_30 = `
.version 3.1
.target sm_30
.address_size 64


.visible .entry kernmulRSymm2Dz(
	.param .u64 kernmulRSymm2Dz_param_0,
	.param .u64 kernmulRSymm2Dz_param_1,
	.param .u32 kernmulRSymm2Dz_param_2,
	.param .u32 kernmulRSymm2Dz_param_3
)
{
	.reg .pred 	%p<5>;
	.reg .s32 	%r<26>;
	.reg .f32 	%f<6>;
	.reg .s64 	%rd<11>;


	ld.param.u64 	%rd3, [kernmulRSymm2Dz_param_0];
	ld.param.u64 	%rd4, [kernmulRSymm2Dz_param_1];
	ld.param.u32 	%r3, [kernmulRSymm2Dz_param_2];
	ld.param.u32 	%r4, [kernmulRSymm2Dz_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd3;
	.loc 2 19 1
	mov.u32 	%r5, %ntid.y;
	mov.u32 	%r6, %ctaid.y;
	mov.u32 	%r7, %tid.y;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	.loc 2 20 1
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r2, %r8, %r9, %r10;
	.loc 2 22 1
	setp.ge.s32 	%p1, %r2, %r3;
	setp.ge.s32 	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	@%p3 bra 	BB0_2;

	.loc 2 26 1
	mad.lo.s32 	%r11, %r1, %r4, %r2;
	.loc 2 27 1
	sub.s32 	%r12, %r4, %r1;
	mad.lo.s32 	%r13, %r12, %r3, %r2;
	.loc 2 29 1
	shl.b32 	%r14, %r11, 1;
	.loc 2 31 1
	mul.wide.s32 	%rd5, %r14, 4;
	add.s64 	%rd6, %rd2, %rd5;
	.loc 2 32 1
	add.s32 	%r15, %r14, 1;
	mul.wide.s32 	%rd7, %r15, 4;
	add.s64 	%rd8, %rd2, %rd7;
	ld.global.f32 	%f1, [%rd8];
	.loc 2 35 1
	shr.u32 	%r17, %r4, 31;
	add.s32 	%r18, %r4, %r17;
	shr.s32 	%r19, %r18, 1;
	add.s32 	%r20, %r19, 1;
	setp.lt.s32 	%p4, %r1, %r20;
	.loc 2 36 1
	selp.b32 	%r21, %r11, %r13, %p4;
	mul.wide.s32 	%rd9, %r21, 4;
	add.s64 	%rd10, %rd1, %rd9;
	.loc 2 41 1
	ld.global.f32 	%f2, [%rd10];
	.loc 2 31 1
	ld.global.f32 	%f3, [%rd6];
	.loc 2 41 1
	mul.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd6], %f4;
	.loc 2 42 1
	mul.f32 	%f5, %f1, %f2;
	st.global.f32 	[%rd8], %f5;

BB0_2:
	.loc 2 43 2
	ret;
}


`
	kernmulRSymm2Dz_ptx_35 = `
.version 3.1
.target sm_35
.address_size 64


.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	.loc 2 66 3
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	.loc 2 71 3
	ret;
}

.visible .entry kernmulRSymm2Dz(
	.param .u64 kernmulRSymm2Dz_param_0,
	.param .u64 kernmulRSymm2Dz_param_1,
	.param .u32 kernmulRSymm2Dz_param_2,
	.param .u32 kernmulRSymm2Dz_param_3
)
{
	.reg .pred 	%p<5>;
	.reg .s32 	%r<25>;
	.reg .f32 	%f<6>;
	.reg .s64 	%rd<11>;


	ld.param.u64 	%rd3, [kernmulRSymm2Dz_param_0];
	ld.param.u64 	%rd4, [kernmulRSymm2Dz_param_1];
	ld.param.u32 	%r3, [kernmulRSymm2Dz_param_2];
	ld.param.u32 	%r4, [kernmulRSymm2Dz_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd3;
	.loc 3 19 1
	mov.u32 	%r5, %ntid.y;
	mov.u32 	%r6, %ctaid.y;
	mov.u32 	%r7, %tid.y;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	.loc 3 20 1
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r2, %r8, %r9, %r10;
	.loc 3 22 1
	setp.ge.s32 	%p1, %r2, %r3;
	setp.ge.s32 	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	@%p3 bra 	BB2_2;

	.loc 3 26 1
	mad.lo.s32 	%r11, %r1, %r4, %r2;
	.loc 3 27 1
	sub.s32 	%r12, %r4, %r1;
	mad.lo.s32 	%r13, %r12, %r3, %r2;
	.loc 3 29 1
	shl.b32 	%r14, %r11, 1;
	.loc 3 31 1
	mul.wide.s32 	%rd5, %r14, 4;
	add.s64 	%rd6, %rd2, %rd5;
	ld.global.f32 	%f1, [%rd6];
	.loc 3 32 1
	add.s32 	%r16, %r14, 1;
	mul.wide.s32 	%rd7, %r16, 4;
	add.s64 	%rd8, %rd2, %rd7;
	ld.global.f32 	%f2, [%rd8];
	.loc 3 35 1
	shr.u32 	%r18, %r4, 31;
	add.s32 	%r19, %r4, %r18;
	shr.s32 	%r20, %r19, 1;
	add.s32 	%r21, %r20, 1;
	setp.lt.s32 	%p4, %r1, %r21;
	.loc 3 36 1
	selp.b32 	%r22, %r11, %r13, %p4;
	mul.wide.s32 	%rd9, %r22, 4;
	add.s64 	%rd10, %rd1, %rd9;
	.loc 3 41 1
	ld.global.nc.f32 	%f3, [%rd10];
	mul.f32 	%f4, %f1, %f3;
	st.global.f32 	[%rd6], %f4;
	.loc 3 42 1
	mul.f32 	%f5, %f2, %f3;
	st.global.f32 	[%rd8], %f5;

BB2_2:
	.loc 3 43 2
	ret;
}


`
)