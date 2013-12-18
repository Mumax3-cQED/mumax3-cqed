package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var kernmulC_code cu.Function

type kernmulC_args struct {
	arg_fftM unsafe.Pointer
	arg_fftK unsafe.Pointer
	arg_Nx   int
	arg_Ny   int
	argptr   [4]unsafe.Pointer
}

// Wrapper for kernmulC CUDA kernel, asynchronous.
func k_kernmulC_async(fftM unsafe.Pointer, fftK unsafe.Pointer, Nx int, Ny int, cfg *config) {
	if synchronous { // debug
		Sync()
	}

	if kernmulC_code == 0 {
		kernmulC_code = fatbinLoad(kernmulC_map, "kernmulC")
	}

	var _a_ kernmulC_args

	_a_.arg_fftM = fftM
	_a_.argptr[0] = unsafe.Pointer(&_a_.arg_fftM)
	_a_.arg_fftK = fftK
	_a_.argptr[1] = unsafe.Pointer(&_a_.arg_fftK)
	_a_.arg_Nx = Nx
	_a_.argptr[2] = unsafe.Pointer(&_a_.arg_Nx)
	_a_.arg_Ny = Ny
	_a_.argptr[3] = unsafe.Pointer(&_a_.arg_Ny)

	args := _a_.argptr[:]
	cu.LaunchKernel(kernmulC_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if synchronous { // debug
		Sync()
	}
}

var kernmulC_map = map[int]string{0: "",
	20: kernmulC_ptx_20,
	30: kernmulC_ptx_30,
	35: kernmulC_ptx_35}

const (
	kernmulC_ptx_20 = `
.version 3.2
.target sm_20
.address_size 64


.visible .entry kernmulC(
	.param .u64 kernmulC_param_0,
	.param .u64 kernmulC_param_1,
	.param .u32 kernmulC_param_2,
	.param .u32 kernmulC_param_3
)
{
	.reg .pred 	%p<4>;
	.reg .s32 	%r<13>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<8>;


	ld.param.u64 	%rd3, [kernmulC_param_0];
	ld.param.u64 	%rd4, [kernmulC_param_1];
	ld.param.u32 	%r3, [kernmulC_param_2];
	ld.param.u32 	%r4, [kernmulC_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd3;
	.loc 1 4 1
	mov.u32 	%r5, %ntid.x;
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %tid.x;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	.loc 1 5 1
	mov.u32 	%r8, %ntid.y;
	mov.u32 	%r9, %ctaid.y;
	mov.u32 	%r10, %tid.y;
	mad.lo.s32 	%r2, %r8, %r9, %r10;
	.loc 1 7 1
	setp.ge.s32	%p1, %r2, %r4;
	setp.ge.s32	%p2, %r1, %r3;
	or.pred  	%p3, %p2, %p1;
	.loc 1 7 1
	@%p3 bra 	BB0_2;

	.loc 1 11 1
	mad.lo.s32 	%r11, %r2, %r3, %r1;
	.loc 1 12 1
	shl.b32 	%r12, %r11, 1;
	mul.wide.s32 	%rd5, %r12, 4;
	add.s64 	%rd6, %rd2, %rd5;
	add.s64 	%rd7, %rd1, %rd5;
	.loc 1 16 1
	ld.global.f32 	%f1, [%rd7];
	.loc 1 14 1
	ld.global.f32 	%f2, [%rd6];
	.loc 1 19 1
	mul.f32 	%f3, %f2, %f1;
	.loc 1 17 1
	ld.global.f32 	%f4, [%rd7+4];
	.loc 1 15 1
	ld.global.f32 	%f5, [%rd6+4];
	.loc 1 19 1
	mul.f32 	%f6, %f5, %f4;
	sub.f32 	%f7, %f3, %f6;
	st.global.f32 	[%rd6], %f7;
	.loc 1 20 1
	mul.f32 	%f8, %f5, %f1;
	fma.rn.f32 	%f9, %f2, %f4, %f8;
	st.global.f32 	[%rd6+4], %f9;

BB0_2:
	.loc 1 21 2
	ret;
}


`
	kernmulC_ptx_30 = `
.version 3.2
.target sm_30
.address_size 64


.visible .entry kernmulC(
	.param .u64 kernmulC_param_0,
	.param .u64 kernmulC_param_1,
	.param .u32 kernmulC_param_2,
	.param .u32 kernmulC_param_3
)
{
	.reg .pred 	%p<4>;
	.reg .s32 	%r<13>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<8>;


	ld.param.u64 	%rd3, [kernmulC_param_0];
	ld.param.u64 	%rd4, [kernmulC_param_1];
	ld.param.u32 	%r3, [kernmulC_param_2];
	ld.param.u32 	%r4, [kernmulC_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd3;
	.loc 1 4 1
	mov.u32 	%r5, %ntid.x;
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %tid.x;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	.loc 1 5 1
	mov.u32 	%r8, %ntid.y;
	mov.u32 	%r9, %ctaid.y;
	mov.u32 	%r10, %tid.y;
	mad.lo.s32 	%r2, %r8, %r9, %r10;
	.loc 1 7 1
	setp.ge.s32	%p1, %r2, %r4;
	setp.ge.s32	%p2, %r1, %r3;
	or.pred  	%p3, %p2, %p1;
	.loc 1 7 1
	@%p3 bra 	BB0_2;

	.loc 1 11 1
	mad.lo.s32 	%r11, %r2, %r3, %r1;
	.loc 1 12 1
	shl.b32 	%r12, %r11, 1;
	mul.wide.s32 	%rd5, %r12, 4;
	add.s64 	%rd6, %rd2, %rd5;
	add.s64 	%rd7, %rd1, %rd5;
	.loc 1 16 1
	ld.global.f32 	%f1, [%rd7];
	.loc 1 14 1
	ld.global.f32 	%f2, [%rd6];
	.loc 1 19 1
	mul.f32 	%f3, %f2, %f1;
	.loc 1 17 1
	ld.global.f32 	%f4, [%rd7+4];
	.loc 1 15 1
	ld.global.f32 	%f5, [%rd6+4];
	.loc 1 19 1
	mul.f32 	%f6, %f5, %f4;
	sub.f32 	%f7, %f3, %f6;
	st.global.f32 	[%rd6], %f7;
	.loc 1 20 1
	mul.f32 	%f8, %f5, %f1;
	fma.rn.f32 	%f9, %f2, %f4, %f8;
	st.global.f32 	[%rd6+4], %f9;

BB0_2:
	.loc 1 21 2
	ret;
}


`
	kernmulC_ptx_35 = `
.version 3.2
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

.visible .entry kernmulC(
	.param .u64 kernmulC_param_0,
	.param .u64 kernmulC_param_1,
	.param .u32 kernmulC_param_2,
	.param .u32 kernmulC_param_3
)
{
	.reg .pred 	%p<4>;
	.reg .s32 	%r<13>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<8>;


	ld.param.u64 	%rd3, [kernmulC_param_0];
	ld.param.u64 	%rd4, [kernmulC_param_1];
	ld.param.u32 	%r3, [kernmulC_param_2];
	ld.param.u32 	%r4, [kernmulC_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd3;
	.loc 1 4 1
	mov.u32 	%r5, %ntid.x;
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %tid.x;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	.loc 1 5 1
	mov.u32 	%r8, %ntid.y;
	mov.u32 	%r9, %ctaid.y;
	mov.u32 	%r10, %tid.y;
	mad.lo.s32 	%r2, %r8, %r9, %r10;
	.loc 1 7 1
	setp.ge.s32	%p1, %r2, %r4;
	setp.ge.s32	%p2, %r1, %r3;
	or.pred  	%p3, %p2, %p1;
	.loc 1 7 1
	@%p3 bra 	BB2_2;

	.loc 1 11 1
	mad.lo.s32 	%r11, %r2, %r3, %r1;
	.loc 1 12 1
	shl.b32 	%r12, %r11, 1;
	mul.wide.s32 	%rd5, %r12, 4;
	add.s64 	%rd6, %rd2, %rd5;
	add.s64 	%rd7, %rd1, %rd5;
	.loc 1 16 1
	ld.global.nc.f32 	%f1, [%rd7];
	.loc 1 14 1
	ld.global.f32 	%f2, [%rd6];
	.loc 1 19 1
	mul.f32 	%f3, %f2, %f1;
	.loc 1 17 1
	ld.global.nc.f32 	%f4, [%rd7+4];
	.loc 1 15 1
	ld.global.f32 	%f5, [%rd6+4];
	.loc 1 19 1
	mul.f32 	%f6, %f5, %f4;
	sub.f32 	%f7, %f3, %f6;
	st.global.f32 	[%rd6], %f7;
	.loc 1 20 1
	mul.f32 	%f8, %f5, %f1;
	fma.rn.f32 	%f9, %f2, %f4, %f8;
	st.global.f32 	[%rd6+4], %f9;

BB2_2:
	.loc 1 21 2
	ret;
}


`
)
