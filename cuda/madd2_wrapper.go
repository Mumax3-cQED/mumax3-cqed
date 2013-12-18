package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var madd2_code cu.Function

type madd2_args struct {
	arg_dst  unsafe.Pointer
	arg_src1 unsafe.Pointer
	arg_fac1 float32
	arg_src2 unsafe.Pointer
	arg_fac2 float32
	arg_N    int
	argptr   [6]unsafe.Pointer
}

// Wrapper for madd2 CUDA kernel, asynchronous.
func k_madd2_async(dst unsafe.Pointer, src1 unsafe.Pointer, fac1 float32, src2 unsafe.Pointer, fac2 float32, N int, cfg *config) {
	if synchronous { // debug
		Sync()
	}

	if madd2_code == 0 {
		madd2_code = fatbinLoad(madd2_map, "madd2")
	}

	var _a_ madd2_args

	_a_.arg_dst = dst
	_a_.argptr[0] = unsafe.Pointer(&_a_.arg_dst)
	_a_.arg_src1 = src1
	_a_.argptr[1] = unsafe.Pointer(&_a_.arg_src1)
	_a_.arg_fac1 = fac1
	_a_.argptr[2] = unsafe.Pointer(&_a_.arg_fac1)
	_a_.arg_src2 = src2
	_a_.argptr[3] = unsafe.Pointer(&_a_.arg_src2)
	_a_.arg_fac2 = fac2
	_a_.argptr[4] = unsafe.Pointer(&_a_.arg_fac2)
	_a_.arg_N = N
	_a_.argptr[5] = unsafe.Pointer(&_a_.arg_N)

	args := _a_.argptr[:]
	cu.LaunchKernel(madd2_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if synchronous { // debug
		Sync()
	}
}

var madd2_map = map[int]string{0: "",
	20: madd2_ptx_20,
	30: madd2_ptx_30,
	35: madd2_ptx_35}

const (
	madd2_ptx_20 = `
.version 3.2
.target sm_20
.address_size 64


.visible .entry madd2(
	.param .u64 madd2_param_0,
	.param .u64 madd2_param_1,
	.param .f32 madd2_param_2,
	.param .u64 madd2_param_3,
	.param .f32 madd2_param_4,
	.param .u32 madd2_param_5
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<7>;
	.reg .s64 	%rd<11>;


	ld.param.u64 	%rd4, [madd2_param_0];
	ld.param.u64 	%rd5, [madd2_param_1];
	ld.param.f32 	%f1, [madd2_param_2];
	ld.param.u64 	%rd6, [madd2_param_3];
	ld.param.f32 	%f2, [madd2_param_4];
	ld.param.u32 	%r2, [madd2_param_5];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd6;
	cvta.to.global.u64 	%rd3, %rd5;
	.loc 1 8 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 10 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd3, %rd7;
	.loc 1 11 1
	ld.global.f32 	%f3, [%rd8];
	add.s64 	%rd9, %rd2, %rd7;
	.loc 1 11 1
	ld.global.f32 	%f4, [%rd9];
	mul.f32 	%f5, %f4, %f2;
	fma.rn.f32 	%f6, %f3, %f1, %f5;
	add.s64 	%rd10, %rd1, %rd7;
	.loc 1 11 1
	st.global.f32 	[%rd10], %f6;

BB0_2:
	.loc 1 13 2
	ret;
}


`
	madd2_ptx_30 = `
.version 3.2
.target sm_30
.address_size 64


.visible .entry madd2(
	.param .u64 madd2_param_0,
	.param .u64 madd2_param_1,
	.param .f32 madd2_param_2,
	.param .u64 madd2_param_3,
	.param .f32 madd2_param_4,
	.param .u32 madd2_param_5
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<7>;
	.reg .s64 	%rd<11>;


	ld.param.u64 	%rd4, [madd2_param_0];
	ld.param.u64 	%rd5, [madd2_param_1];
	ld.param.f32 	%f1, [madd2_param_2];
	ld.param.u64 	%rd6, [madd2_param_3];
	ld.param.f32 	%f2, [madd2_param_4];
	ld.param.u32 	%r2, [madd2_param_5];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd6;
	cvta.to.global.u64 	%rd3, %rd5;
	.loc 1 8 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 10 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd3, %rd7;
	.loc 1 11 1
	ld.global.f32 	%f3, [%rd8];
	add.s64 	%rd9, %rd2, %rd7;
	.loc 1 11 1
	ld.global.f32 	%f4, [%rd9];
	mul.f32 	%f5, %f4, %f2;
	fma.rn.f32 	%f6, %f3, %f1, %f5;
	add.s64 	%rd10, %rd1, %rd7;
	.loc 1 11 1
	st.global.f32 	[%rd10], %f6;

BB0_2:
	.loc 1 13 2
	ret;
}


`
	madd2_ptx_35 = `
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

.visible .entry madd2(
	.param .u64 madd2_param_0,
	.param .u64 madd2_param_1,
	.param .f32 madd2_param_2,
	.param .u64 madd2_param_3,
	.param .f32 madd2_param_4,
	.param .u32 madd2_param_5
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<7>;
	.reg .s64 	%rd<11>;


	ld.param.u64 	%rd4, [madd2_param_0];
	ld.param.u64 	%rd5, [madd2_param_1];
	ld.param.f32 	%f1, [madd2_param_2];
	ld.param.u64 	%rd6, [madd2_param_3];
	ld.param.f32 	%f2, [madd2_param_4];
	ld.param.u32 	%r2, [madd2_param_5];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd6;
	cvta.to.global.u64 	%rd3, %rd5;
	.loc 1 8 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 10 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB2_2;

	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd3, %rd7;
	.loc 1 11 1
	ld.global.nc.f32 	%f3, [%rd8];
	add.s64 	%rd9, %rd2, %rd7;
	.loc 1 11 1
	ld.global.nc.f32 	%f4, [%rd9];
	mul.f32 	%f5, %f4, %f2;
	fma.rn.f32 	%f6, %f3, %f1, %f5;
	add.s64 	%rd10, %rd1, %rd7;
	.loc 1 11 1
	st.global.f32 	[%rd10], %f6;

BB2_2:
	.loc 1 13 2
	ret;
}


`
)
