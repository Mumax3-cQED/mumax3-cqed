package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var madd3_code cu.Function

type madd3_args struct {
	arg_dst  unsafe.Pointer
	arg_src1 unsafe.Pointer
	arg_fac1 float32
	arg_src2 unsafe.Pointer
	arg_fac2 float32
	arg_src3 unsafe.Pointer
	arg_fac3 float32
	arg_N    int
	argptr   [8]unsafe.Pointer
}

// Wrapper for madd3 CUDA kernel, asynchronous.
func k_madd3_async(dst unsafe.Pointer, src1 unsafe.Pointer, fac1 float32, src2 unsafe.Pointer, fac2 float32, src3 unsafe.Pointer, fac3 float32, N int, cfg *config) {
	if synchronous { // debug
		Sync()
	}

	if madd3_code == 0 {
		madd3_code = fatbinLoad(madd3_map, "madd3")
	}

	var _a_ madd3_args

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
	_a_.arg_src3 = src3
	_a_.argptr[5] = unsafe.Pointer(&_a_.arg_src3)
	_a_.arg_fac3 = fac3
	_a_.argptr[6] = unsafe.Pointer(&_a_.arg_fac3)
	_a_.arg_N = N
	_a_.argptr[7] = unsafe.Pointer(&_a_.arg_N)

	args := _a_.argptr[:]
	cu.LaunchKernel(madd3_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if synchronous { // debug
		Sync()
	}
}

var madd3_map = map[int]string{0: "",
	20: madd3_ptx_20,
	30: madd3_ptx_30,
	35: madd3_ptx_35}

const (
	madd3_ptx_20 = `
.version 3.2
.target sm_20
.address_size 64


.visible .entry madd3(
	.param .u64 madd3_param_0,
	.param .u64 madd3_param_1,
	.param .f32 madd3_param_2,
	.param .u64 madd3_param_3,
	.param .f32 madd3_param_4,
	.param .u64 madd3_param_5,
	.param .f32 madd3_param_6,
	.param .u32 madd3_param_7
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<14>;


	ld.param.u64 	%rd5, [madd3_param_0];
	ld.param.u64 	%rd6, [madd3_param_1];
	ld.param.f32 	%f1, [madd3_param_2];
	ld.param.u64 	%rd7, [madd3_param_3];
	ld.param.f32 	%f2, [madd3_param_4];
	ld.param.u64 	%rd8, [madd3_param_5];
	ld.param.f32 	%f3, [madd3_param_6];
	ld.param.u32 	%r2, [madd3_param_7];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd8;
	cvta.to.global.u64 	%rd3, %rd7;
	cvta.to.global.u64 	%rd4, %rd6;
	.loc 1 9 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 11 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd4, %rd9;
	.loc 1 12 1
	ld.global.f32 	%f4, [%rd10];
	add.s64 	%rd11, %rd3, %rd9;
	.loc 1 12 1
	ld.global.f32 	%f5, [%rd11];
	add.s64 	%rd12, %rd2, %rd9;
	.loc 1 12 1
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f6, %f3;
	fma.rn.f32 	%f8, %f5, %f2, %f7;
	fma.rn.f32 	%f9, %f4, %f1, %f8;
	add.s64 	%rd13, %rd1, %rd9;
	.loc 1 12 1
	st.global.f32 	[%rd13], %f9;

BB0_2:
	.loc 1 15 2
	ret;
}


`
	madd3_ptx_30 = `
.version 3.2
.target sm_30
.address_size 64


.visible .entry madd3(
	.param .u64 madd3_param_0,
	.param .u64 madd3_param_1,
	.param .f32 madd3_param_2,
	.param .u64 madd3_param_3,
	.param .f32 madd3_param_4,
	.param .u64 madd3_param_5,
	.param .f32 madd3_param_6,
	.param .u32 madd3_param_7
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<14>;


	ld.param.u64 	%rd5, [madd3_param_0];
	ld.param.u64 	%rd6, [madd3_param_1];
	ld.param.f32 	%f1, [madd3_param_2];
	ld.param.u64 	%rd7, [madd3_param_3];
	ld.param.f32 	%f2, [madd3_param_4];
	ld.param.u64 	%rd8, [madd3_param_5];
	ld.param.f32 	%f3, [madd3_param_6];
	ld.param.u32 	%r2, [madd3_param_7];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd8;
	cvta.to.global.u64 	%rd3, %rd7;
	cvta.to.global.u64 	%rd4, %rd6;
	.loc 1 9 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 11 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd4, %rd9;
	.loc 1 12 1
	ld.global.f32 	%f4, [%rd10];
	add.s64 	%rd11, %rd3, %rd9;
	.loc 1 12 1
	ld.global.f32 	%f5, [%rd11];
	add.s64 	%rd12, %rd2, %rd9;
	.loc 1 12 1
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f6, %f3;
	fma.rn.f32 	%f8, %f5, %f2, %f7;
	fma.rn.f32 	%f9, %f4, %f1, %f8;
	add.s64 	%rd13, %rd1, %rd9;
	.loc 1 12 1
	st.global.f32 	[%rd13], %f9;

BB0_2:
	.loc 1 15 2
	ret;
}


`
	madd3_ptx_35 = `
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

.visible .entry madd3(
	.param .u64 madd3_param_0,
	.param .u64 madd3_param_1,
	.param .f32 madd3_param_2,
	.param .u64 madd3_param_3,
	.param .f32 madd3_param_4,
	.param .u64 madd3_param_5,
	.param .f32 madd3_param_6,
	.param .u32 madd3_param_7
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<14>;


	ld.param.u64 	%rd5, [madd3_param_0];
	ld.param.u64 	%rd6, [madd3_param_1];
	ld.param.f32 	%f1, [madd3_param_2];
	ld.param.u64 	%rd7, [madd3_param_3];
	ld.param.f32 	%f2, [madd3_param_4];
	ld.param.u64 	%rd8, [madd3_param_5];
	ld.param.f32 	%f3, [madd3_param_6];
	ld.param.u32 	%r2, [madd3_param_7];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd8;
	cvta.to.global.u64 	%rd3, %rd7;
	cvta.to.global.u64 	%rd4, %rd6;
	.loc 1 9 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 11 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB2_2;

	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd4, %rd9;
	.loc 1 12 1
	ld.global.nc.f32 	%f4, [%rd10];
	add.s64 	%rd11, %rd3, %rd9;
	.loc 1 12 1
	ld.global.nc.f32 	%f5, [%rd11];
	add.s64 	%rd12, %rd2, %rd9;
	.loc 1 12 1
	ld.global.nc.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f6, %f3;
	fma.rn.f32 	%f8, %f5, %f2, %f7;
	fma.rn.f32 	%f9, %f4, %f1, %f8;
	add.s64 	%rd13, %rd1, %rd9;
	.loc 1 12 1
	st.global.f32 	[%rd13], %f9;

BB2_2:
	.loc 1 15 2
	ret;
}


`
)
