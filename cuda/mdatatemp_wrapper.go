package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import(
	"unsafe"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
)

// CUDA handle for mdatatemp kernel
var mdatatemp_code cu.Function

// Stores the arguments for mdatatemp kernel invocation
type mdatatemp_args_t struct{
	 arg_dst_x unsafe.Pointer
	 arg_dst_y unsafe.Pointer
	 arg_dst_z unsafe.Pointer
	 arg_temp_dt unsafe.Pointer
	 arg_mx_temp unsafe.Pointer
	 arg_my_temp unsafe.Pointer
	 arg_mz_temp unsafe.Pointer
	 arg_dt float32
	 arg_N int
	 argptr [9]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for mdatatemp kernel invocation
var mdatatemp_args mdatatemp_args_t

func init(){
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	 mdatatemp_args.argptr[0] = unsafe.Pointer(&mdatatemp_args.arg_dst_x)
	 mdatatemp_args.argptr[1] = unsafe.Pointer(&mdatatemp_args.arg_dst_y)
	 mdatatemp_args.argptr[2] = unsafe.Pointer(&mdatatemp_args.arg_dst_z)
	 mdatatemp_args.argptr[3] = unsafe.Pointer(&mdatatemp_args.arg_temp_dt)
	 mdatatemp_args.argptr[4] = unsafe.Pointer(&mdatatemp_args.arg_mx_temp)
	 mdatatemp_args.argptr[5] = unsafe.Pointer(&mdatatemp_args.arg_my_temp)
	 mdatatemp_args.argptr[6] = unsafe.Pointer(&mdatatemp_args.arg_mz_temp)
	 mdatatemp_args.argptr[7] = unsafe.Pointer(&mdatatemp_args.arg_dt)
	 mdatatemp_args.argptr[8] = unsafe.Pointer(&mdatatemp_args.arg_N)
	 }

// Wrapper for mdatatemp CUDA kernel, asynchronous.
func k_mdatatemp_async ( dst_x unsafe.Pointer, dst_y unsafe.Pointer, dst_z unsafe.Pointer, temp_dt unsafe.Pointer, mx_temp unsafe.Pointer, my_temp unsafe.Pointer, mz_temp unsafe.Pointer, dt float32, N int,  cfg *config) {
	if Synchronous{ // debug
		Sync()
		timer.Start("mdatatemp")
	}

	mdatatemp_args.Lock()
	defer mdatatemp_args.Unlock()

	if mdatatemp_code == 0{
		mdatatemp_code = fatbinLoad(mdatatemp_map, "mdatatemp")
	}

	 mdatatemp_args.arg_dst_x = dst_x
	 mdatatemp_args.arg_dst_y = dst_y
	 mdatatemp_args.arg_dst_z = dst_z
	 mdatatemp_args.arg_temp_dt = temp_dt
	 mdatatemp_args.arg_mx_temp = mx_temp
	 mdatatemp_args.arg_my_temp = my_temp
	 mdatatemp_args.arg_mz_temp = mz_temp
	 mdatatemp_args.arg_dt = dt
	 mdatatemp_args.arg_N = N
	

	args := mdatatemp_args.argptr[:]
	cu.LaunchKernel(mdatatemp_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous{ // debug
		Sync()
		timer.Stop("mdatatemp")
	}
}

// maps compute capability on PTX code for mdatatemp kernel.
var mdatatemp_map = map[int]string{ 0: "" ,
30: mdatatemp_ptx_30 ,
35: mdatatemp_ptx_35 ,
37: mdatatemp_ptx_37 ,
50: mdatatemp_ptx_50 ,
52: mdatatemp_ptx_52 ,
53: mdatatemp_ptx_53 ,
60: mdatatemp_ptx_60 ,
61: mdatatemp_ptx_61 ,
70: mdatatemp_ptx_70 ,
75: mdatatemp_ptx_75  }

// mdatatemp PTX code for various compute capabilities.
const(
  mdatatemp_ptx_30 = `
.version 6.4
.target sm_30
.address_size 64

	// .globl	mdatatemp

.visible .entry mdatatemp(
	.param .u64 mdatatemp_param_0,
	.param .u64 mdatatemp_param_1,
	.param .u64 mdatatemp_param_2,
	.param .u64 mdatatemp_param_3,
	.param .u64 mdatatemp_param_4,
	.param .u64 mdatatemp_param_5,
	.param .u64 mdatatemp_param_6,
	.param .f32 mdatatemp_param_7,
	.param .u32 mdatatemp_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [mdatatemp_param_0];
	ld.param.u64 	%rd2, [mdatatemp_param_1];
	ld.param.u64 	%rd3, [mdatatemp_param_2];
	ld.param.u64 	%rd4, [mdatatemp_param_3];
	ld.param.u64 	%rd5, [mdatatemp_param_4];
	ld.param.u64 	%rd6, [mdatatemp_param_5];
	ld.param.u64 	%rd7, [mdatatemp_param_6];
	ld.param.f32 	%f1, [mdatatemp_param_7];
	ld.param.u32 	%r2, [mdatatemp_param_8];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd1;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f2, [%rd12];
	ld.global.f32 	%f3, [%rd10];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd12], %f4;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.f32 	%f5, [%rd16];
	ld.global.f32 	%f6, [%rd14];
	add.f32 	%f7, %f6, %f5;
	st.global.f32 	[%rd16], %f7;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.f32 	%f8, [%rd20];
	ld.global.f32 	%f9, [%rd18];
	add.f32 	%f10, %f9, %f8;
	st.global.f32 	[%rd20], %f10;
	cvta.to.global.u64 	%rd21, %rd4;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f1;

BB0_2:
	ret;
}


`
   mdatatemp_ptx_35 = `
.version 6.4
.target sm_35
.address_size 64

	// .globl	mdatatemp

.visible .entry mdatatemp(
	.param .u64 mdatatemp_param_0,
	.param .u64 mdatatemp_param_1,
	.param .u64 mdatatemp_param_2,
	.param .u64 mdatatemp_param_3,
	.param .u64 mdatatemp_param_4,
	.param .u64 mdatatemp_param_5,
	.param .u64 mdatatemp_param_6,
	.param .f32 mdatatemp_param_7,
	.param .u32 mdatatemp_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [mdatatemp_param_0];
	ld.param.u64 	%rd2, [mdatatemp_param_1];
	ld.param.u64 	%rd3, [mdatatemp_param_2];
	ld.param.u64 	%rd4, [mdatatemp_param_3];
	ld.param.u64 	%rd5, [mdatatemp_param_4];
	ld.param.u64 	%rd6, [mdatatemp_param_5];
	ld.param.u64 	%rd7, [mdatatemp_param_6];
	ld.param.f32 	%f1, [mdatatemp_param_7];
	ld.param.u32 	%r2, [mdatatemp_param_8];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd1;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f2, [%rd12];
	ld.global.nc.f32 	%f3, [%rd10];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd12], %f4;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.f32 	%f5, [%rd16];
	ld.global.nc.f32 	%f6, [%rd14];
	add.f32 	%f7, %f6, %f5;
	st.global.f32 	[%rd16], %f7;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.f32 	%f8, [%rd20];
	ld.global.nc.f32 	%f9, [%rd18];
	add.f32 	%f10, %f9, %f8;
	st.global.f32 	[%rd20], %f10;
	cvta.to.global.u64 	%rd21, %rd4;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f1;

BB0_2:
	ret;
}


`
   mdatatemp_ptx_37 = `
.version 6.4
.target sm_37
.address_size 64

	// .globl	mdatatemp

.visible .entry mdatatemp(
	.param .u64 mdatatemp_param_0,
	.param .u64 mdatatemp_param_1,
	.param .u64 mdatatemp_param_2,
	.param .u64 mdatatemp_param_3,
	.param .u64 mdatatemp_param_4,
	.param .u64 mdatatemp_param_5,
	.param .u64 mdatatemp_param_6,
	.param .f32 mdatatemp_param_7,
	.param .u32 mdatatemp_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [mdatatemp_param_0];
	ld.param.u64 	%rd2, [mdatatemp_param_1];
	ld.param.u64 	%rd3, [mdatatemp_param_2];
	ld.param.u64 	%rd4, [mdatatemp_param_3];
	ld.param.u64 	%rd5, [mdatatemp_param_4];
	ld.param.u64 	%rd6, [mdatatemp_param_5];
	ld.param.u64 	%rd7, [mdatatemp_param_6];
	ld.param.f32 	%f1, [mdatatemp_param_7];
	ld.param.u32 	%r2, [mdatatemp_param_8];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd1;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f2, [%rd12];
	ld.global.nc.f32 	%f3, [%rd10];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd12], %f4;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.f32 	%f5, [%rd16];
	ld.global.nc.f32 	%f6, [%rd14];
	add.f32 	%f7, %f6, %f5;
	st.global.f32 	[%rd16], %f7;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.f32 	%f8, [%rd20];
	ld.global.nc.f32 	%f9, [%rd18];
	add.f32 	%f10, %f9, %f8;
	st.global.f32 	[%rd20], %f10;
	cvta.to.global.u64 	%rd21, %rd4;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f1;

BB0_2:
	ret;
}


`
   mdatatemp_ptx_50 = `
.version 6.4
.target sm_50
.address_size 64

	// .globl	mdatatemp

.visible .entry mdatatemp(
	.param .u64 mdatatemp_param_0,
	.param .u64 mdatatemp_param_1,
	.param .u64 mdatatemp_param_2,
	.param .u64 mdatatemp_param_3,
	.param .u64 mdatatemp_param_4,
	.param .u64 mdatatemp_param_5,
	.param .u64 mdatatemp_param_6,
	.param .f32 mdatatemp_param_7,
	.param .u32 mdatatemp_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [mdatatemp_param_0];
	ld.param.u64 	%rd2, [mdatatemp_param_1];
	ld.param.u64 	%rd3, [mdatatemp_param_2];
	ld.param.u64 	%rd4, [mdatatemp_param_3];
	ld.param.u64 	%rd5, [mdatatemp_param_4];
	ld.param.u64 	%rd6, [mdatatemp_param_5];
	ld.param.u64 	%rd7, [mdatatemp_param_6];
	ld.param.f32 	%f1, [mdatatemp_param_7];
	ld.param.u32 	%r2, [mdatatemp_param_8];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd1;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f2, [%rd12];
	ld.global.nc.f32 	%f3, [%rd10];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd12], %f4;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.f32 	%f5, [%rd16];
	ld.global.nc.f32 	%f6, [%rd14];
	add.f32 	%f7, %f6, %f5;
	st.global.f32 	[%rd16], %f7;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.f32 	%f8, [%rd20];
	ld.global.nc.f32 	%f9, [%rd18];
	add.f32 	%f10, %f9, %f8;
	st.global.f32 	[%rd20], %f10;
	cvta.to.global.u64 	%rd21, %rd4;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f1;

BB0_2:
	ret;
}


`
   mdatatemp_ptx_52 = `
.version 6.4
.target sm_52
.address_size 64

	// .globl	mdatatemp

.visible .entry mdatatemp(
	.param .u64 mdatatemp_param_0,
	.param .u64 mdatatemp_param_1,
	.param .u64 mdatatemp_param_2,
	.param .u64 mdatatemp_param_3,
	.param .u64 mdatatemp_param_4,
	.param .u64 mdatatemp_param_5,
	.param .u64 mdatatemp_param_6,
	.param .f32 mdatatemp_param_7,
	.param .u32 mdatatemp_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [mdatatemp_param_0];
	ld.param.u64 	%rd2, [mdatatemp_param_1];
	ld.param.u64 	%rd3, [mdatatemp_param_2];
	ld.param.u64 	%rd4, [mdatatemp_param_3];
	ld.param.u64 	%rd5, [mdatatemp_param_4];
	ld.param.u64 	%rd6, [mdatatemp_param_5];
	ld.param.u64 	%rd7, [mdatatemp_param_6];
	ld.param.f32 	%f1, [mdatatemp_param_7];
	ld.param.u32 	%r2, [mdatatemp_param_8];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd1;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f2, [%rd12];
	ld.global.nc.f32 	%f3, [%rd10];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd12], %f4;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.f32 	%f5, [%rd16];
	ld.global.nc.f32 	%f6, [%rd14];
	add.f32 	%f7, %f6, %f5;
	st.global.f32 	[%rd16], %f7;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.f32 	%f8, [%rd20];
	ld.global.nc.f32 	%f9, [%rd18];
	add.f32 	%f10, %f9, %f8;
	st.global.f32 	[%rd20], %f10;
	cvta.to.global.u64 	%rd21, %rd4;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f1;

BB0_2:
	ret;
}


`
   mdatatemp_ptx_53 = `
.version 6.4
.target sm_53
.address_size 64

	// .globl	mdatatemp

.visible .entry mdatatemp(
	.param .u64 mdatatemp_param_0,
	.param .u64 mdatatemp_param_1,
	.param .u64 mdatatemp_param_2,
	.param .u64 mdatatemp_param_3,
	.param .u64 mdatatemp_param_4,
	.param .u64 mdatatemp_param_5,
	.param .u64 mdatatemp_param_6,
	.param .f32 mdatatemp_param_7,
	.param .u32 mdatatemp_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [mdatatemp_param_0];
	ld.param.u64 	%rd2, [mdatatemp_param_1];
	ld.param.u64 	%rd3, [mdatatemp_param_2];
	ld.param.u64 	%rd4, [mdatatemp_param_3];
	ld.param.u64 	%rd5, [mdatatemp_param_4];
	ld.param.u64 	%rd6, [mdatatemp_param_5];
	ld.param.u64 	%rd7, [mdatatemp_param_6];
	ld.param.f32 	%f1, [mdatatemp_param_7];
	ld.param.u32 	%r2, [mdatatemp_param_8];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd1;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f2, [%rd12];
	ld.global.nc.f32 	%f3, [%rd10];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd12], %f4;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.f32 	%f5, [%rd16];
	ld.global.nc.f32 	%f6, [%rd14];
	add.f32 	%f7, %f6, %f5;
	st.global.f32 	[%rd16], %f7;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.f32 	%f8, [%rd20];
	ld.global.nc.f32 	%f9, [%rd18];
	add.f32 	%f10, %f9, %f8;
	st.global.f32 	[%rd20], %f10;
	cvta.to.global.u64 	%rd21, %rd4;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f1;

BB0_2:
	ret;
}


`
   mdatatemp_ptx_60 = `
.version 6.4
.target sm_60
.address_size 64

	// .globl	mdatatemp

.visible .entry mdatatemp(
	.param .u64 mdatatemp_param_0,
	.param .u64 mdatatemp_param_1,
	.param .u64 mdatatemp_param_2,
	.param .u64 mdatatemp_param_3,
	.param .u64 mdatatemp_param_4,
	.param .u64 mdatatemp_param_5,
	.param .u64 mdatatemp_param_6,
	.param .f32 mdatatemp_param_7,
	.param .u32 mdatatemp_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [mdatatemp_param_0];
	ld.param.u64 	%rd2, [mdatatemp_param_1];
	ld.param.u64 	%rd3, [mdatatemp_param_2];
	ld.param.u64 	%rd4, [mdatatemp_param_3];
	ld.param.u64 	%rd5, [mdatatemp_param_4];
	ld.param.u64 	%rd6, [mdatatemp_param_5];
	ld.param.u64 	%rd7, [mdatatemp_param_6];
	ld.param.f32 	%f1, [mdatatemp_param_7];
	ld.param.u32 	%r2, [mdatatemp_param_8];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd1;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f2, [%rd12];
	ld.global.nc.f32 	%f3, [%rd10];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd12], %f4;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.f32 	%f5, [%rd16];
	ld.global.nc.f32 	%f6, [%rd14];
	add.f32 	%f7, %f6, %f5;
	st.global.f32 	[%rd16], %f7;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.f32 	%f8, [%rd20];
	ld.global.nc.f32 	%f9, [%rd18];
	add.f32 	%f10, %f9, %f8;
	st.global.f32 	[%rd20], %f10;
	cvta.to.global.u64 	%rd21, %rd4;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f1;

BB0_2:
	ret;
}


`
   mdatatemp_ptx_61 = `
.version 6.4
.target sm_61
.address_size 64

	// .globl	mdatatemp

.visible .entry mdatatemp(
	.param .u64 mdatatemp_param_0,
	.param .u64 mdatatemp_param_1,
	.param .u64 mdatatemp_param_2,
	.param .u64 mdatatemp_param_3,
	.param .u64 mdatatemp_param_4,
	.param .u64 mdatatemp_param_5,
	.param .u64 mdatatemp_param_6,
	.param .f32 mdatatemp_param_7,
	.param .u32 mdatatemp_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [mdatatemp_param_0];
	ld.param.u64 	%rd2, [mdatatemp_param_1];
	ld.param.u64 	%rd3, [mdatatemp_param_2];
	ld.param.u64 	%rd4, [mdatatemp_param_3];
	ld.param.u64 	%rd5, [mdatatemp_param_4];
	ld.param.u64 	%rd6, [mdatatemp_param_5];
	ld.param.u64 	%rd7, [mdatatemp_param_6];
	ld.param.f32 	%f1, [mdatatemp_param_7];
	ld.param.u32 	%r2, [mdatatemp_param_8];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd1;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f2, [%rd12];
	ld.global.nc.f32 	%f3, [%rd10];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd12], %f4;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.f32 	%f5, [%rd16];
	ld.global.nc.f32 	%f6, [%rd14];
	add.f32 	%f7, %f6, %f5;
	st.global.f32 	[%rd16], %f7;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.f32 	%f8, [%rd20];
	ld.global.nc.f32 	%f9, [%rd18];
	add.f32 	%f10, %f9, %f8;
	st.global.f32 	[%rd20], %f10;
	cvta.to.global.u64 	%rd21, %rd4;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f1;

BB0_2:
	ret;
}


`
   mdatatemp_ptx_70 = `
.version 6.4
.target sm_70
.address_size 64

	// .globl	mdatatemp

.visible .entry mdatatemp(
	.param .u64 mdatatemp_param_0,
	.param .u64 mdatatemp_param_1,
	.param .u64 mdatatemp_param_2,
	.param .u64 mdatatemp_param_3,
	.param .u64 mdatatemp_param_4,
	.param .u64 mdatatemp_param_5,
	.param .u64 mdatatemp_param_6,
	.param .f32 mdatatemp_param_7,
	.param .u32 mdatatemp_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [mdatatemp_param_0];
	ld.param.u64 	%rd2, [mdatatemp_param_1];
	ld.param.u64 	%rd3, [mdatatemp_param_2];
	ld.param.u64 	%rd4, [mdatatemp_param_3];
	ld.param.u64 	%rd5, [mdatatemp_param_4];
	ld.param.u64 	%rd6, [mdatatemp_param_5];
	ld.param.u64 	%rd7, [mdatatemp_param_6];
	ld.param.f32 	%f1, [mdatatemp_param_7];
	ld.param.u32 	%r2, [mdatatemp_param_8];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd1;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f2, [%rd12];
	ld.global.nc.f32 	%f3, [%rd10];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd12], %f4;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.f32 	%f5, [%rd16];
	ld.global.nc.f32 	%f6, [%rd14];
	add.f32 	%f7, %f6, %f5;
	st.global.f32 	[%rd16], %f7;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.f32 	%f8, [%rd20];
	ld.global.nc.f32 	%f9, [%rd18];
	add.f32 	%f10, %f9, %f8;
	st.global.f32 	[%rd20], %f10;
	cvta.to.global.u64 	%rd21, %rd4;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f1;

BB0_2:
	ret;
}


`
   mdatatemp_ptx_75 = `
.version 6.4
.target sm_75
.address_size 64

	// .globl	mdatatemp

.visible .entry mdatatemp(
	.param .u64 mdatatemp_param_0,
	.param .u64 mdatatemp_param_1,
	.param .u64 mdatatemp_param_2,
	.param .u64 mdatatemp_param_3,
	.param .u64 mdatatemp_param_4,
	.param .u64 mdatatemp_param_5,
	.param .u64 mdatatemp_param_6,
	.param .f32 mdatatemp_param_7,
	.param .u32 mdatatemp_param_8
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [mdatatemp_param_0];
	ld.param.u64 	%rd2, [mdatatemp_param_1];
	ld.param.u64 	%rd3, [mdatatemp_param_2];
	ld.param.u64 	%rd4, [mdatatemp_param_3];
	ld.param.u64 	%rd5, [mdatatemp_param_4];
	ld.param.u64 	%rd6, [mdatatemp_param_5];
	ld.param.u64 	%rd7, [mdatatemp_param_6];
	ld.param.f32 	%f1, [mdatatemp_param_7];
	ld.param.u32 	%r2, [mdatatemp_param_8];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd1;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f2, [%rd12];
	ld.global.nc.f32 	%f3, [%rd10];
	add.f32 	%f4, %f3, %f2;
	st.global.f32 	[%rd12], %f4;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.f32 	%f5, [%rd16];
	ld.global.nc.f32 	%f6, [%rd14];
	add.f32 	%f7, %f6, %f5;
	st.global.f32 	[%rd16], %f7;
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.f32 	%f8, [%rd20];
	ld.global.nc.f32 	%f9, [%rd18];
	add.f32 	%f10, %f9, %f8;
	st.global.f32 	[%rd20], %f10;
	cvta.to.global.u64 	%rd21, %rd4;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f1;

BB0_2:
	ret;
}


`
 )
