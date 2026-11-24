//! `a16` image samples run on the host GPU and WARP: which mip level a
//! translated `image_sample_l`, `_d` and `_b` with 16-bit addresses picks.
//!
//! Each lane loads its address dwords from a raw buffer (so the packed
//! halves, and junk in the halves the hardware ignores, are set bit for
//! bit), samples a 16x4 texture whose mip `m` holds `m` everywhere, and
//! stores the red channel. The texture is not square, so a gradient read
//! as the wrong axis picks a level two off.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 4;
/// 16x4, 8x2, 4x1, 2x1, 1x1.
const LEVELS: u32 = 5;
/// Address dwords per lane in the buffer.
const PER_LANE: usize = 8;

/// A wave32 program of 32 threads: the table at s0 holds the texture
/// (entry 0), the lanes' address dwords (entry 1, raw) and the output
/// (entry 2, raw UAV); the sampler table is at s1. Lane `l` loads dwords
/// `8l..8l + 8` into v4..v11, runs `sample` on v4.. and stores the red
/// channel at dword `l`.
fn program(sample: [u32; 2]) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0502,
        0xFA00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x3422_0085, // v_lshlrev_b32 v17, 5, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE038_1000,
        0x8005_0411, // buffer_load_dwordx4 v[4:7], v17, s[20:23], 0 offen
        0xE038_1010,
        0x8005_0811, // buffer_load_dwordx4 v[8:11], v17, s[20:23], 0 offen offset:16
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
    ];
    w.extend(sample);
    w.extend([
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

/// `<op> v12, v4.., s[12:19], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D a16`.
fn sample_a16(op: u32) -> [u32; 2] {
    [
        0xF000_0000 | op << 18 | 1 << 8 | 1 << 3,
        1 << 30 | 2 << 21 | 3 << 16 | 12 << 8 | 4,
    ]
}

fn build(words: &[u32]) -> HostShader {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0),
            range(ResourceType::Srv, 8, 4, 0),
            range(ResourceType::Uav, 16, 4, 0),
            range(ResourceType::Sampler, 0, 4, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `shader` over `lanes` (32 lanes of [`PER_LANE`] dwords) on the
/// hardware GPU and on WARP (each skipped without a device) and returns,
/// per one that ran, the level each lane sampled.
fn levels(shader: &HostShader, lanes: &[u32]) -> Vec<(&'static str, Vec<f32>)> {
    assert_eq!(lanes.len(), 32 * PER_LANE);
    let views = [
        View::Mips(WIDTH, HEIGHT, LEVELS),
        View::Buffer {
            data: lanes,
            stride: Stride::Raw,
        },
        View::RawUav(32),
    ];
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut out = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // Both tables start at heap index 0.
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].iter().map(|&w| f32::from_bits(w)).collect()));
    }
    assert!(!out.is_empty(), "no device ran the test");
    out
}

/// The f16 bits of `x`, which must be zero or a normal half exactly.
fn h(x: f32) -> u32 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let exp = (b >> 23 & 0xFF) as i32 - 127 + 15;
    assert!(
        (1..31).contains(&exp) && b & 0x1FFF == 0,
        "{x} is no exact half"
    );
    (b >> 31) << 15 | (exp as u32) << 10 | (b >> 13 & 0x3FF)
}

/// Two halves in one dword, `lo` first.
fn pack(lo: u32, hi: u32) -> u32 {
    lo | hi << 16
}

/// Junk for the high half of a dword whose low half alone is read.
const JUNK: u32 = 0x5A5A;

#[test]
fn a16_sample_l_reads_the_level_from_the_low_half() {
    // v4 = (u, v), v5 = (lod, junk).
    let mut lanes = vec![0u32; 32 * PER_LANE];
    for l in 0..32 {
        let d = &mut lanes[l * PER_LANE..];
        d[0] = pack(h(0.5), h(0.5));
        d[1] = pack(h((l % 5) as f32), JUNK);
    }
    let shader = build(&program(sample_a16(0x24)));
    for (label, got) in levels(&shader, &lanes) {
        let want: Vec<f32> = (0..32).map(|l| (l % 5) as f32).collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn a16_sample_d_reads_32_bit_gradients_then_packed_coordinates() {
    // v4..v7 = du/dx, dv/dx, du/dy, dv/dy as f32; v8 = (u, v). Lane l puts
    // a gradient for level l % 5 in component (l / 5) % 4, zero elsewhere.
    let mut lanes = vec![0u32; 32 * PER_LANE];
    for l in 0..32 {
        let (k, c) = (l % 5, (l / 5) % 4);
        // u spans 16 texels, v 4.
        let size = if c % 2 == 0 { WIDTH } else { HEIGHT };
        let d = &mut lanes[l * PER_LANE..];
        d[c] = ((1 << k) as f32 / size as f32).to_bits();
        d[4] = pack(h(0.5), h(0.5));
    }
    let shader = build(&program(sample_a16(0x22)));
    for (label, got) in levels(&shader, &lanes) {
        let want: Vec<f32> = (0..32).map(|l| (l % 5) as f32).collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn a16_sample_b_reads_an_f16_bias_from_the_low_half() {
    // v4 = (bias, junk), v5 = (u, v). Lanes of a quad (2x2: lane bit 0 is
    // x, bit 1 is y) sit one level-`base` texel step apart, so the implicit
    // derivatives give level `base`; the bias moves it.
    const QUADS: [(i32, f32); 8] = [
        (0, 0.0),
        (0, 1.0),
        (0, 2.0),
        (0, 3.0),
        (2, -1.0),
        (2, -2.0),
        (1, 2.0),
        (0, 4.0),
    ];
    let mut lanes = vec![0u32; 32 * PER_LANE];
    for l in 0..32 {
        let (base, bias) = QUADS[l / 4];
        let step = (1 << base) as f32;
        let u = (l & 1) as f32 * step / WIDTH as f32;
        let v = (l >> 1 & 1) as f32 * step / HEIGHT as f32;
        let d = &mut lanes[l * PER_LANE..];
        d[0] = pack(h(bias), JUNK);
        d[1] = pack(h(u), h(v));
    }
    let shader = build(&program(sample_a16(0x25)));
    for (label, got) in levels(&shader, &lanes) {
        let want: Vec<f32> = (0..32)
            .map(|l| {
                let (base, bias) = QUADS[l / 4];
                (base as f32 + bias).clamp(0.0, (LEVELS - 1) as f32)
            })
            .collect();
        assert_eq!(got, want, "{label}");
    }
}
