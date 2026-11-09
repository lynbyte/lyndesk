/*
 * Thin C shim over libx264 for phantom's optional `x264` encoder feature.
 *
 * Everything that touches x264's structs (whose layout depends on X264_BUILD)
 * stays in C, compiled against the installed x264.h; Rust only sees four
 * functions with plain arguments. See crates/server/src/encode/x264.rs.
 *
 * x264 is GPL-2.0-or-later: a binary built with this shim is GPL.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <x264.h>

typedef struct phx264 {
    x264_t *h;
    x264_param_t p;
    x264_picture_t pic;
    int64_t pts;
} phx264;

/* VBV buffer of 1.25 frames at the cap, as voxl uses: every frame stays near
 * one frame's budget (x264 raises QP within the frame), so a full-screen
 * change can't come out as a 400 KB frame that takes 300 ms to send. Below
 * ~1.1 frames x264 switches to single-frame VBV handling. */
static int vbv_kbit(int max_kbps, int fps) {
    if (fps < 1) fps = 1;
    return (max_kbps * 5 + 4 * fps - 1) / (4 * fps);
}

phx264 *phx264_open(int width, int height, int fps, int threads, int max_kbps,
                    float crf, const char *preset) {
    phx264 *e = calloc(1, sizeof *e);
    if (!e) return NULL;
    x264_param_t *p = &e->p;
    if (x264_param_default_preset(p, preset, "zerolatency") < 0) goto fail;
    p->i_width = width;
    p->i_height = height;
    p->i_csp = X264_CSP_I420;
    p->i_fps_num = fps > 0 ? fps : 30;
    p->i_fps_den = 1;
    p->b_vfr_input = 0;
    p->i_threads = threads > 0 ? threads : 1;
    p->b_sliced_threads = threads > 1;
    p->i_slice_count = threads > 1 ? threads : 0;
    /* Keyframes only on request (phantom forces them); never on scene cuts. */
    p->i_keyint_max = X264_KEYINT_MAX_INFINITE;
    p->i_keyint_min = 1;
    p->i_scenecut_threshold = 0;
    p->b_intra_refresh = 0;
    p->i_bframe = 0;
    p->b_annexb = 1;
    p->b_repeat_headers = 1;
    p->b_aud = 0;
    p->i_log_level = X264_LOG_WARNING;
    p->rc.i_rc_method = X264_RC_CRF;
    p->rc.f_rf_constant = crf;
    p->rc.i_lookahead = 0;
    p->rc.b_mb_tree = 0;
    p->i_sync_lookahead = 0;
    /* ultrafast's diamond search only follows motion of a few pixels: a
     * scroll or a dragged window would be coded as new pixels every frame.
     * UMH finds those shifts for about the same CPU (voxl measured a third of
     * the bits on a window drag). */
    p->analyse.i_me_method = X264_ME_UMH;
    p->analyse.i_me_range = 64;
    if (max_kbps > 0) {
        p->rc.i_vbv_max_bitrate = max_kbps;
        p->rc.i_vbv_buffer_size = vbv_kbit(max_kbps, p->i_fps_num);
    }
    /* Constrained baseline: the native client and the benchmark decode with
     * OpenH264, which has no CABAC/High support, and the web client
     * configures WebCodecs with a baseline codec string. */
    if (x264_param_apply_profile(p, "baseline") < 0) goto fail;
    e->h = x264_encoder_open(p);
    if (!e->h) goto fail;
    x264_picture_init(&e->pic);
    e->pic.img.i_csp = X264_CSP_I420;
    e->pic.img.i_plane = 3;
    return e;
fail:
    free(e);
    return NULL;
}

/* Encode one I420 picture. Returns the byte size of the Annex-B output in
 * *out (valid until the next call), 0 if x264 produced nothing, <0 on error. */
int phx264_encode(phx264 *e, const uint8_t *y, const uint8_t *u, const uint8_t *v,
                  int stride_y, int stride_uv, int force_idr, uint8_t **out,
                  int *keyframe) {
    x264_nal_t *nals = NULL;
    int n = 0;
    x264_picture_t pic_out;
    e->pic.img.plane[0] = (uint8_t *)y;
    e->pic.img.plane[1] = (uint8_t *)u;
    e->pic.img.plane[2] = (uint8_t *)v;
    e->pic.img.i_stride[0] = stride_y;
    e->pic.img.i_stride[1] = stride_uv;
    e->pic.img.i_stride[2] = stride_uv;
    e->pic.i_type = force_idr ? X264_TYPE_IDR : X264_TYPE_AUTO;
    e->pic.i_pts = e->pts++;
    int size = x264_encoder_encode(e->h, &nals, &n, &e->pic, &pic_out);
    if (size < 0) return size;
    *keyframe = pic_out.b_keyframe;
    /* x264 guarantees the payloads of one frame are contiguous. */
    *out = size > 0 ? nals[0].p_payload : NULL;
    return size;
}

/* Change the VBV cap (kbit/s) in place: no new stream, no IDR. */
int phx264_set_max_kbps(phx264 *e, int max_kbps) {
    e->p.rc.i_vbv_max_bitrate = max_kbps;
    e->p.rc.i_vbv_buffer_size = vbv_kbit(max_kbps, e->p.i_fps_num);
    return x264_encoder_reconfig(e->h, &e->p);
}

void phx264_close(phx264 *e) {
    if (!e) return;
    if (e->h) x264_encoder_close(e->h);
    free(e);
}
