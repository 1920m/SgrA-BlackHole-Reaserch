// SGR A* engine entry point.
//
//   sgr_engine [--frames N] [--screenshot out.png] [--shader solid|geodesic]
//              [--validation] [--width W] [--height H] [--astar X]
//              [--incl DEG] [--exposure EV]
//              [--display linear|agx|aces] [--spp N] [--bloom S] [--grid]
//              [--observer zamo|orbit|infall|hover|ride] [--ergo] [--aim X Y]
//              [--flare] [--flare-r R] [--flare-t T] [--movie N --movie-dt DT]
//
// Exit code = number of Vulkan validation errors (0 required by the gates).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include "app.hpp"

int main(int argc, char** argv) {
    sgr::AppOptions opts;
#ifndef NDEBUG
    opts.validation = true; // layers ON in every Debug build
#endif
    if (argc == 1) opts.menu = true; // plain launch boots to the menu (Gate 9)
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--frames")) opts.max_frames = std::atoi(next("--frames"));
        else if (!std::strcmp(argv[i], "--screenshot")) opts.screenshot_path = next("--screenshot");
        else if (!std::strcmp(argv[i], "--shader")) opts.shader = next("--shader");
        else if (!std::strcmp(argv[i], "--validation")) opts.validation = true;
        else if (!std::strcmp(argv[i], "--no-validation")) opts.validation = false;
        else if (!std::strcmp(argv[i], "--width")) opts.width = std::atoi(next("--width"));
        else if (!std::strcmp(argv[i], "--height")) opts.height = std::atoi(next("--height"));
        else if (!std::strcmp(argv[i], "--astar")) opts.a_star = static_cast<float>(std::atof(next("--astar")));
        else if (!std::strcmp(argv[i], "--incl")) opts.incl_deg = static_cast<float>(std::atof(next("--incl")));
        else if (!std::strcmp(argv[i], "--exposure")) opts.exposure_ev = static_cast<float>(std::atof(next("--exposure")));
        else if (!std::strcmp(argv[i], "--diagnostic")) opts.shade_mode = 1;
        else if (!std::strcmp(argv[i], "--offscreen")) opts.offscreen = true;
        else if (!std::strcmp(argv[i], "--hud")) opts.hud = true;
        else if (!std::strcmp(argv[i], "--menu")) opts.menu = true;
        else if (!std::strcmp(argv[i], "--demo")) opts.demo_frames = std::atoi(next("--demo"));
        else if (!std::strcmp(argv[i], "--panel")) opts.panel = next("--panel");
        else if (!std::strcmp(argv[i], "--no-mouse")) opts.no_mouse = true;
        else if (!std::strcmp(argv[i], "--rot")) opts.rot_speed = static_cast<float>(std::atof(next("--rot")));
        else if (!std::strcmp(argv[i], "--science")) opts.science = true;
        else if (!std::strcmp(argv[i], "--preset")) opts.preset = std::atoi(next("--preset"));
        else if (!std::strcmp(argv[i], "--scale")) opts.render_scale = static_cast<float>(std::atof(next("--scale")));
        else if (!std::strcmp(argv[i], "--fov")) opts.fov = static_cast<float>(std::atof(next("--fov")));
        else if (!std::strcmp(argv[i], "--tpeak")) opts.t_peak_k = static_cast<float>(std::atof(next("--tpeak")));
        else if (!std::strcmp(argv[i], "--rcam")) opts.r_cam = static_cast<float>(std::atof(next("--rcam")));
        else if (!std::strcmp(argv[i], "--rout")) opts.disk_r_out = static_cast<float>(std::atof(next("--rout")));
        else if (!std::strcmp(argv[i], "--turb")) opts.turb = static_cast<float>(std::atof(next("--turb")));
        else if (!std::strcmp(argv[i], "--blacksky")) opts.black_sky = true;
        else if (!std::strcmp(argv[i], "--bolometric")) opts.bolometric = true;
        else if (!std::strcmp(argv[i], "--grid")) opts.grid = true;
        else if (!std::strcmp(argv[i], "--sky")) opts.sky_ev = static_cast<float>(std::atof(next("--sky")));
        else if (!std::strcmp(argv[i], "--procedural-sky")) opts.procedural_sky = true;
        else if (!std::strcmp(argv[i], "--uniform-sky")) opts.uniform_sky = true;
        else if (!std::strcmp(argv[i], "--no-disk")) opts.no_disk = true;
        else if (!std::strcmp(argv[i], "--observer")) {
            const char* o = next("--observer");
            opts.observer = !std::strcmp(o, "orbit") ? 1 : !std::strcmp(o, "infall") ? 2
                          : !std::strcmp(o, "hover") ? 3 : !std::strcmp(o, "ride") ? 4 : 0;
        }
        else if (!std::strcmp(argv[i], "--ergo")) opts.ergo = true;
        else if (!std::strcmp(argv[i], "--aim")) {
            opts.aim_x = static_cast<float>(std::atof(next("--aim")));
            opts.aim_y = static_cast<float>(std::atof(next("--aim")));
        }
        else if (!std::strcmp(argv[i], "--display")) opts.display = next("--display");
        else if (!std::strcmp(argv[i], "--spp")) opts.spp = std::atoi(next("--spp"));
        else if (!std::strcmp(argv[i], "--bloom")) opts.bloom = static_cast<float>(std::atof(next("--bloom")));
        else if (!std::strcmp(argv[i], "--stars")) opts.show_stars = true;
        else if (!std::strcmp(argv[i], "--nopaths")) opts.star_paths = false;
        else if (!std::strcmp(argv[i], "--starscale")) opts.star_scale = static_cast<float>(std::atof(next("--starscale")));
        else if (!std::strcmp(argv[i], "--focus")) opts.focus_star = next("--focus");
        else if (!std::strcmp(argv[i], "--ride")) opts.focus_ride = true;
        else if (!std::strcmp(argv[i], "--epoch")) opts.epoch_yr = std::atof(next("--epoch"));
        else if (!std::strcmp(argv[i], "--flare")) opts.flare = true;
        else if (!std::strcmp(argv[i], "--flare-r")) {
            opts.flare = true;
            opts.flare_r = static_cast<float>(std::atof(next("--flare-r")));
        }
        else if (!std::strcmp(argv[i], "--flare-t")) opts.flare_t = static_cast<float>(std::atof(next("--flare-t")));
        else if (!std::strcmp(argv[i], "--flare-gray") || !std::strcmp(argv[i], "--gray")) opts.flare_gray = true;
        else if (!std::strcmp(argv[i], "--band")) opts.band = next("--band");
        else if (!std::strcmp(argv[i], "--wormhole")) opts.wormhole = true;
        else if (!std::strcmp(argv[i], "--flight")) opts.flight = true;
        else if (!std::strcmp(argv[i], "--no-sky-shift")) opts.no_sky_shift = true;
        else if (!std::strcmp(argv[i], "--look")) opts.look = next("--look");
        else if (!std::strcmp(argv[i], "--flight-frames")) opts.flight_frames = std::atoi(next("--flight-frames"));
        else if (!std::strcmp(argv[i], "--flight-r")) opts.flight_r = static_cast<float>(std::atof(next("--flight-r")));
        else if (!std::strcmp(argv[i], "--flight-r0")) opts.flight_r0 = static_cast<float>(std::atof(next("--flight-r0")));
        else if (!std::strcmp(argv[i], "--flight-theta")) opts.flight_theta = static_cast<float>(std::atof(next("--flight-theta")));
        else if (!std::strcmp(argv[i], "--flight-rest")) opts.flight_rest = true;
        else if (!std::strcmp(argv[i], "--wh-l")) opts.wh_l = static_cast<float>(std::atof(next("--wh-l")));
        else if (!std::strcmp(argv[i], "--flare-amp")) opts.flare_amp = static_cast<float>(std::atof(next("--flare-amp")));
        else if (!std::strcmp(argv[i], "--movie")) opts.movie_frames = std::atoi(next("--movie"));
        else if (!std::strcmp(argv[i], "--max-steps")) opts.max_steps = std::atoi(next("--max-steps"));
        else if (!std::strcmp(argv[i], "--movie-dt")) opts.movie_dt = static_cast<float>(std::atof(next("--movie-dt")));
        else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (opts.menu) {
        // The menu implies the live disk pipeline + ImGui shell.
        opts.shader = "disk";
        opts.hud = true;
    }
    try {
        sgr::App app(opts);
        return app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 3;
    }
}
