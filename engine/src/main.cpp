// SGR A* engine entry point.
//
//   sgr_engine [--frames N] [--screenshot out.png] [--shader solid|geodesic]
//              [--validation] [--width W] [--height H] [--astar X]
//              [--incl DEG] [--exposure EV]
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
        else if (!std::strcmp(argv[i], "--rcam")) opts.r_cam = static_cast<float>(std::atof(next("--rcam")));
        else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    try {
        sgr::App app(opts);
        return app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 3;
    }
}
