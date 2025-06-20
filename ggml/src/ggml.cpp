#include "ggml-impl.h"

#include <cstdlib>
#include <exception>

// Use an anonymous namespace to keep the implementation private to this file.
namespace {

/**
 * @brief An RAII class to safely install and uninstall a custom terminate handler.
 *
 * A single static instance of this class is created at library load time.
 * Its constructor sets a custom terminate handler to print a backtrace.
 * Its destructor, called automatically on library unload, restores the original handler.
 * This prevents dangling function pointers if the library is dynamically unloaded and reloaded.
 */
class GgmlTerminateHandler {
public:
    GgmlTerminateHandler() {
        // Do not install the handler if the user has opted out.
        if (getenv("GGML_NO_BACKTRACE")) {
            return;
        }

        // Store the existing handler.
        s_previous_handler = std::get_terminate();

        // If our handler is already installed, do nothing. This prevents issues
        // on library reload where the previous instance might not have been
        // destructed yet in some complex scenarios.
        if (s_previous_handler == custom_terminate_handler) {
            return;
        }

        std::set_terminate(custom_terminate_handler);
    }

    ~GgmlTerminateHandler() {
        // Restore the previous handler only if our handler is still active.
        // This is a safeguard in case another part of the program has
        // replaced the terminate handler after us.
        if (std::get_terminate() == custom_terminate_handler) {
            std::set_terminate(s_previous_handler);
        }
    }

private:
    // The custom handler that will be called on std::terminate().
    GGML_NORETURN static void custom_terminate_handler() {
        ggml_print_backtrace();

        if (s_previous_handler) {
            s_previous_handler();
        }

        abort(); // Fallback in case the previous handler returns.
    }

    // The previously registered terminate handler.
    // Must be static to be accessible from the static custom_terminate_handler.
    static std::terminate_handler s_previous_handler;
};

// Define the static member variable.
std::terminate_handler GgmlTerminateHandler::s_previous_handler = nullptr;

// The single static instance that manages the handler's lifetime.
// Its constructor is called on library load, and its destructor on unload.
static GgmlTerminateHandler g_ggml_terminate_handler_instance;

} // namespace
