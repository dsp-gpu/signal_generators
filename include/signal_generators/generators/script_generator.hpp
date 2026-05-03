#pragma once

// ============================================================================
// ScriptGenerator — компилятор текстового DSL в OpenCL kernel
//
// ЧТО:    Парсер мини-языка [Params]/[Defs]/[Signal] в OpenCL kernel:
//         юзер описывает формулу сигнала текстом, генератор формирует
//         OpenCL-исходник, компилирует через cl_program и запускает на
//         GPU. Встроенные переменные: ID (антенна), T (сэмпл), M_PI_F.
//         Поддерживает реальный (res) и комплексный (res_re/res_im) выход.
//
// ЗАЧЕМ:  Гибкая radar-симуляция: набор импульсов произвольной формы
//         (CW/LFM/паузы/комбинации) задаётся в JSON/text-конфиге, без
//         перекомпиляции C++. Удобно для исследований, прототипирования
//         новых сигналов, batch-тестов с разными параметрами от Python.
//
// ПОЧЕМУ: - Runtime-компиляция через clBuildProgram — пользовательский
//           DSL транслируется в OpenCL C на лету.
//         - Move-only: cl_program/queue/context уникальны на инстанс.
//         - backend не владеет — caller гарантирует переживание объекта.
//         - GetKernelSource() — для отладки сгенерированного kernel.
//         - OpenCL-вариант (legacy). ROCm-вариант: ScriptGeneratorROCm.
//
// Использование:
//   signal_gen::ScriptGenerator gen(backend);
//   gen.LoadScript(R"(
//       [Params] ANTENNAS = 8  POINTS = 4096
//       [Defs]   var_W = 0.1 + (float)ID * 0.005
//       [Signal] res = sin(var_W * (float)T);
//   )");
//   cl_mem out = gen.Generate();
//   // ... использовать ...
//   clReleaseMemObject(out);
//
// История:
//   - Создан: 2026-02-13 (legacy OpenCL-ветка)
// ============================================================================

#include <core/interface/i_backend.hpp>
#include <CL/cl.h>
#include <string>
#include <vector>
#include <complex>
#include <cstdint>

namespace signal_gen {

/// Parsed script parameters from [Params] section
struct ScriptParams {
    uint32_t antennas = 1;     ///< Number of antennas/beams
    uint32_t points   = 1024;  ///< Number of samples per antenna
};

/// Result of parsing a script
struct ParsedScript {
    ScriptParams params;
    std::vector<std::string> defs;          ///< Lines from [Defs]
    std::vector<std::string> signal_lines;  ///< Lines from [Signal]
    bool has_res     = false;  ///< 'res' variable detected
    bool has_res_re  = false;  ///< 'res_re' variable detected
    bool has_res_im  = false;  ///< 'res_im' variable detected
};

/**
 * @class ScriptGenerator
 * @brief Компилятор текстового DSL в OpenCL kernel с исполнением на GPU.
 *
 * @note Move-only: cl_program/queue/context уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note OpenCL-вариант. ROCm-аналог: ScriptGeneratorROCm.
 * @see signal_gen::ScriptGeneratorROCm
 *
 * @code
 * ScriptGenerator gen(backend);
 * gen.LoadScript(R"(
 *     [Params]
 *     ANTENNAS = 8
 *     POINTS = 4096
 *     [Defs]
 *     var_W = 0.1 + (float)ID * 0.005
 *     [Signal]
 *     res = sin(var_W * (float)T);
 * )");
 *
 * cl_mem gpu_buf = gen.Generate();
 * // ... use gpu_buf ...
 * clReleaseMemObject(gpu_buf);
 * @endcode
 */
class ScriptGenerator {
public:
    explicit ScriptGenerator(drv_gpu_lib::IBackend* backend);
    ~ScriptGenerator();

    // No copy
    ScriptGenerator(const ScriptGenerator&) = delete;
    ScriptGenerator& operator=(const ScriptGenerator&) = delete;

    // Move
    ScriptGenerator(ScriptGenerator&& other) noexcept;
    ScriptGenerator& operator=(ScriptGenerator&& other) noexcept;

    /**
     * @brief Parse and compile script from string
     * @param script_text Script in [Params]/[Defs]/[Signal] format
     * @throws std::runtime_error on parse or compilation failure
     */
    void LoadScript(const std::string& script_text);

    /**
     * @brief Parse and compile script from file
     * @param file_path Path to .signal or .txt file
     *   @test { values=["/tmp/test_config.json"] }
     * @throws std::runtime_error if file cannot be read
     */
    void LoadFile(const std::string& file_path);

    /**
     * @brief Generate signal on GPU
     * @return cl_mem buffer [antennas * points * sizeof(complex<float>)]
     * @note Caller must release via clReleaseMemObject()
     *   @test_check result != nullptr (требуется LoadScript/LoadFile перед Generate)
     */
    cl_mem Generate();

    /**
     * @brief Generate and read back to CPU
     * @return Vector of complex samples [antennas * points]
     *   @test_check result.size() == script_.params.antennas * script_.params.points
     */
    std::vector<std::complex<float>> GenerateToCpu();

    /// Get parsed params
    uint32_t GetAntennas() const { return script_.params.antennas; }
    uint32_t GetPoints() const { return script_.params.points; }
    size_t   GetTotalSamples() const { return static_cast<size_t>(script_.params.antennas) * script_.params.points; }

    /// Get generated OpenCL kernel source (for debugging)
    const std::string& GetKernelSource() const { return kernel_source_; }

    /// Check if script is loaded and compiled
    bool IsReady() const { return program_ != nullptr; }

private:
    ParsedScript ParseScript(const std::string& text);
    std::string  GenerateKernelSource(const ParsedScript& script);
    std::string  PrepareExpression(const std::string& line);
    void CompileKernel(const std::string& source);
    void ReleaseGpuResources();

    static std::string Trim(const std::string& s);
    static std::string ToUpper(const std::string& s);

    drv_gpu_lib::IBackend* backend_ = nullptr;
    cl_context context_       = nullptr;
    cl_command_queue queue_    = nullptr;
    cl_device_id device_      = nullptr;
    cl_program program_       = nullptr;

    ParsedScript script_;
    std::string  kernel_source_;
};

} // namespace signal_gen
