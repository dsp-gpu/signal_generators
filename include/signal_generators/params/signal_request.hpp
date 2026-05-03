#pragma once

/**
 * @brief Типы запросов на генерацию сигналов: enum + POD-параметры + variant-обёртка.
 *
 * @note Тип B (simple structs + enum): только данные + сравнения + один helper getter.
 *       Содержимое:
 *         - SignalKind     — enum: CW / LFM / NOISE / FORM_SIGNAL
 *         - NoiseType      — enum: WHITE / GAUSSIAN
 *         - CwParams       — f0, phase, amplitude, complex_iq, freq_step (multi-beam)
 *         - LfmParams      — f_start, f_end, amplitude, complex_iq + GetChirpRate(duration)
 *         - NoiseParams    — type, power (дисперсия для Gaussian), seed
 *         - SignalRequest  — kind + system + variant<CwParams, LfmParams, NoiseParams, FormParams>
 *       FormParams — отдельный сложный тип в form_params.hpp (с парсером).
 *       operator==/!= нужны для кеширования генераторов в SignalService (перекомпиляция
 *       ядра только при смене params).
 *
 * История:
 *   - Создан:  2026-02-13
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#include <signal_generators/params/system_sampling.hpp>
#include <signal_generators/params/form_params.hpp>
#include <cstdint>
#include <variant>

namespace signal_gen {

// ════════════════════════════════════════════════════════════════════════════
// Типы сигналов
// ════════════════════════════════════════════════════════════════════════════

enum class SignalKind { CW, LFM, NOISE, FORM_SIGNAL };

// ════════════════════════════════════════════════════════════════════════════
// Параметры CW (continuous wave — синусоида)
// ════════════════════════════════════════════════════════════════════════════

struct CwParams {
    double f0 = 100.0;           ///< Частота (Hz)
    double phase = 0.0;          ///< Начальная фаза (rad)
    double amplitude = 1.0;      ///< Амплитуда
    bool complex_iq = true;      ///< true = exp(j*phase), false = real only (imag=0)

    // Для multi-beam: freq_i = f0 + i * freq_step
    double freq_step = 0.0;      ///< Шаг частоты между лучами (Hz), 0 = все одинаковые

    bool operator==(const CwParams& o) const {
        return f0 == o.f0 && phase == o.phase && amplitude == o.amplitude
            && complex_iq == o.complex_iq && freq_step == o.freq_step;
    }
    bool operator!=(const CwParams& o) const { return !(*this == o); }
};

// ════════════════════════════════════════════════════════════════════════════
// Параметры LFM (linear frequency modulation — chirp)
// ════════════════════════════════════════════════════════════════════════════

struct LfmParams {
    double f_start = 100.0;      ///< Начальная частота (Hz)
    double f_end = 500.0;        ///< Конечная частота (Hz)
    double amplitude = 1.0;      ///< Амплитуда
    bool complex_iq = true;      ///< Комплексный выход

    /**
     * @brief Возвращает скорость изменения частоты chirp: k = (f_end − f_start) / duration.
     *
     * @param duration Длительность сигнала в секундах.
     *
     * @return Chirp rate в Гц/с.
     *   @test_check std::isfinite(result)
     */
    double GetChirpRate(double duration) const {
        return (f_end - f_start) / duration;
    }

    bool operator==(const LfmParams& o) const {
        return f_start == o.f_start && f_end == o.f_end
            && amplitude == o.amplitude && complex_iq == o.complex_iq;
    }
    bool operator!=(const LfmParams& o) const { return !(*this == o); }
};

// ════════════════════════════════════════════════════════════════════════════
// Параметры шума
// ════════════════════════════════════════════════════════════════════════════

enum class NoiseType { WHITE, GAUSSIAN };

struct NoiseParams {
    NoiseType type = NoiseType::GAUSSIAN;
    double power = 1.0;          ///< Мощность шума (дисперсия для Gaussian)
    uint64_t seed = 0;           ///< 0 = random seed

    bool operator==(const NoiseParams& o) const {
        return type == o.type && power == o.power && seed == o.seed;
    }
    bool operator!=(const NoiseParams& o) const { return !(*this == o); }
};

// ════════════════════════════════════════════════════════════════════════════
// Единый запрос
// ════════════════════════════════════════════════════════════════════════════

struct SignalRequest {
    SignalKind kind;
    SystemSampling system;
    std::variant<CwParams, LfmParams, NoiseParams, FormParams> params;
};

} // namespace signal_gen
