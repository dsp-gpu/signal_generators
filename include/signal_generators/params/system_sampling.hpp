#pragma once

/**
 * @brief POD-структура системных параметров дискретизации сигнала.
 *
 * @note Тип B (simple struct): только данные, без поведения / валидации.
 *       Общая для всех генераторов модуля (CW / LFM / Noise) — задаёт fs и length.
 *       Передаётся по значению в ISignalGenerator::GenerateToCpu / GenerateToGpu.
 *
 * История:
 *   - Создан:  2026-02-13
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#include <cstddef>

namespace signal_gen {

/**
 * @struct SystemSampling
 * @brief Параметры дискретизации — общие для всех генераторов.
 */
struct SystemSampling {
    double fs = 1000.0;    ///< Частота дискретизации (Hz)
    size_t length = 1024;  ///< Количество отсчётов на луч
};

} // namespace signal_gen
