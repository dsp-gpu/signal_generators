#pragma once

/**
 * @brief CollectOrRelease — header-only утилита для профилирования OpenCL событий.
 *
 * @note Тип B (technical header): один inline-template в namespace signal_gen.
 *       Поведение:
 *         - prof_events == nullptr → production-путь: clReleaseEvent(ev)
 *         - prof_events != nullptr → сохранить пару (name, ev) для отчёта
 *       Правило: вызывать ПОСЛЕ того как event использован как wait-dependency
 *       следующего kernel'а — иначе release освободит event до использования.
 *       Используется во всех cpp-генераторах модуля (CW/LFM/Noise/Form) для
 *       устранения копипасты «if profiling … else release».
 *
 * История:
 *   - Создан:  2026-03-09
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#include <CL/cl.h>
#include <utility>
#include <vector>

namespace signal_gen {

/**
 * @brief Сохранить cl_event для профилирования или освободить (production path)
 *
 * @tparam ProfEvents  Тип вектора событий (vector<pair<const char*, cl_event>>)
 * @param ev           OpenCL event (может быть nullptr — тогда ничего не делает)
 * @param name         Имя стадии (строковый литерал)
 * @param prof_events  nullptr → production (clReleaseEvent); иначе → сохранить в вектор
 */
template <typename ProfEvents>
inline void CollectOrRelease(cl_event ev, const char* name, ProfEvents* prof_events) {
  if (!ev) return;
  if (prof_events)
    prof_events->push_back({name, ev});
  else
    clReleaseEvent(ev);
}

}  // namespace signal_gen
