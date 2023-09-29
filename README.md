# MASHA - Multi-Agent Subgoal Hierarchy Algorithm

Данный репозиторий посвящен разработке иерархического метода для мультиагентного обучения с подкреплением, используя обнаружение подцелей - MASHA :woman_astronaut:.

Это сочетание многоуровневого иерархического обучения с обнаружением промежуточных целей и мультиагентного обучения с подкреплением с воспроизведением ретроспективного опыта, которое позволяет множеству агентов эффективно обучаться в сложных средах, в том числе в средах с редкими вознаграждениями. Демонстрация результатов проводится в одной из таких сред внутри стратегической игры StarCraft II, кроме того приводится сравнение с другими современными подходами. Метод MASHA разработан в парадигме централизованного обучения с децентрализованным исполнением, что позволяет достичь баланса между координацией и автономностью агентов.

### Архитектура MASHA
Верхний уровень (Уровень 0) получает начальную командную цель $g_0$, она одинакова для всех $n$ агентов. Исходя из наблюдений агентов и начальной цели, модули исполнителей верхнего уровня генерируют командные подцели $g_1 = (g_{1,1}, ..., g_{1,n})$ для нижестоящего уровня в качестве своих действий.

<p align="center">
  <img src="/resources/arch.png" width=70% height=70%>
</p>

### Мультиагентная среда в StarCraft II
Для проведения сравнительных экспериментов по мультиагентному обучению с подкреплением была выбрана популярная программная библиотека SMAC, предоставляющая возможность децентрализованного управления множеством агентов в среде стратегической компьютерной игры StarCraft II. Библиотека SMAC сегодня является одним из главных международных экспериментальных стендов для объективного анализа мультиагентных методов машинного обучения.

<p align="center">
  <img src="/resources/env_overview.png" width=60% />
</p>

<p align="center">
  <img src="/resources/2v10_hide.gif" width=45% title="cima_2v2" />
  <img src="/resources/subgoals.gif" width=45% title="cima_2v10" />
</p>
